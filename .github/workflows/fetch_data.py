"""
fetch_data.py
Pulls ESPN scoreboard + odds data daily and saves to /data folder.
Runs via GitHub Actions on a schedule.
"""

import json
import requests
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import os

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

TODAY = datetime.utcnow().strftime("%Y%m%d")
YESTERDAY = (datetime.utcnow() - timedelta(days=1)).strftime("%Y%m%d")


# ─────────────────────────────────────────────
# 1. ESPN SCOREBOARD
# ─────────────────────────────────────────────
def fetch_espn_scores(date_str: str) -> list[dict]:
    """Fetch all D1 NCAAB games for a given date (YYYYMMDD)."""
    url = (
        "https://site.api.espn.com/apis/site/v2/sports/basketball"
        f"/mens-college-basketball/scoreboard?limit=200&dates={date_str}&groups=50"
    )
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    games = []
    for event in data.get("events", []):
        comp = event["competitions"][0]

        # Skip unfinished games
        if not comp["status"]["type"].get("completed", False):
            continue

        teams = comp["competitors"]
        if len(teams) != 2:
            continue

        t1, t2 = teams[0], teams[1]
        winner = t1 if t1.get("winner") else t2
        loser  = t2 if t1.get("winner") else t1

        def stat(team, name):
            for s in team.get("statistics", []):
                if s["name"] == name:
                    try: return float(s["displayValue"])
                    except: return None
            return None

        def record(team):
            for r in team.get("records", []):
                if r.get("type") == "total":
                    return r.get("summary", "")
            return ""

        tournament_note = ""
        if event.get("notes"):
            tournament_note = event["notes"][0].get("headline", "")

        games.append({
            "date":            date_str,
            "game_id":         event["id"],
            "game_name":       event["name"],
            "winner":          winner["team"]["displayName"],
            "winner_abbr":     winner["team"]["abbreviation"],
            "loser":           loser["team"]["displayName"],
            "loser_abbr":      loser["team"]["abbreviation"],
            "winner_score":    int(winner.get("score", 0)),
            "loser_score":     int(loser.get("score", 0)),
            "margin":          int(winner.get("score", 0)) - int(loser.get("score", 0)),
            "is_tournament":   comp["type"].get("abbreviation") == "TRNMNT",
            "is_neutral":      comp.get("neutralSite", False),
            "is_conf":         comp.get("conferenceCompetition", False),
            "tournament_note": tournament_note,
            "conference":      event.get("groups", {}).get("shortName", ""),
            "winner_record":   record(winner),
            "loser_record":    record(loser),
            "winner_fg_pct":   stat(winner, "fieldGoalPct"),
            "winner_3p_pct":   stat(winner, "threePointFieldGoalPct"),
            "winner_ft_pct":   stat(winner, "freeThrowPct"),
            "loser_fg_pct":    stat(loser,  "fieldGoalPct"),
            "loser_3p_pct":    stat(loser,  "threePointFieldGoalPct"),
            "loser_ft_pct":    stat(loser,  "freeThrowPct"),
        })

    return games


def update_scores():
    """Fetch yesterday + today, append to master CSV."""
    master_path = DATA_DIR / "espn_scores.csv"
    existing = pd.read_csv(master_path) if master_path.exists() else pd.DataFrame()

    new_rows = []
    for date in [YESTERDAY, TODAY]:
        try:
            rows = fetch_espn_scores(date)
            new_rows.extend(rows)
            print(f"  ESPN {date}: {len(rows)} games fetched")
        except Exception as e:
            print(f"  ESPN {date} ERROR: {e}")

    if new_rows:
        new_df = pd.DataFrame(new_rows)
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["game_id", "date"])
        combined.to_csv(master_path, index=False)
        print(f"  Scores master: {len(combined)} total rows")

    return master_path


# ─────────────────────────────────────────────
# 2. ODDS — THE-ODDS-API
# ─────────────────────────────────────────────
def fetch_odds(api_key: str) -> list[dict]:
    """Fetch current NCAAB odds from The-Odds-API."""
    url = "https://api.the-odds-api.com/v4/sports/basketball_ncaab/odds"
    params = {
        "apiKey":   api_key,
        "regions":  "us",
        "markets":  "h2h,spreads,totals",
        "oddsFormat": "american",
        "bookmakers": "pinnacle,draftkings,fanduel,betmgm,caesars",
    }
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    rows = []
    for game in data:
        home = game["home_team"]
        away = game["away_team"]
        commence = game["commence_time"]

        for book in game.get("bookmakers", []):
            book_name = book["key"]
            for market in book.get("markets", []):
                mkt = market["key"]
                for outcome in market.get("outcomes", []):
                    rows.append({
                        "fetched_at":   TODAY,
                        "commence_time": commence,
                        "home_team":    home,
                        "away_team":    away,
                        "bookmaker":    book_name,
                        "market":       mkt,
                        "team":         outcome["name"],
                        "price":        outcome["price"],
                        "point":        outcome.get("point"),
                    })

    return rows


def update_odds(api_key: str):
    """Fetch current odds and append to master CSV."""
    if not api_key:
        print("  No ODDS_API_KEY set — skipping odds fetch")
        return

    master_path = DATA_DIR / "live_odds.csv"
    existing = pd.read_csv(master_path) if master_path.exists() else pd.DataFrame()

    try:
        rows = fetch_odds(api_key)
        new_df = pd.DataFrame(rows)
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined = combined.drop_duplicates(
            subset=["fetched_at", "commence_time", "home_team", "away_team", "bookmaker", "market", "team"]
        )
        combined.to_csv(master_path, index=False)
        print(f"  Odds master: {len(combined)} total rows, {len(rows)} new today")
    except Exception as e:
        print(f"  Odds ERROR: {e}")


# ─────────────────────────────────────────────
# 3. MOMENTUM PROFILES
# ─────────────────────────────────────────────
TOURNEY_TEAMS = [
    "Auburn", "Alabama", "Florida", "Tennessee", "Duke", "Houston",
    "Iowa State", "St. John's", "Michigan State", "Texas Tech", "Wisconsin",
    "Kentucky", "Arizona", "Marquette", "Purdue", "Connecticut",
    "Gonzaga", "Kansas", "Maryland", "Missouri", "BYU", "Texas A&M",
    "Mississippi State", "Baylor", "Saint Mary's", "VCU", "Vanderbilt",
    "North Carolina", "Ole Miss", "Memphis", "Colorado State", "Drake",
    "Arkansas", "New Mexico", "Utah State", "Georgia", "Creighton",
    "Oklahoma", "UCLA", "Cincinnati", "Louisville", "TCU", "Illinois",
    "Michigan", "Oregon", "Clemson",
]


def build_momentum_profiles() -> pd.DataFrame:
    """Build per-team conf tournament momentum from scores master."""
    scores_path = DATA_DIR / "espn_scores.csv"
    if not scores_path.exists():
        print("  No scores file yet — skipping momentum")
        return pd.DataFrame()

    df = pd.read_csv(scores_path)
    conf_games = df[df["is_tournament"] == True].copy()

    profiles = []
    for team in TOURNEY_TEAMS:
        wins   = conf_games[conf_games["winner"].str.contains(team, case=False, na=False)]
        losses = conf_games[conf_games["loser"].str.contains(team, case=False, na=False)]

        n_w = len(wins)
        n_l = len(losses)

        avg_win_margin  = wins["margin"].mean()  if n_w > 0 else 0.0
        avg_loss_margin = losses["margin"].mean() if n_l > 0 else 0.0

        momentum = (n_w * avg_win_margin) - (n_l * avg_loss_margin * 2)

        profiles.append({
            "team":               team,
            "conf_wins":          n_w,
            "conf_losses":        n_l,
            "conf_games":         n_w + n_l,
            "conf_champ":         n_l == 0 and n_w >= 2,
            "avg_win_margin":     round(avg_win_margin, 1),
            "avg_loss_margin":    round(avg_loss_margin, 1),
            "momentum_score":     round(momentum, 1),
            "hot":                n_w >= 2 and avg_win_margin >= 10,
            "cold":               n_l > 0 and avg_loss_margin >= 10,
            "last_updated":       TODAY,
        })

    out_df = pd.DataFrame(profiles).sort_values("momentum_score", ascending=False)
    out_path = DATA_DIR / "momentum_profiles.csv"
    out_df.to_csv(out_path, index=False)
    print(f"  Momentum profiles saved: {out_path}")
    return out_df


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print(f"\n{'='*50}")
    print(f"March Madness Data Fetcher — {TODAY}")
    print(f"{'='*50}")

    print("\n[1/3] Fetching ESPN scores...")
    update_scores()

    print("\n[2/3] Fetching live odds...")
    api_key = os.environ.get("ODDS_API_KEY", "")
    update_odds(api_key)

    print("\n[3/3] Building momentum profiles...")
    profiles = build_momentum_profiles()
    if not profiles.empty:
        print("\nTop 10 by momentum:")
        print(profiles[["team","conf_wins","conf_losses","momentum_score","hot","cold"]].head(10).to_string(index=False))

    print(f"\n✅ Done — {TODAY}")
