"""
fetch_data.py
Pulls ESPN scoreboard + odds data daily and saves to /data folder.
Runs via GitHub Actions on a schedule.

Outputs:
  espn_raw.csv         — completed games (all D1, used for momentum profiles)
  tourney_schedule.csv — upcoming + live NCAA tournament games (feed for generate_picks.py)
  live_odds.csv        — current odds from The-Odds-API
  momentum_profiles.csv
"""

import json
import requests
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import os

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

TODAY     = datetime.utcnow().strftime("%Y%m%d")
YESTERDAY = (datetime.utcnow() - timedelta(days=1)).strftime("%Y%m%d")

ROUND_LABEL = {
    "First Round":        "R64",
    "Second Round":       "R32",
    "Sweet 16":           "S16",
    "Elite Eight":        "E8",
    "Final Four":         "FF",
    "National Championship": "CHAMP",
    # ESPN sometimes uses these variants
    "1st Round":          "R64",
    "2nd Round":          "R32",
}


# ─────────────────────────────────────────────
# TOURNEY TEAMS — dynamic from bracket.csv
# ─────────────────────────────────────────────
def load_tourney_teams() -> list:
    bracket_path = DATA_DIR / "bracket.csv"
    if bracket_path.exists():
        try:
            df = pd.read_csv(bracket_path)
            teams = set()
            if "team1" in df.columns: teams.update(df["team1"].dropna().astype(str))
            if "team2" in df.columns: teams.update(df["team2"].dropna().astype(str))
            teams = [t for t in teams if t and t.lower() not in ("nan","tbd","")]
            if teams:
                print(f"  Loaded {len(teams)} teams from bracket.csv")
                return sorted(teams)
        except Exception as e:
            print(f"  Warning: could not read bracket.csv: {e}")

    print("  bracket.csv not found — using static team list")
    return [
        "Auburn","Alabama","Florida","Tennessee","Duke","Houston",
        "Iowa State","St. John's","Michigan State","Texas Tech","Wisconsin",
        "Kentucky","Arizona","Marquette","Purdue","Connecticut",
        "Gonzaga","Kansas","Maryland","Missouri","BYU","Texas A&M",
        "Mississippi State","Baylor","Saint Mary's","VCU","Vanderbilt",
        "North Carolina","Ole Miss","Memphis","Colorado State","Drake",
        "Arkansas","New Mexico","Utah State","Georgia","Creighton",
        "Oklahoma","UCLA","Cincinnati","Louisville","TCU","Illinois",
        "Michigan","Oregon","Clemson","Virginia","Iowa","Nebraska",
        "Texas","Miami FL","Villanova","Wisconsin","Akron","SMU",
    ]


# ─────────────────────────────────────────────
# 1a. COMPLETED GAMES  (espn_raw.csv)
# ─────────────────────────────────────────────
def fetch_espn_scores(date_str: str) -> list:
    """Fetch completed D1 NCAAB games for a given date."""
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
    """Fetch yesterday + today completed games, append to espn_raw.csv."""
    master_path = DATA_DIR / "espn_raw.csv"
    existing    = pd.read_csv(master_path) if master_path.exists() else pd.DataFrame()

    rows = []
    for date in [YESTERDAY, TODAY]:
        try:
            rows.extend(fetch_espn_scores(date))
        except Exception as e:
            print(f"  ESPN scores ERROR for {date}: {e}")

    if rows:
        new_df   = pd.DataFrame(rows)
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["game_id"])
        combined.to_csv(master_path, index=False)
        print(f"  Raw scores (espn_raw.csv): {len(combined)} total rows, {len(rows)} new")
    else:
        print("  No new completed games fetched")


# ─────────────────────────────────────────────
# 1b. UPCOMING TOURNAMENT GAMES (tourney_schedule.csv)
# ─────────────────────────────────────────────
def fetch_upcoming_tourney(date_str: str) -> list:
    """
    Fetch ALL games (completed + scheduled) for a date and return
    NCAA tournament games with their status, teams, seeds, round, region.
    """
    url = (
        "https://site.api.espn.com/apis/site/v2/sports/basketball"
        f"/mens-college-basketball/scoreboard?limit=200&dates={date_str}&groups=50"
    )
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    games = []
    for event in data.get("events", []):
        comp   = event["competitions"][0]
        status = comp["status"]["type"]

        # Only care about NCAA tournament games
        is_tourney = comp["type"].get("abbreviation") == "TRNMNT"
        notes = event.get("notes", [])
        note_text = notes[0].get("headline", "") if notes else ""
        if not is_tourney and "NCAA" not in note_text:
            continue

        teams = comp.get("competitors", [])
        if len(teams) != 2:
            continue

        # Parse team info
        def parse_team(t):
            team_obj = t.get("team", {})
            seed = None
            # seed is sometimes in curatedRank or in notes
            if "curatedRank" in t:
                seed = t["curatedRank"].get("current")
            if seed is None:
                # try the order attribute as fallback
                try:
                    seed = int(t.get("order", 0)) or None
                except:
                    seed = None
            return {
                "name":  team_obj.get("displayName", ""),
                "abbr":  team_obj.get("abbreviation", ""),
                "score": t.get("score", ""),
                "seed":  seed,
                "winner": t.get("winner", False),
            }

        ta = parse_team(teams[0])
        tb = parse_team(teams[1])

        # Determine round from ESPN headline/note
        round_str = "R64"
        for key, val in ROUND_LABEL.items():
            if key.lower() in note_text.lower():
                round_str = val
                break

        # Region from note text (e.g. "NCAA Men's Basketball Tournament - East Regional")
        region = ""
        for r in ["East","West","South","Midwest","Final Four","National"]:
            if r.lower() in note_text.lower():
                region = r if r not in ("Final Four","National") else ("Final Four" if r=="Final Four" else "Championship")
                break

        # Status
        state = status.get("name", "").lower()  # scheduled, in_progress, final
        completed = status.get("completed", False)
        game_time = event.get("date", "")  # ISO 8601

        # Score info
        score_a = ta["score"]
        score_b = tb["score"]
        winner_name = ""
        if completed:
            winner_name = ta["name"] if ta["winner"] else tb["name"]

        games.append({
            "game_id":    event["id"],
            "date":       date_str,
            "game_time":  game_time,
            "status":     state,            # 'scheduled', 'in_progress', 'final'
            "completed":  completed,
            "round":      round_str,
            "region":     region,
            "team1":      ta["name"],
            "seed1":      ta["seed"],
            "score1":     score_a if completed else "",
            "team2":      tb["name"],
            "seed2":      tb["seed"],
            "score2":     score_b if completed else "",
            "winner":     winner_name,
            "note":       note_text,
        })

    return games


def update_tourney_schedule():
    """
    Fetch upcoming + live + recent NCAA tournament games.
    Looks at yesterday, today, and next 4 days to capture full tournament windows.
    Writes tourney_schedule.csv — the feed for generate_picks.py.
    """
    master_path = DATA_DIR / "tourney_schedule.csv"
    existing    = pd.read_csv(master_path) if master_path.exists() else pd.DataFrame()

    all_games = []
    base = datetime.utcnow()
    # Check -1 to +4 days to capture games already played and coming up
    for delta in range(-1, 5):
        date_str = (base + timedelta(days=delta)).strftime("%Y%m%d")
        try:
            games = fetch_upcoming_tourney(date_str)
            if games:
                print(f"    {date_str}: {len(games)} tournament games")
            all_games.extend(games)
        except Exception as e:
            print(f"    ESPN tourney fetch ERROR for {date_str}: {e}")

    if not all_games:
        print("  No tournament games found in schedule window")
        return

    new_df = pd.DataFrame(all_games)

    # Merge with existing: prefer new data (has fresher scores/status)
    if not existing.empty:
        combined = pd.concat([new_df, existing], ignore_index=True)
        combined = combined.drop_duplicates(subset=["game_id"], keep="first")
    else:
        combined = new_df

    combined = combined.sort_values(["round", "region", "game_time"])
    combined.to_csv(master_path, index=False)
    total = len(combined)
    upcoming = len(combined[~combined["completed"].astype(bool)])
    done = len(combined[combined["completed"].astype(bool)])
    print(f"  tourney_schedule.csv: {total} total games ({done} final, {upcoming} upcoming/live)")


# ─────────────────────────────────────────────
# 2. ODDS
# ─────────────────────────────────────────────
def fetch_odds(api_key: str) -> list:
    url = "https://api.the-odds-api.com/v4/sports/basketball_ncaab/odds"
    params = {
        "apiKey":     api_key,
        "regions":    "us",
        "markets":    "h2h,spreads",
        "oddsFormat": "american",
    }
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    rows = []
    for game in data:
        for bk in game.get("bookmakers", []):
            for market in bk.get("markets", []):
                for outcome in market.get("outcomes", []):
                    rows.append({
                        "fetched_at":    TODAY,
                        "commence_time": game.get("commence_time"),
                        "home_team":     game.get("home_team"),
                        "away_team":     game.get("away_team"),
                        "bookmaker":     bk["key"],
                        "market":        market["key"],
                        "team":          outcome["name"],
                        "price":         outcome["price"],
                        "point":         outcome.get("point"),
                    })
    return rows


def update_odds(api_key: str):
    if not api_key:
        print("  No ODDS_API_KEY set — skipping odds fetch")
        return
    master_path = DATA_DIR / "live_odds.csv"
    existing    = pd.read_csv(master_path) if master_path.exists() else pd.DataFrame()
    try:
        rows     = fetch_odds(api_key)
        new_df   = pd.DataFrame(rows)
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined = combined.drop_duplicates(
            subset=["fetched_at","commence_time","home_team","away_team","bookmaker","market","team"]
        )
        combined.to_csv(master_path, index=False)
        print(f"  Odds (live_odds.csv): {len(combined)} rows, {len(rows)} new")
    except Exception as e:
        print(f"  Odds ERROR: {e}")


# ─────────────────────────────────────────────
# 3. MOMENTUM PROFILES
# ─────────────────────────────────────────────
def build_momentum_profiles():
    scores_path = DATA_DIR / "espn_raw.csv"
    if not scores_path.exists():
        print("  No espn_raw.csv yet — skipping momentum")
        return pd.DataFrame()

    tourney_teams = load_tourney_teams()
    df         = pd.read_csv(scores_path)
    conf_games = df[df["is_tournament"] == True].copy()

    profiles = []
    for team in tourney_teams:
        wins   = conf_games[conf_games["winner"].str.contains(team, case=False, na=False)]
        losses = conf_games[conf_games["loser"].str.contains(team,  case=False, na=False)]
        n_w = len(wins); n_l = len(losses)
        avg_win_margin  = wins["margin"].mean()   if n_w > 0 else 0.0
        avg_loss_margin = losses["margin"].mean() if n_l > 0 else 0.0
        momentum = (n_w * avg_win_margin) - (n_l * avg_loss_margin * 2)
        profiles.append({
            "team":            team,
            "conf_wins":       n_w,
            "conf_losses":     n_l,
            "conf_games":      n_w + n_l,
            "conf_champ":      n_l == 0 and n_w >= 2,
            "avg_win_margin":  round(avg_win_margin,  1),
            "avg_loss_margin": round(avg_loss_margin, 1),
            "momentum_score":  round(momentum, 1),
            "hot":             n_w >= 2 and avg_win_margin >= 10,
            "cold":            n_l > 0  and avg_loss_margin >= 10,
            "last_updated":    TODAY,
        })

    out_df   = pd.DataFrame(profiles).sort_values("momentum_score", ascending=False)
    out_path = DATA_DIR / "momentum_profiles.csv"
    out_df.to_csv(out_path, index=False)
    print(f"  Momentum profiles: {len(profiles)} teams")
    return out_df


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print(f"\n{'='*52}")
    print(f"March Madness Data Fetcher — {TODAY}")
    print(f"{'='*52}")

    print("\n[1/4] Fetching completed ESPN scores...")
    update_scores()

    print("\n[2/4] Fetching tournament schedule (upcoming + live)...")
    update_tourney_schedule()

    print("\n[3/4] Fetching live odds...")
    api_key = os.environ.get("ODDS_API_KEY", "")
    update_odds(api_key)

    print("\n[4/4] Building momentum profiles...")
    profiles = build_momentum_profiles()
    if not profiles.empty:
        print("\nTop 10 by momentum:")
        print(profiles[["team","conf_wins","conf_losses","momentum_score","hot","cold"]].head(10).to_string(index=False))

    print(f"\n✅ Done — {TODAY}")
