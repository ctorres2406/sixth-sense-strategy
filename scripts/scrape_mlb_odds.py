"""
scrape_mlb_odds.py
==================
Sixth Sense Strategy — MLB Live Odds + Props Fetcher
Hits The Odds API for moneyline, run line, totals, and top props.
Merges live odds into data/mlb/picks.csv.
Writes player props to data/mlb/props.csv.

Run after generate_mlb_picks.py:
    python scrape_mlb_odds.py

Requires:
    ODDS_API_KEY environment variable (or edit KEY below)
    pip install requests pandas --break-system-packages
"""

import os
import json
import requests
import pandas as pd
from datetime import date, datetime

# ── Config ────────────────────────────────────────────────────────────────────
ODDS_API_KEY = os.environ.get("ODDS_API_KEY", "YOUR_KEY_HERE")
BASE_URL     = "https://api.the-odds-api.com/v4"
SPORT        = "baseball_mlb"
REGION       = "us"             # us = DraftKings, FanDuel, BetMGM, etc.
BOOKMAKERS   = "draftkings,fanduel,betmgm"

DATA_DIR     = "data/mlb"
PICKS_CSV    = f"{DATA_DIR}/picks.csv"
PROPS_CSV    = f"{DATA_DIR}/props.csv"
TODAY        = date.today().strftime("%Y%m%d")

# Markets to fetch for game-level odds
GAME_MARKETS = "h2h,spreads,totals"

# Player prop markets (uses more API credits — comment out to save)
PROP_MARKETS = [
    "batter_home_runs",
    "batter_hits",
    "batter_rbis",
    "batter_strikeouts",
    "pitcher_strikeouts",
    "pitcher_hits_allowed",
    "pitcher_earned_runs",
]

# Minimum model-vs-market edge to include prop in output
MIN_PROP_EDGE = 0.05

# Preferred book order for line display
PREFERRED_BOOKS = ["draftkings", "fanduel", "betmgm", "espnbet", "caesars", "pointsbet"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_best_odds(bookmakers: list, market_key: str, outcome_name: str) -> tuple[str, str]:
    """Return (price_str, book_name) for the best price across bookmakers."""
    best_price = None
    best_book  = ""

    for bm in bookmakers:
        for market in bm.get("markets", []):
            if market.get("key") != market_key:
                continue
            for outcome in market.get("outcomes", []):
                if outcome.get("name", "").lower() != outcome_name.lower():
                    continue
                price = outcome.get("price")
                if price is None:
                    continue
                # Convert to int for comparison
                try:
                    p = int(price)
                except (ValueError, TypeError):
                    continue
                # For moneyline: higher = better (less juice, more return)
                # Simple: pick highest
                if best_price is None or p > best_price:
                    best_price = p
                    best_book  = bm.get("title", "")

    if best_price is None:
        return "", ""

    price_str = f"+{best_price}" if best_price > 0 else str(best_price)
    return price_str, best_book


def norm(s: str) -> str:
    return str(s).lower().replace(" ", "").replace(".", "")


def match_team(api_name: str, csv_name: str) -> bool:
    """Check if an API team name matches a CSV team name."""
    an = norm(api_name)
    cn = norm(csv_name)
    # Check substring in either direction (handles "Los Angeles Dodgers" vs "Dodgers")
    return an in cn or cn in an or an[:6] == cn[:6]


# ── Fetch game odds ───────────────────────────────────────────────────────────

def fetch_game_odds() -> list[dict]:
    """Pull moneyline, run line, totals for all today's games."""
    print("[1/3] Fetching game odds from The Odds API...")

    url = f"{BASE_URL}/sports/{SPORT}/odds"
    params = {
        "apiKey":  ODDS_API_KEY,
        "regions": REGION,
        "markets": GAME_MARKETS,
        "oddsFormat": "american",
        "bookmakers": BOOKMAKERS,
    }

    try:
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
        remaining = r.headers.get("x-requests-remaining", "?")
        used = r.headers.get("x-requests-used", "?")
        print(f"  API credits used: {used} | remaining: {remaining}")
    except requests.exceptions.HTTPError as e:
        if r.status_code == 401:
            raise ValueError("Invalid ODDS_API_KEY. Set ODDS_API_KEY env var.")
        raise

    print(f"  Received {len(data)} games from API")
    return data


def merge_odds_into_picks(picks_df: pd.DataFrame, api_games: list) -> pd.DataFrame:
    """Match API games to picks rows by team name and fill in odds fields."""
    print(f"[2/3] Merging odds into {len(picks_df)} picks...")

    filled = 0
    for i, row in picks_df.iterrows():
        t1 = str(row.get("team1", ""))  # home team
        t2 = str(row.get("team2", ""))  # away team

        # Find matching API game
        api_match = None
        for ag in api_games:
            ht = ag.get("home_team", "")
            at = ag.get("away_team", "")
            if match_team(ht, t1) and match_team(at, t2):
                api_match = ag
                break
            # Also try reversed (ESPN occasionally flips home/away)
            if match_team(at, t1) and match_team(ht, t2):
                api_match = ag
                break

        if api_match is None:
            continue

        bms = api_match.get("bookmakers", [])
        ml_pick = str(row.get("ml_pick", ""))
        ats_pick = str(row.get("ats_pick", ""))
        ou_pick = str(row.get("ou_pick", ""))

        # ── Moneyline odds for our pick ───────────────────────────────────────
        ml_odds, _ = get_best_odds(bms, "h2h", ml_pick)
        if ml_odds:
            picks_df.at[i, "ml_odds"] = ml_odds

        # ── Run line odds (always ~-110 each side, but get actual) ────────────
        rl_odds, _ = get_best_odds(bms, "spreads", ats_pick.split(" -")[0].split(" +")[0])
        if rl_odds:
            picks_df.at[i, "ats_odds"] = rl_odds   # bonus col, not in FINAL_COLS but useful

        # ── Total line + O/U odds ─────────────────────────────────────────────
        # Get the actual posted total from the API (override our default estimate)
        for bm in bms:
            for market in bm.get("markets", []):
                if market.get("key") == "totals":
                    for outcome in market.get("outcomes", []):
                        pt = outcome.get("point")
                        if pt:
                            picks_df.at[i, "total_line"] = float(pt)
                            # Update ou_pick to reflect actual line
                            direction = "Over" if "Over" in ou_pick else "Under"
                            picks_df.at[i, "ou_pick"] = f"{direction} {pt}"
                            break
                    break

        ou_dir = "Over" if "Over" in ou_pick else "Under"
        ou_odds_val, _ = get_best_odds(bms, "totals", ou_dir)
        if ou_odds_val:
            picks_df.at[i, "ou_odds"] = ou_odds_val

        filled += 1

    print(f"  Filled odds for {filled}/{len(picks_df)} games")
    return picks_df


# ── Fetch player props ────────────────────────────────────────────────────────

def fetch_props(api_games: list) -> pd.DataFrame:
    """Pull player prop odds for all today's games."""
    print("[3/3] Fetching player props...")

    if ODDS_API_KEY == "YOUR_KEY_HERE":
        print("  Skipping props — no API key set")
        return pd.DataFrame()

    all_props = []
    for ag in api_games[:5]:  # Limit to first 5 games to conserve credits
        event_id = ag.get("id", "")
        home = ag.get("home_team", "")
        away = ag.get("away_team", "")

        for market in PROP_MARKETS:
            url = f"{BASE_URL}/sports/{SPORT}/events/{event_id}/odds"
            params = {
                "apiKey":     ODDS_API_KEY,
                "regions":    REGION,
                "markets":    market,
                "oddsFormat": "american",
                "bookmakers": BOOKMAKERS,
            }
            try:
                r = requests.get(url, params=params, timeout=10)
                if r.status_code == 422:
                    continue  # Market not available
                r.raise_for_status()
                data = r.json()
            except Exception:
                continue

            for bm in data.get("bookmakers", []):
                for mkt in bm.get("markets", []):
                    if mkt.get("key") != market:
                        continue
                    for outcome in mkt.get("outcomes", []):
                        price = outcome.get("price", 0)
                        point = outcome.get("point", "")
                        direction = outcome.get("name", "")  # "Over" / "Under" / player name
                        desc = outcome.get("description", direction)

                        all_props.append({
                            "game_date":  TODAY,
                            "home_team":  home,
                            "away_team":  away,
                            "market":     market,
                            "player":     desc,
                            "direction":  direction,
                            "line":       point,
                            "odds":       f"+{price}" if int(price or 0) > 0 else str(price),
                            "book":       bm.get("title", ""),
                            "pick":       "",        # filled below
                            "confidence": 0,
                            "result":     "pending",
                        })

    if not all_props:
        print("  No props returned")
        return pd.DataFrame()

    df = pd.DataFrame(all_props)

    # Simple prop model: flag Over lines at -115 or better as value
    # (Real prop model coming in v2 — this is a heuristic placeholder)
    df["confidence"] = df["odds"].apply(lambda o: _prop_conf_heuristic(o))
    df["pick"] = df.apply(
        lambda r: f"{r['player']} {r['direction']} {r['line']}" if r["confidence"] >= 55 else "",
        axis=1
    )

    # Sort by confidence desc, drop low-conf
    df = df[df["confidence"] >= 55].sort_values("confidence", ascending=False)

    print(f"  {len(df)} prop bets with confidence >= 55%")
    return df


def _prop_conf_heuristic(odds_str: str) -> int:
    """Heuristic: price closer to even = more uncertain, steep fav = higher conf."""
    try:
        o = int(str(odds_str).replace("+", ""))
    except (ValueError, TypeError):
        return 50
    # Very rough: -200 = 67% implied = ~63% conf after juice, -130 = 56%, even = 50%
    if o <= -200: return 65
    if o <= -160: return 62
    if o <= -130: return 58
    if o <= -110: return 55
    return 50


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print(f"SIXTH SENSE STRATEGY — MLB ODDS   [{TODAY}]")
    print("=" * 55)

    # Load picks
    if not os.path.exists(PICKS_CSV):
        raise FileNotFoundError(f"{PICKS_CSV} not found. Run generate_mlb_picks.py first.")

    picks_df = pd.read_csv(PICKS_CSV, dtype=str)
    print(f"  Loaded {len(picks_df)} picks from {PICKS_CSV}")

    if picks_df.empty:
        print("  No picks to enrich. Exiting.")
        return

    # Fetch game odds
    api_games = fetch_game_odds()

    # Merge odds into picks
    picks_df = merge_odds_into_picks(picks_df, api_games)

    # Recalculate edge for any pick that now has odds
    def calc_edge(row):
        ml_odds = str(row.get("ml_odds", ""))
        ml_conf = int(row.get("ml_confidence", 0) or 0)
        if not ml_odds:
            return ""
        try:
            o = int(ml_odds.replace("+", ""))
            imp = (100 / (o + 100)) if o > 0 else (abs(o) / (abs(o) + 100))
            edge = ml_conf / 100 - imp
            return f"+{int(edge*100)}" if edge >= 0 else str(int(edge*100))
        except Exception:
            return ""

    picks_df["ml_edge"] = picks_df.apply(calc_edge, axis=1)

    # Drop raw probability columns before final write
    drop_cols = [c for c in picks_df.columns if c.startswith("_")]
    picks_df = picks_df.drop(columns=drop_cols, errors="ignore")

    # Save enriched picks
    picks_df.to_csv(PICKS_CSV, index=False)
    print(f"\n  ✓ Updated {PICKS_CSV} with live odds")

    # Fetch + save props
    props_df = fetch_props(api_games)
    if not props_df.empty:
        props_df.to_csv(PROPS_CSV, index=False)
        print(f"  ✓ Saved {len(props_df)} props to {PROPS_CSV}")

    # Print odds summary
    print("\n" + "=" * 55)
    print("ODDS SUMMARY")
    print("=" * 55)
    for _, row in picks_df.iterrows():
        ml = row.get("ml_odds", "—")
        ou = row.get("ou_odds", "—")
        edge = row.get("ml_edge", "")
        print(
            f"  {str(row.get('team2',''))[:18]:18s} @ {str(row.get('team1',''))[:18]:18s}  "
            f"ML: {str(ml):6s}  O/U: {str(ou):6s}  edge: {edge}"
        )

    remaining = "—"  # Already printed above from header
    print(f"\n✓ Complete. picks.csv and props.csv ready for dashboard.")


if __name__ == "__main__":
    main()
