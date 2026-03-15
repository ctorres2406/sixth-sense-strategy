"""
fetch_mlb_data.py
=================
Sixth Sense Strategy — MLB Daily Data Fetch
Pulls today's MLB schedule, probable starters, and team stats.
Writes to data/mlb/raw_games.json for generate_mlb_picks.py to consume.

Run daily (8am ET via GitHub Actions):
    python fetch_mlb_data.py
"""

import os
import json
import time
import requests
import numpy as np
import pandas as pd
from datetime import datetime, date
import warnings

warnings.filterwarnings("ignore")

try:
    import pybaseball as pyb
    from pybaseball import pitching_stats_range, batting_stats, pitching_stats
    pyb.cache.enable()
except ImportError:
    raise ImportError("pip install pybaseball --break-system-packages")

# ── Config ────────────────────────────────────────────────────────────────────
OUTPUT_DIR = "data/mlb"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TODAY = date.today().strftime("%Y%m%d")
CURRENT_YEAR = date.today().year

ESPN_SCHEDULE_URL = (
    f"https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard"
    f"?dates={TODAY}&limit=20"
)

ESPN_TEAM_URL = "https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/teams"

# Park factors (update annually — 2024 values)
PARK_FACTORS = {
    "COL": 118, "BOS": 109, "CIN": 107, "TEX": 107, "PHI": 106,
    "BAL": 105, "ATL": 104, "NYY": 104, "LAA": 103, "MIL": 103,
    "TOR": 103, "CHC": 102, "HOU": 102, "MIN": 102, "NYM": 102,
    "CLE": 101, "STL": 101, "WSH": 101, "ARI": 100, "DET": 100,
    "KC":  100, "LAD": 100, "MIA": 100, "PIT": 100, "SEA": 100,
    "CWS":  99, "OAK":  99, "SD":   99, "SF":   97, "TB":   96,
}

# ESPN abbreviation → our standard abbreviation
ESPN_TO_STD = {
    "CHW": "CWS", "WSH": "WSH", "KC": "KC", "SD": "SD",
    "SF": "SF", "TB": "TB", "NYY": "NYY", "NYM": "NYM",
}


def std_abbr(abbr: str) -> str:
    abbr = (abbr or "").upper().strip()
    return ESPN_TO_STD.get(abbr, abbr)


# ── ESPN Schedule ─────────────────────────────────────────────────────────────

def fetch_espn_schedule() -> list[dict]:
    """Pull today's games from ESPN scoreboard API."""
    print(f"[1/4] Fetching ESPN schedule for {TODAY}...")
    try:
        r = requests.get(ESPN_SCHEDULE_URL, timeout=15)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"  ERROR fetching ESPN schedule: {e}")
        return []

    games = []
    for event in data.get("events", []):
        comp = event.get("competitions", [{}])[0]
        competitors = comp.get("competitors", [])
        if len(competitors) < 2:
            continue

        # Identify home/away
        home = next((c for c in competitors if c.get("homeAway") == "home"), competitors[0])
        away = next((c for c in competitors if c.get("homeAway") == "away"), competitors[1])

        home_abbr = std_abbr(home.get("team", {}).get("abbreviation", ""))
        away_abbr = std_abbr(away.get("team", {}).get("abbreviation", ""))

        home_name = home.get("team", {}).get("displayName", home_abbr)
        away_name = away.get("team", {}).get("displayName", away_abbr)

        # Scores (if game started/final)
        home_score = home.get("score", "")
        away_score = away.get("score", "")

        # Status
        status_obj  = comp.get("status", {}).get("type", {})
        status_name = status_obj.get("name", "STATUS_SCHEDULED")
        status_desc = status_obj.get("description", "Scheduled")

        # Game time (ET)
        game_date = event.get("date", "")
        try:
            dt = datetime.fromisoformat(game_date.replace("Z", "+00:00"))
            # Convert UTC → ET (rough -4 or -5)
            et_hour = (dt.hour - 4) % 24
            ampm = "PM" if et_hour >= 12 else "AM"
            disp_hour = et_hour % 12 or 12
            game_time_et = f"{disp_hour}:{dt.minute:02d} {ampm} ET"
        except Exception:
            game_time_et = ""

        # Probable starters (ESPN sometimes includes these in notes)
        pitcher_home = ""
        pitcher_away = ""
        for note in comp.get("notes", []):
            text = note.get("headline", "")
            if "SP:" in text or "Probable" in text.lower():
                # Try to parse "Home SP: Name / Away SP: Name"
                parts = text.split("/")
                for p in parts:
                    if "home" in p.lower() and "sp:" in p.lower():
                        pitcher_home = p.split(":")[-1].strip()
                    elif "away" in p.lower() and "sp:" in p.lower():
                        pitcher_away = p.split(":")[-1].strip()

        # Winner (if final)
        winner = ""
        if "FINAL" in status_name.upper():
            hs, as_ = int(home_score or 0), int(away_score or 0)
            winner = home_abbr if hs > as_ else away_abbr

        game = {
            "game_id":     event.get("id", ""),
            "game_date":   TODAY,
            "game_time":   game_time_et,
            "status":      status_desc,
            "status_code": status_name,
            "home_abbr":   home_abbr,
            "away_abbr":   away_abbr,
            "home_name":   home_name,
            "away_name":   away_name,
            "home_score":  home_score,
            "away_score":  away_score,
            "winner":      winner,
            "pitcher_home": pitcher_home,
            "pitcher_away": pitcher_away,
            "park_factor": PARK_FACTORS.get(home_abbr, 100),
        }
        games.append(game)

    print(f"  Found {len(games)} games today")
    return games


# ── Team Stats (season-to-date via pybaseball) ────────────────────────────────

def fetch_team_stats(season: int) -> dict:
    """Pull season-to-date batting and pitching stats for all teams."""
    print(f"[2/4] Fetching {season} team stats via pybaseball...")

    stats = {}

    # Batting
    try:
        bat = batting_stats(season, season, ind=1)
        bat_cols = ["Team", "wOBA", "wRC+", "BB%", "K%", "AVG", "OBP", "SLG"]
        bat = bat[[c for c in bat_cols if c in bat.columns]].copy()

        for _, row in bat.iterrows():
            t = str(row.get("Team", "")).upper()
            if t not in stats:
                stats[t] = {}
            stats[t]["bat_woba"]  = float(row.get("wOBA", 0) or 0)
            stats[t]["bat_wrc"]   = float(row.get("wRC+", 100) or 100)
            stats[t]["bat_bb_pct"]= float(str(row.get("BB%", 0)).replace("%","") or 0)
            stats[t]["bat_k_pct"] = float(str(row.get("K%",  0)).replace("%","") or 0)
        print(f"  Batting: {len(bat)} teams")
    except Exception as e:
        print(f"  WARNING: batting stats failed — {e}")

    # Pitching
    try:
        pit = pitching_stats(season, season, ind=1)
        pit_cols = ["Team", "ERA", "FIP", "xFIP", "WHIP", "K/9", "BB/9", "HR/9"]
        pit = pit[[c for c in pit_cols if c in pit.columns]].copy()

        for _, row in pit.iterrows():
            t = str(row.get("Team", "")).upper()
            if t not in stats:
                stats[t] = {}
            stats[t]["pit_era"]  = float(row.get("ERA",  4.5) or 4.5)
            stats[t]["pit_fip"]  = float(row.get("FIP",  4.3) or 4.3)
            stats[t]["pit_xfip"] = float(row.get("xFIP", 4.3) or 4.3)
            stats[t]["pit_whip"] = float(row.get("WHIP", 1.3) or 1.3)
            stats[t]["pit_k9"]   = float(row.get("K/9",  8.5) or 8.5)
            stats[t]["pit_bb9"]  = float(row.get("BB/9", 3.0) or 3.0)
        print(f"  Pitching: {len(pit)} teams")
    except Exception as e:
        print(f"  WARNING: pitching stats failed — {e}")

    return stats


# ── Pitcher Stats ─────────────────────────────────────────────────────────────

def fetch_pitcher_stats(season: int) -> pd.DataFrame:
    """Pull individual pitcher stats for probable starter lookup."""
    print(f"[3/4] Fetching {season} individual pitcher stats...")
    try:
        df = pitching_stats(season, season, ind=0)
        # Keep starters (GS > 0)
        if "GS" in df.columns:
            df = df[df["GS"] > 0].copy()
        keep = ["Name", "Team", "ERA", "FIP", "xFIP", "WHIP",
                "K/9", "BB/9", "HR/9", "IP", "GS"]
        df = df[[c for c in keep if c in df.columns]].copy()
        print(f"  {len(df)} individual starters found")
        return df
    except Exception as e:
        print(f"  WARNING: pitcher stats failed — {e}")
        return pd.DataFrame()


def match_pitcher(name: str, pitcher_df: pd.DataFrame) -> dict:
    """Fuzzy match a pitcher name to stats row."""
    if pitcher_df.empty or not name:
        return {}

    name_lower = name.lower().strip()
    # Try exact match first
    for _, row in pitcher_df.iterrows():
        if str(row.get("Name","")).lower().strip() == name_lower:
            return {
                "era":  float(row.get("ERA",  4.5) or 4.5),
                "fip":  float(row.get("FIP",  4.3) or 4.3),
                "xfip": float(row.get("xFIP", 4.3) or 4.3),
                "whip": float(row.get("WHIP", 1.3) or 1.3),
                "k9":   float(row.get("K/9",  8.5) or 8.5),
                "bb9":  float(row.get("BB/9", 3.0) or 3.0),
            }

    # Fuzzy: last name match
    last = name_lower.split()[-1] if name_lower else ""
    for _, row in pitcher_df.iterrows():
        if last and last in str(row.get("Name","")).lower():
            return {
                "era":  float(row.get("ERA",  4.5) or 4.5),
                "fip":  float(row.get("FIP",  4.3) or 4.3),
                "xfip": float(row.get("xFIP", 4.3) or 4.3),
                "whip": float(row.get("WHIP", 1.3) or 1.3),
                "k9":   float(row.get("K/9",  8.5) or 8.5),
                "bb9":  float(row.get("BB/9", 3.0) or 3.0),
            }

    return {}  # No match — generate_picks will use team averages


# ── Assemble game objects ─────────────────────────────────────────────────────

def assemble_games(games: list, team_stats: dict, pitcher_df: pd.DataFrame) -> list:
    """Enrich each game with team stats and pitcher stats."""
    print(f"[4/4] Assembling {len(games)} game records...")

    # Pybaseball uses full team names; map to abbrs we use
    # Build a simple abbr lookup from both spellings
    def get_team(abbr):
        abbr = abbr.upper()
        # Try direct match
        if abbr in team_stats:
            return team_stats[abbr]
        # Try common aliases
        aliases = {
            "CWS": ["CHW", "WHITE SOX"], "SD": ["SDP", "PADRES"],
            "SF": ["SFG", "GIANTS"], "TB": ["TBR", "RAYS"],
            "WSH": ["WSN", "NATIONALS"], "KC": ["KCR", "ROYALS"],
            "MIA": ["FLA", "MARLINS"], "CLE": ["IND", "GUARDIANS"],
        }
        for k, v in aliases.items():
            if abbr in v or abbr == k:
                direct = team_stats.get(k, team_stats.get(v[0], {}))
                if direct:
                    return direct
        return {}

    # League average fallbacks
    LEAGUE_AVG_BAT = {"bat_woba": 0.317, "bat_wrc": 100, "bat_bb_pct": 8.5, "bat_k_pct": 22.0}
    LEAGUE_AVG_PIT = {"pit_era": 4.20, "pit_fip": 4.10, "pit_xfip": 4.10,
                      "pit_whip": 1.30, "pit_k9": 8.7, "pit_bb9": 3.1}
    LEAGUE_AVG_SP  = {"era": 4.20, "fip": 4.10, "xfip": 4.10,
                      "whip": 1.30, "k9": 8.7, "bb9": 3.1}

    enriched = []
    for g in games:
        h = g["home_abbr"]
        a = g["away_abbr"]

        ht = get_team(h)
        at = get_team(a)

        # Merge league average for missing values
        ht = {**LEAGUE_AVG_BAT, **LEAGUE_AVG_PIT, **ht}
        at = {**LEAGUE_AVG_BAT, **LEAGUE_AVG_PIT, **at}

        # Probable starter stats
        hp = match_pitcher(g["pitcher_home"], pitcher_df)
        ap = match_pitcher(g["pitcher_away"], pitcher_df)
        hp = {**LEAGUE_AVG_SP, **hp}
        ap = {**LEAGUE_AVG_SP, **ap}

        pf = g.get("park_factor", 100)

        enriched.append({
            **g,

            # Home team stats
            "h_woba":     ht["bat_woba"],
            "h_wrc":      ht["bat_wrc"],
            "h_bb_pct":   ht["bat_bb_pct"],
            "h_k_pct":    ht["bat_k_pct"],
            "h_pit_era":  ht["pit_era"],
            "h_pit_fip":  ht["pit_fip"],
            "h_pit_xfip": ht["pit_xfip"],
            "h_pit_whip": ht["pit_whip"],
            "h_pit_k9":   ht["pit_k9"],
            "h_pit_bb9":  ht["pit_bb9"],

            # Away team stats
            "a_woba":     at["bat_woba"],
            "a_wrc":      at["bat_wrc"],
            "a_bb_pct":   at["bat_bb_pct"],
            "a_k_pct":    at["bat_k_pct"],
            "a_pit_era":  at["pit_era"],
            "a_pit_fip":  at["pit_fip"],
            "a_pit_xfip": at["pit_xfip"],
            "a_pit_whip": at["pit_whip"],
            "a_pit_k9":   at["pit_k9"],
            "a_pit_bb9":  at["pit_bb9"],

            # Probable starter stats
            "h_sp_era":  hp["era"],
            "h_sp_fip":  hp["fip"],
            "h_sp_xfip": hp["xfip"],
            "h_sp_whip": hp["whip"],
            "h_sp_k9":   hp["k9"],
            "h_sp_bb9":  hp["bb9"],

            "a_sp_era":  ap["era"],
            "a_sp_fip":  ap["fip"],
            "a_sp_xfip": ap["xfip"],
            "a_sp_whip": ap["whip"],
            "a_sp_k9":   ap["k9"],
            "a_sp_bb9":  ap["bb9"],

            # Differentials
            "woba_diff": ht["bat_woba"] - at["bat_woba"],
            "sp_era_diff": ap["era"] - hp["era"],    # positive = home SP advantage
            "sp_fip_diff": ap["fip"] - hp["fip"],

            # Park / context
            "park_factor": pf,
            "is_home": 1,
            "season": CURRENT_YEAR,
        })

    return enriched


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print(f"SIXTH SENSE STRATEGY — MLB FETCH  [{TODAY}]")
    print("=" * 55)

    games = fetch_espn_schedule()
    if not games:
        print("No games today — writing empty output.")
        with open(f"{OUTPUT_DIR}/raw_games.json", "w") as f:
            json.dump([], f)
        return

    team_stats = fetch_team_stats(CURRENT_YEAR)
    pitcher_df = fetch_pitcher_stats(CURRENT_YEAR)

    enriched = assemble_games(games, team_stats, pitcher_df)

    out_path = f"{OUTPUT_DIR}/raw_games.json"
    with open(out_path, "w") as f:
        json.dump(enriched, f, indent=2, default=str)

    print(f"\n✓ Wrote {len(enriched)} games to {out_path}")
    print(f"  Next: run generate_mlb_picks.py")


if __name__ == "__main__":
    main()
