"""
scrape_ats.py
Scrapes game-level ATS data from covers.com for NCAAB.
Pulls every tournament team's game log for 2006 through current season.
Saves to data/ats_game_log.csv
"""

import requests
import pandas as pd
import time
import re
from pathlib import Path
from datetime import datetime
from bs4 import BeautifulSoup

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

TODAY = datetime.utcnow().strftime("%Y%m%d")

SEASONS = [
    "2006-2007",
    "2007-2008",
    "2008-2009",
    "2009-2010",
    "2010-2011",
    "2011-2012",
    "2012-2013",
    "2013-2014",
    "2014-2015",
    "2015-2016",
    "2016-2017",
    "2017-2018",
    "2018-2019",
    "2019-2020",
    "2020-2021",
    "2021-2022",
    "2022-2023",
    "2023-2024",
    "2024-2025",
    "2025-2026",
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Referer": "https://www.covers.com/",
}

BASE_URL = "https://www.covers.com"


def get_team_list(season: str) -> list[dict]:
    """Scrape the team list page and return team name + slug pairs."""
    url = f"{BASE_URL}/sport/basketball/ncaab/statistics/team-betting/{season}"
    resp = requests.get(url, headers=HEADERS, timeout=20)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    teams = []
    # Find all team links in the stats table
    for a in soup.select("table a[href*='/teams/main/']"):
        href = a.get("href", "")
        name = a.get_text(strip=True)
        if href and name:
            teams.append({"name": name, "slug": href, "season": season})

    print(f"  Found {len(teams)} teams for {season}")
    return teams


def get_team_game_log(team_slug: str, season: str) -> list[dict]:
    """Scrape individual team's game-by-game ATS results."""
    # Convert team page URL to game log URL
    # e.g. /sport/basketball/ncaab/teams/main/florida-gators
    # → /sport/basketball/ncaab/teams/main/florida-gators/results/{season}
    url = f"{BASE_URL}{team_slug}/results/{season}"
    
    try:
        resp = requests.get(url, headers=HEADERS, timeout=20)
        if resp.status_code != 200:
            return []
        soup = BeautifulSoup(resp.text, "html.parser")
    except Exception as e:
        print(f"    Error fetching {url}: {e}")
        return []

    games = []
    # Find results table rows
    table = soup.select_one("table.table-results, table#team-results, .covers-CoversTable")
    if not table:
        # Try any table with game data
        tables = soup.find_all("table")
        for t in tables:
            headers_row = t.find("tr")
            if headers_row and any(
                kw in headers_row.get_text().lower()
                for kw in ["spread", "ats", "score", "opponent"]
            ):
                table = t
                break

    if not table:
        return []

    rows = table.find_all("tr")[1:]  # Skip header
    for row in rows:
        cols = [td.get_text(strip=True) for td in row.find_all(["td", "th"])]
        if len(cols) < 5:
            continue

        try:
            game = {
                "season":   season,
                "raw_cols": cols,
            }

            # Parse what we can from the columns
            # Covers format varies but generally:
            # Date | Opponent | Score | ATS Result | Spread | O/U | O/U Result
            text = " ".join(cols)

            # Extract score pattern like "W 85-72" or "L 67-74"
            score_match = re.search(r"([WL])\s+(\d+)-(\d+)", text)
            if score_match:
                game["result"]     = score_match.group(1)
                game["team_score"] = int(score_match.group(2))
                game["opp_score"]  = int(score_match.group(3))
                game["margin"]     = game["team_score"] - game["opp_score"]

            # Extract spread
            spread_match = re.search(r"([+-]?\d+\.?\d*)", " ".join(cols[3:6]))
            if spread_match:
                game["spread"] = float(spread_match.group(1))

            # ATS result
            for col in cols:
                if col.upper() in ["W", "L", "P"]:
                    game["ats_result"] = col.upper()
                    break

            # Is it a tournament game?
            game["is_tournament"] = any(
                kw in text.lower()
                for kw in ["ncaa", "tournament", "march madness", "first round",
                           "second round", "sweet 16", "elite eight", "final four"]
            )

            game["cols_count"] = len(cols)
            games.append(game)

        except Exception:
            continue

    return games


def scrape_tournament_teams_only(season: str, target_teams: list[str]) -> list[dict]:
    """
    More targeted approach: scrape only known tournament teams.
    Uses covers.com team URL pattern directly.
    """
    # Covers uses lowercase hyphenated team names
    def to_slug(name: str) -> str:
        name = name.lower()
        name = re.sub(r"[^a-z0-9\s-]", "", name)
        name = re.sub(r"\s+", "-", name.strip())
        return name

    results = []
    for team in target_teams:
        slug = to_slug(team)
        url = f"{BASE_URL}/sport/basketball/ncaab/teams/main/{slug}-*/results/{season}"

        # Try a few common slug patterns
        candidate_slugs = [
            f"/sport/basketball/ncaab/teams/main/{slug}",
            f"/sport/basketball/ncaab/teams/main/{slug.replace('state', 'st')}",
            f"/sport/basketball/ncaab/teams/main/{slug}-wildcats",
            f"/sport/basketball/ncaab/teams/main/{slug}-tigers",
            f"/sport/basketball/ncaab/teams/main/{slug}-gators",
            f"/sport/basketball/ncaab/teams/main/{slug}-cyclones",
        ]

        for candidate in candidate_slugs[:1]:  # Start with most likely
            games = get_team_game_log(candidate, season)
            if games:
                for g in games:
                    g["team"] = team
                results.extend(games)
                print(f"    {team}: {len(games)} games")
                break

        time.sleep(0.5)  # Be polite

    return results


# ─────────────────────────────────────────────
# MAIN APPROACH: Use the team list page to get correct slugs
# ─────────────────────────────────────────────

TOURNAMENT_TEAMS = [
    "Auburn", "Alabama", "Florida", "Tennessee", "Duke", "Houston",
    "Iowa State", "St. John's", "Michigan State", "Texas Tech", "Wisconsin",
    "Kentucky", "Arizona", "Marquette", "Purdue", "Connecticut",
    "Gonzaga", "Kansas", "Maryland", "Missouri", "BYU", "Texas A&M",
    "Mississippi State", "Baylor", "Saint Mary's", "VCU", "Vanderbilt",
    "North Carolina", "Ole Miss", "Memphis", "Colorado State", "Drake",
    "Arkansas", "New Mexico", "Utah State", "Georgia", "Creighton",
    "Oklahoma", "UCLA", "Cincinnati", "Louisville", "TCU", "Illinois",
    "Michigan", "Oregon", "Clemson", "UC San Diego", "McNeese State",
    "Liberty", "High Point", "Bryant", "Nebraska Omaha", "Wofford",
    "Norfolk State", "Alabama State", "SIU Edwardsville",
    "Mount St. Mary's", "UNC Wilmington", "Grand Canyon",
    "Troy", "Lipscomb", "Akron", "Robert Morris", "Montana",
]


def run_scrape():
    """Main scrape function — get ATS game logs for tournament teams."""
    master_path = DATA_DIR / "ats_game_log.csv"
    existing = pd.read_csv(master_path) if master_path.exists() else pd.DataFrame()

    all_games = []

    for season in SEASONS:
        print(f"\nScraping {season}...")

        # Step 1: Get the team list with correct slugs
        try:
            teams = get_team_list(season)
            time.sleep(1)
        except Exception as e:
            print(f"  Could not get team list for {season}: {e}")
            continue

        # Step 2: Filter to tournament-relevant teams
        tourney_teams = [
            t for t in teams
            if any(
                tt.lower()[:5] in t["name"].lower()
                for tt in TOURNAMENT_TEAMS
            )
        ]
        print(f"  Filtered to {len(tourney_teams)} tournament-relevant teams")

        # Step 3: Get game logs
        for team in tourney_teams:
            try:
                games = get_team_game_log(team["slug"], season)
                for g in games:
                    g["team"]   = team["name"]
                    g["season"] = season
                all_games.extend(games)
                print(f"    {team['name']}: {len(games)} games")
                time.sleep(0.75)  # Polite delay
            except Exception as e:
                print(f"    Error for {team['name']}: {e}")
                continue

    if all_games:
        new_df = pd.DataFrame(all_games)
        # Drop the raw_cols column for clean storage
        if "raw_cols" in new_df.columns:
            new_df = new_df.drop(columns=["raw_cols"])
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined = combined.drop_duplicates()
        combined.to_csv(master_path, index=False)
        print(f"\n✅ ATS game log saved: {len(combined)} total rows")
        print(f"   Teams covered: {combined['team'].nunique() if 'team' in combined.columns else 'N/A'}")
        print(f"   Seasons: {combined['season'].unique() if 'season' in combined.columns else 'N/A'}")
    else:
        print("\n⚠️  No games scraped — covers.com may require JavaScript rendering")
        print("   Consider using Playwright or Selenium for dynamic content")

    return master_path


if __name__ == "__main__":
    import os
    import sys

    print(f"\n{'='*50}")
    print(f"ATS Scraper — {TODAY}")
    print(f"{'='*50}")

    historical_done = DATA_DIR / "ats_historical_complete.flag"
    mode = os.environ.get("SCRAPE_MODE", "daily")

    if mode == "historical" or not historical_done.exists():
        print("MODE: Full historical scrape — all seasons 2006-2026")
        print("This runs once only and takes 10-20 mins...")
        run_scrape()
        historical_done.write_text(f"Completed: {TODAY}")
        print("Historical scrape complete — future daily runs will only refresh current season")
    else:
        print("MODE: Daily update — current season only")
        SEASONS[:] = ["2025-2026"]
        run_scrape()
        print("Daily ATS update complete")

