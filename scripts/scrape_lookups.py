"""
scrape_lookups.py
-----------------
Scrapes all free data sources needed to update the model lookup CSVs for 2026.
Run this once after Selection Sunday (or as soon as possible before March 19).

Sources:
  - barttorvik.com  → KenPom_Barttorvik.csv, Resumes.csv,
                      INT___KenPom___Height.csv, ats_team_profiles.csv
  - evanmiya.com    → EvanMiya.csv
  - sports-reference.com → AP_Poll_Data.csv, Coach_Results.csv,
                           REF___Current_NCAAM_Coaches.csv

Usage:
  python scripts/scrape_lookups.py

Outputs all CSVs to data/lookups/ appending 2026 rows.
Existing rows from prior years are preserved.
"""

import re
import time
import warnings
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore")

YEAR       = 2026
LOOKUP_DIR = Path(__file__).parent.parent / "data" / "lookups"
LOOKUP_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}

def get(url, params=None, retries=3, delay=2.0):
    """Polite fetch with retries."""
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, headers=HEADERS, timeout=20)
            resp.raise_for_status()
            time.sleep(delay)
            return resp
        except Exception as e:
            print(f"    Attempt {attempt+1} failed: {e}")
            time.sleep(delay * (attempt + 1))
    raise RuntimeError(f"Failed to fetch {url} after {retries} attempts")


def load_existing(fname):
    """Load existing CSV, return empty DataFrame if missing."""
    p = LOOKUP_DIR / fname
    if p.exists():
        return pd.read_csv(p, low_memory=False)
    return pd.DataFrame()


def save_merged(fname, existing, new_df, year_col="YEAR", team_col="TEAM"):
    """
    Merge new 2026 rows into existing CSV.
    Drops any existing 2026 rows first to avoid duplication on re-runs.
    """
    p = LOOKUP_DIR / fname
    if not existing.empty and year_col in existing.columns:
        existing = existing[existing[year_col].astype(str) != str(YEAR)]
    combined = pd.concat([existing, new_df], ignore_index=True)
    combined.to_csv(p, index=False)
    print(f"  ✅ Saved {fname} ({len(new_df)} new rows, {len(combined)} total)")
    return combined


# ═══════════════════════════════════════════════════════════════════════════════
# 1. BARTTORVIK — T-Rank ratings → KenPom_Barttorvik.csv
# ═══════════════════════════════════════════════════════════════════════════════
def scrape_barttorvik_trank():
    """
    Scrapes Barttorvik T-Rank page for current season ratings.
    Maps columns to match KenPom_Barttorvik.csv schema.

    Barttorvik BADJ EM ≈ KenPom KADJ EM (r > 0.97 historically).
    We populate both KADJ EM and BADJ EM from Barttorvik's adjusted EM.
    """
    print("\n[1/7] Scraping Barttorvik T-Rank...")

    url = "https://barttorvik.com/trank.php"
    params = {"year": YEAR, "json": 1}

    try:
        resp = get(url, params=params)
        data = resp.json()
    except Exception:
        # Fallback: try the HTML table
        print("  JSON endpoint failed, trying HTML table...")
        return scrape_barttorvik_trank_html()

    rows = []
    for team in data:
        try:
            rows.append({
                "YEAR":     YEAR,
                "TEAM":     team.get("team", ""),
                "CONF":     team.get("conf", ""),
                "SEED":     team.get("seed", np.nan),
                "KADJ EM":  float(team.get("adjoe", 0)) - float(team.get("adjde", 0)),
                "KADJ O":   float(team.get("adjoe", 0)),
                "KADJ D":   float(team.get("adjde", 0)),
                "BADJ EM":  float(team.get("adjoe", 0)) - float(team.get("adjde", 0)),
                "BARTHAG":  float(team.get("barthag", np.nan)),
                "TALENT":   float(team.get("talent", np.nan)),
                "EXP":      float(team.get("exp", np.nan)),
                "TEMPO":    float(team.get("tempo", np.nan)),
                "RANK":     int(team.get("rk", 999)),
            })
        except Exception as e:
            print(f"    Row error: {e}")
            continue

    df = pd.DataFrame(rows)
    print(f"  Fetched {len(df)} teams from Barttorvik JSON")

    existing = load_existing("KenPom_Barttorvik.csv")
    return save_merged("KenPom_Barttorvik.csv", existing, df)


def scrape_barttorvik_trank_html():
    """HTML fallback for Barttorvik T-Rank."""
    url = f"https://barttorvik.com/trank.php?year={YEAR}"
    resp = get(url)
    tables = pd.read_html(resp.text)
    if not tables:
        raise RuntimeError("No tables found on Barttorvik T-Rank page")

    df_raw = tables[0]
    df_raw.columns = [str(c).strip() for c in df_raw.columns]
    print(f"  HTML table columns: {list(df_raw.columns)}")

    # Flexible column mapping — Barttorvik occasionally renames columns
    col_map = {
        "team": "TEAM", "Team": "TEAM",
        "conf": "CONF", "Conf": "CONF",
        "adjoe": "KADJ O", "AdjOE": "KADJ O", "Adj OE": "KADJ O",
        "adjde": "KADJ D", "AdjDE": "KADJ D", "Adj DE": "KADJ D",
        "barthag": "BARTHAG", "Barthag": "BARTHAG",
        "talent": "TALENT", "Talent": "TALENT",
        "exp": "EXP", "Exp": "EXP",
        "tempo": "TEMPO", "Tempo": "TEMPO",
        "seed": "SEED", "Seed": "SEED",
        "rk": "RANK", "Rk": "RANK", "Rank": "RANK",
    }
    df_raw = df_raw.rename(columns={k: v for k, v in col_map.items() if k in df_raw.columns})

    if "KADJ O" in df_raw.columns and "KADJ D" in df_raw.columns:
        df_raw["KADJ EM"] = pd.to_numeric(df_raw["KADJ O"], errors="coerce") - \
                            pd.to_numeric(df_raw["KADJ D"], errors="coerce")
        df_raw["BADJ EM"] = df_raw["KADJ EM"]

    df_raw["YEAR"] = YEAR
    df_raw = df_raw[df_raw["TEAM"].notna() & (df_raw["TEAM"].astype(str) != "Team")]

    existing = load_existing("KenPom_Barttorvik.csv")
    return save_merged("KenPom_Barttorvik.csv", existing, df_raw)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. BARTTORVIK — Resumes → Resumes.csv
# ═══════════════════════════════════════════════════════════════════════════════
def scrape_barttorvik_resumes():
    """Scrapes Barttorvik resume page for ELO, Q1/Q2 records, WAB, B Power."""
    print("\n[2/7] Scraping Barttorvik Resumes...")

    url = f"https://barttorvik.com/team-tables.php#"
    params = {"year": YEAR, "sort": "WAB", "top": 400, "mingames": 25}

    try:
        resp = get("https://barttorvik.com/team-tables.php", params=params)
        tables = pd.read_html(resp.text)
        if not tables:
            raise RuntimeError("No tables found")
        df_raw = tables[0]
    except Exception as e:
        print(f"  Resume scrape failed: {e}")
        print("  Trying resume JSON endpoint...")
        return scrape_barttorvik_resumes_json()

    df_raw.columns = [str(c).strip() for c in df_raw.columns]
    print(f"  Resume table columns: {list(df_raw.columns)}")

    col_map = {
        "team": "TEAM", "Team": "TEAM",
        "elo": "ELO", "ELO": "ELO",
        "q1 w": "Q1 W", "Q1 W": "Q1 W", "q1w": "Q1 W",
        "q2 w": "Q2 W", "Q2 W": "Q2 W", "q2w": "Q2 W",
        "wab": "WAB RANK", "WAB": "WAB RANK",
        "b power": "B POWER", "B Power": "B POWER", "bpower": "B POWER",
    }
    df_raw = df_raw.rename(columns={k: v for k, v in col_map.items() if k in df_raw.columns})
    df_raw["YEAR"] = YEAR
    df_raw = df_raw[df_raw["TEAM"].notna() & (df_raw["TEAM"].astype(str) != "Team")]

    # Ensure required columns exist
    for col in ["ELO", "Q1 W", "Q2 W", "WAB RANK", "B POWER"]:
        if col not in df_raw.columns:
            df_raw[col] = np.nan

    keep = ["YEAR", "TEAM", "ELO", "Q1 W", "Q2 W", "WAB RANK", "B POWER"]
    df_out = df_raw[[c for c in keep if c in df_raw.columns]].copy()

    existing = load_existing("Resumes.csv")
    return save_merged("Resumes.csv", existing, df_out)


def scrape_barttorvik_resumes_json():
    """JSON fallback for resumes."""
    url = "https://barttorvik.com/trank.php"
    resp = get(url, params={"year": YEAR, "json": 1})
    data = resp.json()
    rows = []
    for team in data:
        try:
            rows.append({
                "YEAR":    YEAR,
                "TEAM":    team.get("team", ""),
                "ELO":     float(team.get("elo", np.nan)),
                "Q1 W":    int(team.get("q1_w", 0)),
                "Q2 W":    int(team.get("q2_w", 0)),
                "WAB RANK":float(team.get("wab", np.nan)),
                "B POWER": float(team.get("bpower", np.nan)),
            })
        except Exception:
            continue
    df = pd.DataFrame(rows)
    existing = load_existing("Resumes.csv")
    return save_merged("Resumes.csv", existing, df)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. BARTTORVIK — Height/Experience → INT___KenPom___Height.csv
# ═══════════════════════════════════════════════════════════════════════════════
def scrape_barttorvik_height():
    """Scrapes Barttorvik for height and experience data."""
    print("\n[3/7] Scraping Barttorvik Height/Experience...")

    url = "https://barttorvik.com/trank.php"
    try:
        resp = get(url, params={"year": YEAR, "json": 1})
        data = resp.json()
    except Exception as e:
        print(f"  Height JSON failed: {e} — skipping")
        return pd.DataFrame()

    rows = []
    for team in data:
        try:
            rows.append({
                "Season":          YEAR,
                "TeamName":        team.get("team", ""),
                "AvgHeight":       float(team.get("hgt", np.nan)),
                "EffectiveHeight": float(team.get("eff_hgt", np.nan)),
                "Experience":      float(team.get("exp", np.nan)),
            })
        except Exception:
            continue

    df = pd.DataFrame(rows)
    print(f"  Fetched height data for {len(df)} teams")

    existing = load_existing("INT___KenPom___Height.csv")
    return save_merged("INT___KenPom___Height.csv", existing, df,
                       year_col="Season", team_col="TeamName")


# ═══════════════════════════════════════════════════════════════════════════════
# 4. BARTTORVIK — ATS Profiles → ats_team_profiles.csv
# ═══════════════════════════════════════════════════════════════════════════════
def scrape_barttorvik_ats():
    """Scrapes Barttorvik ATS page for team cover percentages."""
    print("\n[4/7] Scraping Barttorvik ATS profiles...")

    season_str = f"{YEAR-1}-{YEAR}"  # e.g. "2025-2026"

    url = "https://barttorvik.com/ats-ratings.php"
    try:
        resp = get(url, params={"year": YEAR})
        tables = pd.read_html(resp.text)
        if not tables:
            raise RuntimeError("No ATS tables found")
        df_raw = tables[0]
    except Exception as e:
        print(f"  ATS scrape failed: {e} — will use prior year data")
        return pd.DataFrame()

    df_raw.columns = [str(c).strip() for c in df_raw.columns]
    print(f"  ATS table columns: {list(df_raw.columns)}")

    col_map = {
        "team": "team", "Team": "team",
        "ats pct": "ats_pct", "ATS Pct": "ats_pct", "ATS%": "ats_pct",
        "dog cover": "dog_cover_pct", "Dog Cover": "dog_cover_pct",
        "fav cover": "fav_cover_pct", "Fav Cover": "fav_cover_pct",
    }
    df_raw = df_raw.rename(columns={k: v for k, v in col_map.items() if k in df_raw.columns})
    df_raw["season"] = season_str
    df_raw = df_raw[df_raw["team"].notna() & (df_raw["team"].astype(str) != "Team")]

    # Convert pct strings to floats if needed
    for col in ["ats_pct", "dog_cover_pct", "fav_cover_pct"]:
        if col in df_raw.columns:
            df_raw[col] = df_raw[col].astype(str).str.replace("%", "").str.strip()
            df_raw[col] = pd.to_numeric(df_raw[col], errors="coerce")
            # If values look like percentages (50-100) convert to decimal
            if df_raw[col].median() > 1:
                df_raw[col] = df_raw[col] / 100

    keep = ["season", "team", "ats_pct", "dog_cover_pct", "fav_cover_pct"]
    df_out = df_raw[[c for c in keep if c in df_raw.columns]].copy()

    existing = load_existing("ats_team_profiles.csv")
    # ATS uses season string not YEAR int — drop matching season rows
    if not existing.empty and "season" in existing.columns:
        existing = existing[existing["season"] != season_str]
    combined = pd.concat([existing, df_out], ignore_index=True)
    p = LOOKUP_DIR / "ats_team_profiles.csv"
    combined.to_csv(p, index=False)
    print(f"  ✅ Saved ats_team_profiles.csv ({len(df_out)} new rows, {len(combined)} total)")
    return combined


# ═══════════════════════════════════════════════════════════════════════════════
# 5. EVANMIYA — Ratings → EvanMiya.csv
# ═══════════════════════════════════════════════════════════════════════════════
def scrape_evanmiya():
    """Scrapes EvanMiya.com for relative rating, offensive and defensive rates."""
    print("\n[5/7] Scraping EvanMiya...")

    # EvanMiya exposes a public JSON endpoint
    url = "https://evanmiya.com/api/ratings"
    params = {"season": YEAR}

    try:
        resp = get(url, params=params)
        data = resp.json()

        rows = []
        for team in (data.get("teams") or data if isinstance(data, list) else []):
            try:
                rows.append({
                    "YEAR":            YEAR,
                    "TEAM":            team.get("team") or team.get("name", ""),
                    "RELATIVE RATING": float(team.get("relative_rating") or team.get("rating", np.nan)),
                    "O RATE":          float(team.get("o_rate") or team.get("offense", np.nan)),
                    "D RATE":          float(team.get("d_rate") or team.get("defense", np.nan)),
                    "RANK":            int(team.get("rank", 999)),
                })
            except Exception:
                continue

        if rows:
            df = pd.DataFrame(rows)
            print(f"  Fetched {len(df)} teams from EvanMiya JSON")
            existing = load_existing("EvanMiya.csv")
            return save_merged("EvanMiya.csv", existing, df)

    except Exception as e:
        print(f"  EvanMiya JSON failed: {e}")

    # Fallback: scrape HTML table
    print("  Trying EvanMiya HTML...")
    try:
        resp = get("https://evanmiya.com/", params={"season": YEAR})
        tables = pd.read_html(resp.text)
        if not tables:
            raise RuntimeError("No tables on EvanMiya page")

        df_raw = tables[0]
        df_raw.columns = [str(c).strip() for c in df_raw.columns]
        print(f"  EvanMiya HTML columns: {list(df_raw.columns)}")

        col_map = {
            "team": "TEAM", "Team": "TEAM",
            "rating": "RELATIVE RATING", "Rating": "RELATIVE RATING",
            "relative rating": "RELATIVE RATING",
            "o rate": "O RATE", "O Rate": "O RATE", "offense": "O RATE",
            "d rate": "D RATE", "D Rate": "D RATE", "defense": "D RATE",
        }
        df_raw = df_raw.rename(columns={k: v for k, v in col_map.items() if k in df_raw.columns})
        df_raw["YEAR"] = YEAR
        df_raw = df_raw[df_raw["TEAM"].notna() & (df_raw["TEAM"].astype(str) != "Team")]

        for col in ["RELATIVE RATING", "O RATE", "D RATE"]:
            if col not in df_raw.columns:
                df_raw[col] = np.nan

        existing = load_existing("EvanMiya.csv")
        return save_merged("EvanMiya.csv", existing, df_raw)

    except Exception as e:
        print(f"  EvanMiya HTML also failed: {e}")
        print("  ⚠️  EvanMiya skipped — model will use 0 for evan_rel/evan_o/evan_d")
        return pd.DataFrame()


# ═══════════════════════════════════════════════════════════════════════════════
# 6. SPORTS REFERENCE — AP Poll → AP_Poll_Data.csv
# ═══════════════════════════════════════════════════════════════════════════════
def scrape_ap_poll():
    """
    Scrapes Sports Reference for AP Poll weekly rankings.
    Model uses weeks 6 and 18+ so we fetch the full season.
    """
    print("\n[6/7] Scraping AP Poll from Sports Reference...")

    # Sports-reference NCAAB poll page
    season_yr = YEAR  # 2026 = 2025-26 season
    url = f"https://www.sports-reference.com/cbb/seasons/men/{season_yr}-polls.html"

    try:
        resp = get(url)
        tables = pd.read_html(resp.text, header=1)
        if not tables:
            raise RuntimeError("No poll tables found")
    except Exception as e:
        print(f"  AP Poll scrape failed: {e}")
        print("  ⚠️  AP Poll skipped — model will use default ap_rank=50")
        return pd.DataFrame()

    rows = []
    for week_num, table in enumerate(tables, start=1):
        table.columns = [str(c).strip() for c in table.columns]
        # Drop rows that are header repeats
        table = table[table.iloc[:, 0].astype(str).str.strip() != "School"]
        table = table[table.iloc[:, 0].notna()]

        for _, row in table.iterrows():
            try:
                team_col = [c for c in table.columns if "school" in c.lower() or "team" in c.lower()]
                rank_col = [c for c in table.columns if "rank" in c.lower() or "ap" in c.lower()]
                team = str(row[team_col[0]]).strip() if team_col else str(row.iloc[0]).strip()
                rank_val = row[rank_col[0]] if rank_col else row.iloc[1]
                rank = float(rank_val) if pd.notna(rank_val) and str(rank_val).strip() not in ("", "—", "NR") else np.nan
                rows.append({
                    "YEAR":    YEAR,
                    "WEEK":    week_num,
                    "TEAM":    team,
                    "AP RANK": rank,
                    "RANK?":   1 if pd.notna(rank) else 0,
                })
            except Exception:
                continue

    if not rows:
        print("  ⚠️  No AP Poll rows parsed")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = df[df["TEAM"].str.strip() != ""]
    print(f"  Fetched {len(df)} AP Poll entries across {df['WEEK'].nunique()} weeks")

    existing = load_existing("AP_Poll_Data.csv")
    return save_merged("AP_Poll_Data.csv", existing, df)


# ═══════════════════════════════════════════════════════════════════════════════
# 7. SPORTS REFERENCE — Coach lookup → REF___Current_NCAAM_Coaches.csv
# ═══════════════════════════════════════════════════════════════════════════════
def scrape_coaches():
    """
    Scrapes Sports Reference for current head coaches per team.
    Updates REF___Current_NCAAM_Coaches.csv with 2026 coach assignments.
    Coach tournament history (Coach_Results.csv) is historical and doesn't
    need annual updates — only the team→coach mapping does.
    """
    print("\n[7/7] Scraping current coaches from Sports Reference...")

    url = f"https://www.sports-reference.com/cbb/seasons/men/{YEAR}-coaches.html"

    try:
        resp = get(url)
        tables = pd.read_html(resp.text)
        if not tables:
            raise RuntimeError("No coach tables found")
        df_raw = tables[0]
    except Exception as e:
        print(f"  Coach scrape failed: {e}")
        print("  ⚠️  Coach data skipped — prior year mappings will be used")
        return pd.DataFrame()

    df_raw.columns = [str(c).strip() for c in df_raw.columns]
    print(f"  Coach table columns: {list(df_raw.columns)}")

    col_map = {
        "school": "Join Team", "School": "Join Team", "team": "Join Team", "Team": "Join Team",
        "coach": "Current Coach", "Coach": "Current Coach",
        "head coach": "Current Coach", "Head Coach": "Current Coach",
    }
    df_raw = df_raw.rename(columns={k: v for k, v in col_map.items() if k in df_raw.columns})

    if "Join Team" not in df_raw.columns or "Current Coach" not in df_raw.columns:
        # Try positional — usually school is col 0, coach is col 1 or 2
        cols = list(df_raw.columns)
        df_raw = df_raw.rename(columns={cols[0]: "Join Team", cols[1]: "Current Coach"})

    df_raw = df_raw[["Join Team", "Current Coach"]].dropna()
    df_raw = df_raw[df_raw["Join Team"].astype(str).str.strip() != "School"]
    df_raw["YEAR"] = YEAR

    print(f"  Fetched {len(df_raw)} coach assignments")

    # For this file we replace entirely rather than append — it's always current-year only
    p = LOOKUP_DIR / "REF___Current_NCAAM_Coaches.csv"
    df_raw.to_csv(p, index=False)
    print(f"  ✅ Saved REF___Current_NCAAM_Coaches.csv ({len(df_raw)} rows)")
    return df_raw


# ═══════════════════════════════════════════════════════════════════════════════
# Z RATING + HEAT CHECK — carry forward from most recent year
# ═══════════════════════════════════════════════════════════════════════════════
def carry_forward(fname, year_col="YEAR", team_col="TEAM"):
    """
    For files we can't scrape (Z_Rating, Heat_Check), find the most recent
    year's rows and duplicate them with YEAR=2026. This gives the model
    something to work with rather than zeros.
    """
    existing = load_existing(fname)
    if existing.empty:
        print(f"  ⚠️  {fname} not found — skipping carry-forward")
        return

    if year_col not in existing.columns:
        print(f"  ⚠️  {fname} has no {year_col} column — skipping")
        return

    # Already has 2026 rows?
    if str(YEAR) in existing[year_col].astype(str).values:
        print(f"  {fname} already has {YEAR} rows — skipping carry-forward")
        return

    latest_yr = existing[year_col].astype(str).replace("nan","").pipe(
        lambda s: s[s != ""]
    ).astype(float).max()

    prior_rows = existing[existing[year_col].astype(float) == latest_yr].copy()
    prior_rows[year_col] = YEAR

    combined = pd.concat([existing, prior_rows], ignore_index=True)
    p = LOOKUP_DIR / fname
    combined.to_csv(p, index=False)
    print(f"  ✅ {fname} — carried forward {len(prior_rows)} rows from {int(latest_yr)} → {YEAR}")


# ═══════════════════════════════════════════════════════════════════════════════
# KenPom_Preseason — derive from Barttorvik current ratings
# ═══════════════════════════════════════════════════════════════════════════════
def build_preseason_from_barttorvik():
    """
    KenPom_Preseason.csv needs PRESEASON KADJ EM, KADJ EM RANK CHANGE, KADJ EM CHANGE.
    Since we don't have KenPom access, we approximate:
      - PRESEASON KADJ EM  = prior year's KADJ EM (best available proxy)
      - KADJ EM CHANGE     = current KADJ EM - prior year KADJ EM
      - KADJ EM RANK CHANGE = rank change year over year
    """
    print("\n  Building KenPom_Preseason from Barttorvik...")

    kp = load_existing("KenPom_Barttorvik.csv")
    if kp.empty or "YEAR" not in kp.columns:
        print("  ⚠️  KenPom_Barttorvik.csv not ready — skipping preseason build")
        return

    kp["YEAR"] = kp["YEAR"].astype(int)
    curr = kp[kp["YEAR"] == YEAR].copy()
    prev = kp[kp["YEAR"] == YEAR - 1].copy()

    if curr.empty:
        print("  ⚠️  No 2026 Barttorvik data yet — skipping preseason build")
        return

    rows = []
    for _, row in curr.iterrows():
        team = row["TEAM"]
        curr_em = float(row.get("KADJ EM", np.nan))
        curr_rank = int(row.get("RANK", 999)) if pd.notna(row.get("RANK")) else 999

        # Find prior year
        prev_match = prev[prev["TEAM"] == team]
        if len(prev_match):
            prev_em   = float(prev_match.iloc[0].get("KADJ EM", curr_em))
            prev_rank = int(prev_match.iloc[0].get("RANK", curr_rank)) if pd.notna(prev_match.iloc[0].get("RANK")) else curr_rank
        else:
            prev_em   = curr_em
            prev_rank = curr_rank

        rows.append({
            "YEAR":               YEAR,
            "TEAM":               team,
            "PRESEASON KADJ EM":  prev_em,
            "KADJ EM CHANGE":     round(curr_em - prev_em, 2) if not (np.isnan(curr_em) or np.isnan(prev_em)) else 0.0,
            "KADJ EM RANK CHANGE":prev_rank - curr_rank,
        })

    df = pd.DataFrame(rows)
    existing = load_existing("KenPom_Preseason.csv")
    save_merged("KenPom_Preseason.csv", existing, df)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    start = datetime.now()
    print(f"\n{'='*60}")
    print(f"Sixth Sense Strategy — Lookup Scraper {YEAR}")
    print(f"Output: {LOOKUP_DIR}")
    print(f"{'='*60}")

    # ── Barttorvik (free, most important) ─────────────────────────────────────
    scrape_barttorvik_trank()
    scrape_barttorvik_resumes()
    scrape_barttorvik_height()
    scrape_barttorvik_ats()

    # ── EvanMiya ───────────────────────────────────────────────────────────────
    scrape_evanmiya()

    # ── Sports Reference ───────────────────────────────────────────────────────
    scrape_ap_poll()
    scrape_coaches()

    # ── Derived / carry-forward ────────────────────────────────────────────────
    build_preseason_from_barttorvik()
    carry_forward("Z_Rating_Teams.csv")
    carry_forward("Heat_Check_Tournament_Index.csv")
    carry_forward("Heat_Check_Ratings.csv")

    elapsed = (datetime.now() - start).seconds
    print(f"\n{'='*60}")
    print(f"✅ Done in {elapsed}s")
    print(f"\nNext steps:")
    print(f"  1. Review data/lookups/ — spot-check team names match your bracket")
    print(f"  2. git add data/lookups/ && git commit -m 'feat: 2026 lookup data'")
    print(f"  3. Run generate_picks.py once to verify predictions load correctly")
    print(f"{'='*60}\n")
