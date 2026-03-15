"""
scrape_lookups.py
-----------------
Scrapes all free data sources needed to update the model lookup CSVs for 2026.
Run this once after Selection Sunday (or as soon as possible before March 19).

Sources:
  - barttorvik.com  → KenPom_Barttorvik.csv, Resumes.csv,
                      INT___KenPom___Height.csv, ats_team_profiles.csv
  - evanmiya.com    → EvanMiya.csv
  - sports-reference.com → AP_Poll_Data.csv, REF___Current_NCAAM_Coaches.csv

All Barttorvik data fetched via JSON API endpoints — bypasses bot detection.

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

# Barttorvik JSON endpoints — these return raw JSON, no JS challenge
# trank.php?json=1 returns one dict per team with all ratings
BART_JSON_URL = "https://barttorvik.com/trank.php"
BART_ATS_URL  = "https://barttorvik.com/trank.php"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://barttorvik.com/",
}

SR_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "en-US,en;q=0.9",
}


def get_json(url, params=None, retries=3, delay=2.0):
    """Fetch JSON endpoint with retries."""
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, headers=HEADERS, timeout=20)
            resp.raise_for_status()
            time.sleep(delay)
            return resp.json()
        except Exception as e:
            print(f"    Attempt {attempt+1} failed: {e}")
            time.sleep(delay * (attempt + 1))
    raise RuntimeError(f"Failed to fetch {url} after {retries} attempts")


def get_html(url, params=None, retries=3, delay=3.0):
    """Fetch HTML page with retries (used for Sports Reference)."""
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, headers=SR_HEADERS, timeout=25)
            resp.raise_for_status()
            time.sleep(delay)
            return resp
        except Exception as e:
            print(f"    Attempt {attempt+1} failed: {e}")
            time.sleep(delay * (attempt + 1))
    raise RuntimeError(f"Failed to fetch {url} after {retries} attempts")


def load_existing(fname):
    p = LOOKUP_DIR / fname
    if p.exists():
        return pd.read_csv(p, low_memory=False)
    return pd.DataFrame()


def save_merged(fname, existing, new_df, year_col="YEAR", team_col="TEAM"):
    """Merge new 2026 rows into existing CSV, dropping any old 2026 rows first."""
    p = LOOKUP_DIR / fname
    if not existing.empty and year_col in existing.columns:
        existing = existing[existing[year_col].astype(str) != str(YEAR)]
    combined = pd.concat([existing, new_df], ignore_index=True)
    combined.to_csv(p, index=False)
    print(f"  ✅ Saved {fname} ({len(new_df)} new rows, {len(combined)} total)")
    return combined


# ═══════════════════════════════════════════════════════════════════════════════
# BARTTORVIK — single JSON fetch for ALL team data
# getadvstats.php returns one row per team with every stat column
# ═══════════════════════════════════════════════════════════════════════════════
def fetch_barttorvik_json():
    """
    Fetches the main Barttorvik JSON endpoint which contains all team stats
    in one call: efficiency, resume, height, ATS, tempo, etc.
    Returns raw list of dicts.
    """
    print("\n  Fetching Barttorvik JSON (all stats)...")
    params = {
        "year":     YEAR,
        "top":      400,
        "type":     "All",
        "mingames": 10,
    }
    try:
        data = get_json(BART_JSON_URL, params=params)
        # Response is either a list or {"data": [...]}
        if isinstance(data, dict):
            data = data.get("data", data.get("teams", []))
        print(f"  Fetched {len(data)} teams from Barttorvik JSON")
        return data
    except Exception as e:
        print(f"  ⚠️  Barttorvik JSON failed: {e}")
        return []


def parse_float(val, default=np.nan):
    try:
        return float(val) if val not in (None, "", "null") else default
    except Exception:
        return default


def parse_int(val, default=0):
    try:
        return int(float(val)) if val not in (None, "", "null") else default
    except Exception:
        return default


# ═══════════════════════════════════════════════════════════════════════════════
# 1. KenPom_Barttorvik.csv
# ═══════════════════════════════════════════════════════════════════════════════
def scrape_barttorvik_trank(data):
    print("\n[1/7] Building KenPom_Barttorvik.csv...")
    if not data:
        print("  ⚠️  No data — skipping")
        return pd.DataFrame()

    rows = []
    for t in data:
        try:
            # Barttorvik field names vary — try multiple known keys
            adjoe = parse_float(t.get("adjoe") or t.get("adj_oe") or t.get("AdjOE"))
            adjde = parse_float(t.get("adjde") or t.get("adj_de") or t.get("AdjDE"))
            adj_em = (adjoe - adjde) if not (np.isnan(adjoe) or np.isnan(adjde)) else np.nan
            rows.append({
                "YEAR":    YEAR,
                "TEAM":    t.get("team") or t.get("Team") or t.get("teamname", ""),
                "CONF":    t.get("conf") or t.get("Conf", ""),
                "SEED":    parse_float(t.get("seed") or t.get("Seed")),
                "KADJ EM": adj_em,
                "KADJ O":  adjoe,
                "KADJ D":  adjde,
                "BADJ EM": adj_em,
                "BARTHAG": parse_float(t.get("barthag") or t.get("Barthag")),
                "TALENT":  parse_float(t.get("talent") or t.get("Talent")),
                "EXP":     parse_float(t.get("exp") or t.get("experience")),
                "TEMPO":   parse_float(t.get("tempo") or t.get("adj_tempo")),
                "RANK":    parse_int(t.get("rk") or t.get("rank") or t.get("Rk"), 999),
            })
        except Exception as e:
            print(f"    Row error: {e}")

    df = pd.DataFrame(rows)
    df = df[df["TEAM"].astype(str).str.strip() != ""]
    print(f"  Parsed {len(df)} teams")
    existing = load_existing("KenPom_Barttorvik.csv")
    return save_merged("KenPom_Barttorvik.csv", existing, df)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Resumes.csv
# ═══════════════════════════════════════════════════════════════════════════════
def scrape_barttorvik_resumes(data):
    print("\n[2/7] Building Resumes.csv...")
    if not data:
        print("  ⚠️  No data — skipping")
        return pd.DataFrame()

    rows = []
    for t in data:
        try:
            rows.append({
                "YEAR":     YEAR,
                "TEAM":     t.get("team") or t.get("Team", ""),
                "ELO":      parse_float(t.get("elo") or t.get("ELO")),
                "Q1 W":     parse_int(t.get("q1w") or t.get("q1_w") or t.get("Q1W")),
                "Q2 W":     parse_int(t.get("q2w") or t.get("q2_w") or t.get("Q2W")),
                "WAB RANK": parse_float(t.get("wab") or t.get("WAB")),
                "B POWER":  parse_float(t.get("bpower") or t.get("b_power") or t.get("BPower")),
            })
        except Exception as e:
            print(f"    Row error: {e}")

    df = pd.DataFrame(rows)
    df = df[df["TEAM"].astype(str).str.strip() != ""]
    print(f"  Parsed {len(df)} teams")
    existing = load_existing("Resumes.csv")
    return save_merged("Resumes.csv", existing, df)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. INT___KenPom___Height.csv
# ═══════════════════════════════════════════════════════════════════════════════
def scrape_barttorvik_height(data):
    print("\n[3/7] Building INT___KenPom___Height.csv...")
    if not data:
        print("  ⚠️  No data — skipping")
        return pd.DataFrame()

    rows = []
    for t in data:
        try:
            rows.append({
                "Season":          YEAR,
                "TeamName":        t.get("team") or t.get("Team", ""),
                "AvgHeight":       parse_float(t.get("hgt") or t.get("avg_hgt") or t.get("AvgHeight")),
                "EffectiveHeight": parse_float(t.get("eff_hgt") or t.get("EffHgt") or t.get("EffectiveHeight")),
                "Experience":      parse_float(t.get("exp") or t.get("experience")),
            })
        except Exception as e:
            print(f"    Row error: {e}")

    df = pd.DataFrame(rows)
    df = df[df["TeamName"].astype(str).str.strip() != ""]
    print(f"  Parsed {len(df)} teams")
    existing = load_existing("INT___KenPom___Height.csv")
    return save_merged("INT___KenPom___Height.csv", existing, df,
                       year_col="Season", team_col="TeamName")


# ═══════════════════════════════════════════════════════════════════════════════
# 4. ats_team_profiles.csv
# ═══════════════════════════════════════════════════════════════════════════════
def scrape_barttorvik_ats(data):
    """
    ATS data from the same JSON payload — Barttorvik includes cover pct fields.
    Falls back to prior year if fields aren't present.
    """
    print("\n[4/7] Building ats_team_profiles.csv...")
    season_str = f"{YEAR-1}-{YEAR}"

    rows = []
    if data:
        for t in data:
            try:
                ats_pct = parse_float(
                    t.get("ats_pct") or t.get("atspct") or t.get("cover_pct")
                )
                dog_cover = parse_float(
                    t.get("dog_cover") or t.get("dog_cover_pct") or t.get("underdog_cover")
                )
                fav_cover = parse_float(
                    t.get("fav_cover") or t.get("fav_cover_pct") or t.get("favorite_cover")
                )
                # Convert from percentage to decimal if needed
                for val in [ats_pct, dog_cover, fav_cover]:
                    if not np.isnan(val) and val > 1:
                        val = val / 100

                rows.append({
                    "season":        season_str,
                    "team":          t.get("team") or t.get("Team", ""),
                    "ats_pct":       ats_pct / 100 if (not np.isnan(ats_pct) and ats_pct > 1) else ats_pct,
                    "dog_cover_pct": dog_cover / 100 if (not np.isnan(dog_cover) and dog_cover > 1) else dog_cover,
                    "fav_cover_pct": fav_cover / 100 if (not np.isnan(fav_cover) and fav_cover > 1) else fav_cover,
                })
            except Exception:
                continue

    # If we got meaningful ATS data, save it
    valid = [r for r in rows if not np.isnan(r.get("ats_pct", np.nan))]
    if valid:
        df = pd.DataFrame(rows)
        df = df[df["team"].astype(str).str.strip() != ""]
        print(f"  Parsed {len(df)} ATS profiles from JSON")
        existing = load_existing("ats_team_profiles.csv")
        if not existing.empty and "season" in existing.columns:
            existing = existing[existing["season"] != season_str]
        combined = pd.concat([existing, df], ignore_index=True)
        p = LOOKUP_DIR / "ats_team_profiles.csv"
        combined.to_csv(p, index=False)
        print(f"  ✅ Saved ats_team_profiles.csv ({len(df)} new rows, {len(combined)} total)")
        return combined
    else:
        print("  ⚠️  No ATS fields in JSON — carrying forward prior year ATS data")
        existing = load_existing("ats_team_profiles.csv")
        if existing.empty:
            print("  ⚠️  No prior ATS data found either — skipping")
            return pd.DataFrame()
        # Duplicate most recent season rows as current season
        latest = existing.sort_values("season").iloc[-1]["season"]
        prior = existing[existing["season"] == latest].copy()
        prior["season"] = season_str
        combined = pd.concat([existing, prior], ignore_index=True)
        p = LOOKUP_DIR / "ats_team_profiles.csv"
        combined.to_csv(p, index=False)
        print(f"  ✅ Carried forward {len(prior)} ATS rows from {latest} → {season_str}")
        return combined


# ═══════════════════════════════════════════════════════════════════════════════
# 5. EvanMiya.csv
# ═══════════════════════════════════════════════════════════════════════════════
def scrape_evanmiya():
    print("\n[5/7] Scraping EvanMiya...")

    # Try known EvanMiya JSON endpoints
    endpoints = [
        f"https://evanmiya.com/api/ratings?season={YEAR}",
        f"https://evanmiya.com/api/teams?season={YEAR}",
        f"https://evanmiya.com/?season={YEAR}",
    ]

    data = None
    for url in endpoints:
        try:
            resp = requests.get(url, headers=HEADERS, timeout=20)
            resp.raise_for_status()
            ct = resp.headers.get("content-type", "")
            if "json" in ct:
                data = resp.json()
                if isinstance(data, dict):
                    data = data.get("teams") or data.get("data") or []
                if data:
                    print(f"  Got {len(data)} teams from {url}")
                    break
            time.sleep(2)
        except Exception as e:
            print(f"  Endpoint {url} failed: {e}")
            continue

    if data:
        rows = []
        for t in data:
            try:
                rows.append({
                    "YEAR":            YEAR,
                    "TEAM":            t.get("team") or t.get("name") or t.get("Team", ""),
                    "RELATIVE RATING": parse_float(t.get("relative_rating") or t.get("rating") or t.get("rel_rating")),
                    "O RATE":          parse_float(t.get("o_rate") or t.get("offense") or t.get("ortg")),
                    "D RATE":          parse_float(t.get("d_rate") or t.get("defense") or t.get("drtg")),
                    "RANK":            parse_int(t.get("rank") or t.get("rk"), 999),
                })
            except Exception:
                continue

        if rows:
            df = pd.DataFrame(rows)
            df = df[df["TEAM"].astype(str).str.strip() != ""]
            existing = load_existing("EvanMiya.csv")
            return save_merged("EvanMiya.csv", existing, df)

    # HTML fallback
    print("  JSON failed — trying EvanMiya HTML table...")
    try:
        resp = requests.get("https://evanmiya.com/", headers=SR_HEADERS, timeout=20)
        tables = pd.read_html(resp.text)
        if tables:
            df_raw = tables[0]
            df_raw.columns = [str(c).strip() for c in df_raw.columns]
            col_map = {
                "team": "TEAM", "Team": "TEAM",
                "rating": "RELATIVE RATING", "Rating": "RELATIVE RATING",
                "relative rating": "RELATIVE RATING", "Relative Rating": "RELATIVE RATING",
                "o rate": "O RATE", "O Rate": "O RATE", "offense": "O RATE",
                "d rate": "D RATE", "D Rate": "D RATE", "defense": "D RATE",
            }
            df_raw = df_raw.rename(columns={k: v for k, v in col_map.items() if k in df_raw.columns})
            df_raw["YEAR"] = YEAR
            for col in ["RELATIVE RATING", "O RATE", "D RATE"]:
                if col not in df_raw.columns:
                    df_raw[col] = np.nan
            df_raw = df_raw[df_raw["TEAM"].notna() & (df_raw["TEAM"].astype(str) != "Team")]
            existing = load_existing("EvanMiya.csv")
            return save_merged("EvanMiya.csv", existing, df_raw)
    except Exception as e:
        print(f"  EvanMiya HTML also failed: {e}")

    print("  ⚠️  EvanMiya skipped — model will use 0 for evan_rel/evan_o/evan_d")
    return pd.DataFrame()


# ═══════════════════════════════════════════════════════════════════════════════
# 6. AP_Poll_Data.csv — Sports Reference
# ═══════════════════════════════════════════════════════════════════════════════
def scrape_ap_poll():
    print("\n[6/7] Scraping AP Poll from Sports Reference...")

    url = f"https://www.sports-reference.com/cbb/seasons/men/{YEAR}-polls.html"

    try:
        resp = get_html(url)
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
        table = table[table.iloc[:, 0].astype(str).str.strip() != "School"]
        table = table[table.iloc[:, 0].notna()]

        # Find team and rank columns flexibly
        team_col = next((c for c in table.columns if any(
            k in c.lower() for k in ["school","team","name"])), None)
        rank_col = next((c for c in table.columns if any(
            k in c.lower() for k in ["rank","ap","poll"])), None)

        if not team_col:
            team_col = table.columns[0]
        if not rank_col:
            rank_col = table.columns[1] if len(table.columns) > 1 else table.columns[0]

        for _, row in table.iterrows():
            try:
                team = str(row[team_col]).strip()
                if not team or team in ("nan", "School", "Team"):
                    continue
                rank_val = row[rank_col]
                rank = float(rank_val) if pd.notna(rank_val) and str(rank_val).strip() not in ("", "—", "NR", "nan") else np.nan
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
# 7. REF___Current_NCAAM_Coaches.csv — Sports Reference
# ═══════════════════════════════════════════════════════════════════════════════
def scrape_coaches():
    print("\n[7/7] Scraping current coaches from Sports Reference...")

    url = f"https://www.sports-reference.com/cbb/seasons/men/{YEAR}-coaches.html"

    try:
        resp = get_html(url)
        tables = pd.read_html(resp.text)
        if not tables:
            raise RuntimeError("No coach tables found")
        df_raw = tables[0]
    except Exception as e:
        print(f"  Coach scrape failed: {e}")
        print("  ⚠️  Coach data skipped — prior year mappings will be used")
        return pd.DataFrame()

    df_raw.columns = [str(c).strip() for c in df_raw.columns]

    col_map = {
        "school": "Join Team", "School": "Join Team",
        "team":   "Join Team", "Team":   "Join Team",
        "coach":       "Current Coach", "Coach":       "Current Coach",
        "head coach":  "Current Coach", "Head Coach":  "Current Coach",
    }
    df_raw = df_raw.rename(columns={k: v for k, v in col_map.items() if k in df_raw.columns})

    if "Join Team" not in df_raw.columns or "Current Coach" not in df_raw.columns:
        cols = list(df_raw.columns)
        df_raw = df_raw.rename(columns={cols[0]: "Join Team", cols[1]: "Current Coach"})

    df_raw = df_raw[["Join Team","Current Coach"]].dropna()
    df_raw = df_raw[df_raw["Join Team"].astype(str).str.strip() != "School"]
    df_raw["YEAR"] = YEAR

    p = LOOKUP_DIR / "REF___Current_NCAAM_Coaches.csv"
    df_raw.to_csv(p, index=False)
    print(f"  ✅ Saved REF___Current_NCAAM_Coaches.csv ({len(df_raw)} rows)")
    return df_raw


# ═══════════════════════════════════════════════════════════════════════════════
# Carry-forward for Z Rating + Heat Check (no public source)
# ═══════════════════════════════════════════════════════════════════════════════
def carry_forward(fname, year_col="YEAR"):
    existing = load_existing(fname)
    if existing.empty:
        print(f"  ⚠️  {fname} not found — skipping carry-forward")
        return
    if year_col not in existing.columns:
        print(f"  ⚠️  {fname} has no {year_col} column — skipping")
        return
    if str(YEAR) in existing[year_col].astype(str).values:
        print(f"  {fname} already has {YEAR} rows — skipping")
        return

    latest_yr = (
        existing[year_col].astype(str)
        .replace("nan", "")
        .pipe(lambda s: s[s != ""])
        .astype(float).max()
    )
    prior = existing[existing[year_col].astype(float) == latest_yr].copy()
    prior[year_col] = YEAR
    combined = pd.concat([existing, prior], ignore_index=True)
    (LOOKUP_DIR / fname).parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(LOOKUP_DIR / fname, index=False)
    print(f"  ✅ {fname} — carried {len(prior)} rows from {int(latest_yr)} → {YEAR}")


# ═══════════════════════════════════════════════════════════════════════════════
# KenPom_Preseason — derived from Barttorvik year-over-year delta
# ═══════════════════════════════════════════════════════════════════════════════
def build_preseason_from_barttorvik():
    print("\n  Building KenPom_Preseason.csv from Barttorvik delta...")

    kp = load_existing("KenPom_Barttorvik.csv")
    if kp.empty or "YEAR" not in kp.columns:
        print("  ⚠️  KenPom_Barttorvik.csv not ready — skipping")
        return

    kp["YEAR"] = kp["YEAR"].astype(int)
    curr = kp[kp["YEAR"] == YEAR].copy()
    prev = kp[kp["YEAR"] == YEAR - 1].copy()

    if curr.empty:
        print("  ⚠️  No 2026 Barttorvik rows — skipping")
        return

    rows = []
    for _, row in curr.iterrows():
        team     = row["TEAM"]
        curr_em  = parse_float(row.get("KADJ EM"))
        curr_rank = parse_int(row.get("RANK"), 999)

        prev_match = prev[prev["TEAM"] == team]
        if len(prev_match):
            prev_em   = parse_float(prev_match.iloc[0].get("KADJ EM", curr_em))
            prev_rank = parse_int(prev_match.iloc[0].get("RANK", curr_rank), 999)
        else:
            prev_em   = curr_em
            prev_rank = curr_rank

        rows.append({
            "YEAR":                YEAR,
            "TEAM":                team,
            "PRESEASON KADJ EM":   prev_em,
            "KADJ EM CHANGE":      round(curr_em - prev_em, 2) if not (np.isnan(curr_em) or np.isnan(prev_em)) else 0.0,
            "KADJ EM RANK CHANGE": prev_rank - curr_rank,
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

    # Single Barttorvik JSON fetch — reused for all 4 files
    bart_data = fetch_barttorvik_json()

    print("\n── Barttorvik ──────────────────────────────────────────")
    scrape_barttorvik_trank(bart_data)
    scrape_barttorvik_resumes(bart_data)
    scrape_barttorvik_height(bart_data)
    scrape_barttorvik_ats(bart_data)

    print("\n── EvanMiya ────────────────────────────────────────────")
    scrape_evanmiya()

    print("\n── Sports Reference ────────────────────────────────────")
    scrape_ap_poll()
    scrape_coaches()

    print("\n── Derived / carry-forward ─────────────────────────────")
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
    print(f"  3. Trigger manual workflow run to verify predictions are non-trivial")
    print(f"{'='*60}\n")
