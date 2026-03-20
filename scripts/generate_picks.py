"""
generate_picks.py
-----------------
Reads upcoming NCAA tournament games from tourney_schedule.csv,
runs the v10 GBM model on each one, and writes espn_scores.csv
for the dashboard.

Only predicts games that are NOT yet completed.
Completed games are passed through with their real scores and results.

Run order in GitHub Actions:
  1. fetch_data.py      → writes tourney_schedule.csv + live_odds.csv
  2. generate_picks.py  → reads tourney_schedule.csv, writes espn_scores.csv

Dashboard reads espn_scores.csv — column contract:
  round, region, team1, seed1, team2, seed2,
  ml_pick, ml_confidence, ml_odds, spread,
  ats_pick, ats_confidence,
  result, ml_result, ats_result,
  score1, score2, time
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent.parent
DATA_DIR    = ROOT / "data"
MODEL_PATH  = ROOT / "model" / "model_v10.pkl"
LOOKUP_DIR  = DATA_DIR / "lookups"
SCHEDULE    = DATA_DIR / "tourney_schedule.csv"   # written by fetch_data.py
ODDS_CSV    = DATA_DIR / "live_odds.csv"
OUT_CSV     = DATA_DIR / "espn_scores.csv"

YEAR = 2026


# ── Helpers ────────────────────────────────────────────────────────────────────
def norm(n):
    return str(n).lower().replace(" ","").replace(".","").replace("-","").replace("'","").replace("&","")[:10]

def safe_float(v):
    try:
        if isinstance(v, pd.Series): v = v.iloc[0]
        if isinstance(v, (bool, np.bool_)): return float(v)
        f = float(v)
        return f if not np.isnan(f) else np.nan
    except:
        return np.nan

def fb(a, b):
    """Return a if valid float, else b."""
    a_f = safe_float(a)
    return a_f if not np.isnan(a_f) else safe_float(b)


# ── Load lookup files ──────────────────────────────────────────────────────────
def _load(fname, low_memory=True):
    p = LOOKUP_DIR / fname
    if not p.exists():
        print(f"  WARNING: lookup missing: {p.name}")
        return pd.DataFrame()
    return pd.read_csv(p, low_memory=low_memory)

print("Loading lookup files...")
kp_raw     = _load("KenPom_Barttorvik.csv")
dev_raw    = _load("DEV___March_Madness.csv", low_memory=False)
evan_raw   = _load("EvanMiya.csv")
zrat_raw   = _load("Z_Rating_Teams.csv")
hci_raw    = _load("Heat_Check_Tournament_Index.csv")
hcr_raw    = _load("Heat_Check_Ratings.csv")
coach_res  = _load("Coach_Results.csv")
coaches_26 = _load("REF___Current_NCAAM_Coaches.csv")
reg_raw    = _load("MRegularSeasonDetailedResults.csv")

if not kp_raw.empty:
    kp_raw["YEAR"] = pd.to_numeric(kp_raw["YEAR"], errors="coerce").astype("Int64")


# ── Build lookups as plain dicts (avoids DataFrame fragmentation bug) ──────────
def build_lkp(df, year_col, name_col):
    if df.empty: return {}
    df = df.copy()
    df["_nk"] = df[name_col].apply(norm)
    df = df.drop_duplicates(subset=[year_col, "_nk"], keep="first")
    return {(int(r[year_col]), r["_nk"]): r.to_dict() for _, r in df.iterrows()}

kp_lkp   = build_lkp(kp_raw,  "YEAR",   "TEAM")
dev_lkp  = build_lkp(dev_raw, "Season", "Mapped ESPN Team Name") if not dev_raw.empty else {}
zrat_lkp = build_lkp(zrat_raw,"YEAR",   "TEAM")
hci_lkp  = build_lkp(hci_raw, "YEAR",   "TEAM")
hcr_lkp  = build_lkp(hcr_raw, "YEAR",   "TEAM")
evan_lkp = build_lkp(evan_raw,"YEAR",   "TEAM")

coach_perf = {}
if not coach_res.empty:
    coach_res["_nk"] = coach_res["COACH"].apply(norm)
    for _, r in coach_res.iterrows():
        coach_perf[r["_nk"]] = {"pake": safe_float(r.get("PAKE")), "pase": safe_float(r.get("PASE"))}

team_to_coach  = {}
team_to_tenure = {}
if not coaches_26.empty:
    for _, r in coaches_26.iterrows():
        nt = norm(str(r.get("Join Team", "") or r.get("Team", "")))
        nc = norm(str(r.get("Current Coach", "")))
        team_to_coach[nt] = nc
        try:
            since = int(str(r.get("Since", ""))[:4])
            team_to_tenure[nt] = YEAR - since
        except:
            pass


# ── 2026 regular season stats ──────────────────────────────────────────────────
reg_feats_26 = {}
if not reg_raw.empty:
    reg_2026 = reg_raw[reg_raw["Season"] == YEAR]
    for _, g in reg_2026.iterrows():
        for role, tid, pf, pa, fgm3, fga3, ftm, fta, stl, blk in [
            ("W", g["WTeamID"], g["WScore"], g["LScore"], g["WFGM3"], g["WFGA3"], g["WFTM"], g["WFTA"], g["WStl"], g["WBlk"]),
            ("L", g["LTeamID"], g["LScore"], g["WScore"], g["LFGM3"], g["LFGA3"], g["LFTM"], g["LFTA"], g["LStl"], g["LBlk"]),
        ]:
            k = int(tid)
            if k not in reg_feats_26:
                reg_feats_26[k] = {"w":0,"l":0,"pf":0,"pa":0,"fgm3":0,"fga3":0,"ftm":0,"fta":0,"stl":0,"blk":0}
            r = reg_feats_26[k]
            r["w"] += (role == "W"); r["l"] += (role == "L")
            r["pf"] += pf; r["pa"] += pa
            r["fgm3"] += fgm3; r["fga3"] += fga3
            r["ftm"] += ftm; r["fta"] += fta
            r["stl"] += stl; r["blk"] += blk

    # Convert to rate stats
    reg_stats_26 = {}
    for tid, r in reg_feats_26.items():
        g = max(r["w"] + r["l"], 1)
        fga3 = max(r["fga3"], 1); fta = max(r["fta"], 1)
        reg_stats_26[tid] = {
            "win_pct":  r["w"] / g,
            "net_ppg":  (r["pf"] - r["pa"]) / g,
            "ft_pct":   r["ftm"] / fta,
            "blk_pg":   r["blk"] / g,
            "stl_pg":   r["stl"] / g,
        }
else:
    reg_stats_26 = {}

# We need TeamID -> TeamName map for reg stats lookup
team_id_map = {}
teams_file = LOOKUP_DIR / "MTeams.csv"
if teams_file.exists():
    teams_df = pd.read_csv(teams_file)
    team_id_map = {norm(r["TeamName"]): int(r["TeamID"]) for _, r in teams_df.iterrows()}


# ── Massey 2026 ────────────────────────────────────────────────────────────────
massey_26 = {}
massey_file = LOOKUP_DIR / "MMasseyOrdinals_condensed.csv"
if massey_file.exists():
    mdf = pd.read_csv(massey_file)
    m26 = mdf[mdf["Season"] == YEAR] if "Season" in mdf.columns else pd.DataFrame()
    if not m26.empty:
        for _, r in m26.iterrows():
            massey_26[int(r["TeamID"])] = safe_float(r.get("massey_avg", np.nan))
        print(f"  Massey 2026: {len(massey_26)} teams")


# ── Row lookup helper ──────────────────────────────────────────────────────────
def find_row(lkp, season, name):
    n = norm(name)
    r = lkp.get((season, n))
    if r is None:
        for (yr, kn), v in lkp.items():
            if yr == season and kn[:6] == n[:6]:
                r = v; break
    return r


# ── Feature extraction ─────────────────────────────────────────────────────────
def get_features(team_name):
    """
    Build 43-element feature vector for a 2026 team.
    Matches exactly the feature order used to train v10.
    """
    n = norm(team_name)

    kp_row  = find_row(kp_lkp,   YEAR, team_name) or find_row(kp_lkp, YEAR-1, team_name)
    dev_row = find_row(dev_lkp,  YEAR, team_name)
    z_row   = find_row(zrat_lkp, YEAR, team_name)
    h_row   = find_row(hci_lkp,  YEAR, team_name)
    hr_row  = find_row(hcr_lkp,  YEAR, team_name)
    e_row   = find_row(evan_lkp, YEAR, team_name)

    coach_n = team_to_coach.get(n)
    if coach_n is None:
        for k in team_to_coach:
            if k[:6] == n[:6]: coach_n = team_to_coach[k]; break
    cp      = coach_perf.get(coach_n, {}) if coach_n else {}
    tenure  = team_to_tenure.get(n, np.nan)
    if tenure != tenure: tenure = np.nan  # nan check

    # RegSeason via team ID lookup
    tid = team_id_map.get(n)
    if tid is None:
        for k, v in team_id_map.items():
            if k[:6] == n[:6]: tid = v; break
    rs = reg_stats_26.get(tid, {}) if tid else {}

    # Massey
    ms_avg = massey_26.get(tid, np.nan) if tid else np.nan

    def kg(c): return safe_float(kp_row.get(c))  if kp_row  else np.nan
    def dg(c): return safe_float(dev_row.get(c)) if dev_row else np.nan
    def eg(c): return safe_float(e_row.get(c))   if e_row   else np.nan
    def zg(c): return safe_float(z_row.get(c))   if z_row   else np.nan
    def hg(c): return safe_float(h_row.get(c))   if h_row   else np.nan
    def hrg(c):return safe_float(hr_row.get(c))  if hr_row  else np.nan
    def rg(c): return rs.get(c, np.nan)

    kadj_em = fb(kg("KADJ EM"), dg("AdjEM"))
    kadj_o  = fb(kg("KADJ O"),  dg("AdjOE"))
    kadj_d  = fb(kg("KADJ D"),  dg("AdjDE"))
    exp     = fb(kg("EXP"),     dg("Experience"))

    pt_em = dg("Pre-Tournament.AdjEM"); pt_em = pt_em if not np.isnan(pt_em) else kadj_em
    pt_oe = dg("Pre-Tournament.AdjOE"); pt_oe = pt_oe if not np.isnan(pt_oe) else kadj_o
    pt_de = dg("Pre-Tournament.AdjDE"); pt_de = pt_de if not np.isnan(pt_de) else kadj_d
    overp = (kadj_em - pt_em) if not (np.isnan(kadj_em) or np.isnan(pt_em)) else 0.0
    two_way = (kadj_o - kadj_d) if not (np.isnan(kadj_o) or np.isnan(kadj_d)) else np.nan

    # 43 features — must match v10 training order exactly
    return [
        kadj_em, kadj_o, kadj_d,
        kg("BARTHAG"), kg("BADJ EM"), kg("TALENT"), exp,
        fb(kg("AVG HGT"), dg("AvgHeight")),
        fb(kg("EFG%"),    dg("eFGPct")),  kg("EFG%D"),
        fb(kg("3PT%"),    dg("FG3Pct")),  fb(kg("3PT%D"), dg("OppFG3Pct")),
        fb(kg("2PT%"),    dg("FG2Pct")),  kg("2PT%D"),
        fb(kg("TOV%"),    dg("TOPct")),   fb(kg("OREB%"), dg("ORPct")),
        rg("win_pct"),  rg("net_ppg"),   rg("ft_pct"),
        rg("blk_pg"),   rg("stl_pg"),
        kg("WAB"),
        np.nan,                          # conf_tw (not available at prediction time)
        np.nan,                          # last10 (not in KP lookup)
        pt_em, pt_oe, pt_de,
        eg("RELATIVE RATING"), eg("O RATE"), eg("D RATE"),
        zg("Z RATING"),
        hg("POWER"), hg("PATH"),
        hrg("EASY DRAW"), hrg("TOUGH DRAW"), hrg("DARK HORSE"), hrg("UPSET ALERT"),
        cp.get("pake", np.nan), cp.get("pase", np.nan),
        float(tenure) if tenure == tenure else np.nan,
        two_way, overp,
        float(ms_avg) if ms_avg == ms_avg else np.nan,
    ]

FEAT_KEYS = [
    "kadj_em","kadj_o","kadj_d","barthag","badj_em","talent","exp","avg_hgt",
    "efg_pct","efg_d","fg3_pct","opp_fg3","fg2_pct","opp_fg2","to_pct","or_pct",
    "win_pct","net_ppg","ft_pct","blk_pg","stl_pg","wab","conf_tw","last10",
    "pt_em","pt_oe","pt_de","evan_rel","evan_o","evan_d",
    "z_rating","hc_power","hc_path",
    "easy_draw","tough_draw","dark_horse","upset_alert",
    "coach_pake","coach_pase","coach_tenure","two_way","overperform","massey_avg",
]
DIFF_COLS     = [f"{k}_diff" for k in FEAT_KEYS]
ALL_FEAT_COLS = DIFF_COLS + ["seed_diff", "round_num", "exp_round_adj"]
ROUND_NUM_MAP = {"R64":64,"R32":32,"S16":16,"E8":8,"FF":4,"CHAMP":2}


# ── Odds lookup ────────────────────────────────────────────────────────────────
def build_odds_lkp():
    if not ODDS_CSV.exists(): return {}, {}
    odds = pd.read_csv(ODDS_CSV)
    h2h = odds[odds["market"] == "h2h"]
    spreads = odds[odds["market"] == "spreads"]

    # Most recent odds per team pair
    h2h_lkp = {}
    for _, r in h2h.sort_values("fetched_at").iterrows():
        key = norm(r["team"])
        h2h_lkp[key] = int(r["price"])

    spread_lkp = {}
    for _, r in spreads.sort_values("fetched_at").iterrows():
        key = norm(r["team"])
        spread_lkp[key] = r["point"]

    return h2h_lkp, spread_lkp


def get_ml_odds(pick, h2h_lkp):
    return h2h_lkp.get(norm(pick), "")


def get_spread(t1, t2, s1, s2, ml_conf, spread_lkp):
    """Get spread from odds if available, otherwise estimate from seed gap."""
    pick_spread = spread_lkp.get(norm(t1)) or spread_lkp.get(norm(t2))
    if pick_spread is not None:
        return f"{pick_spread:+.1f}"
    # Estimate from seed gap
    gap = abs(s1 - s2)
    est = -(gap * 2.0 + 1.5) if s1 < s2 else (gap * 2.0 + 1.5)
    return f"{est:+.1f}"


def compute_results(t1, t2, ml_pick, ats_pick, score1, score2, spread_str):
    """Determine win/loss/pending for each pick type."""
    try:
        sc1, sc2 = float(score1), float(score2)
    except:
        return "pending", "pending", "pending"

    actual_winner = t1 if sc1 > sc2 else t2

    # ML result
    ml_result = "win" if norm(ml_pick) == norm(actual_winner) else "loss"

    # Overall result = same as ML (did our pick win the game)
    result = ml_result

    # ATS result
    ats_result = "pending"
    try:
        spread_val = float(spread_str.replace("+",""))
        pick_team = ats_pick.split()[0] if ats_pick else ml_pick
        if norm(pick_team) == norm(t1):
            cover_margin = sc1 - sc2
            ats_result = "win" if cover_margin > -spread_val else ("push" if cover_margin == -spread_val else "loss")
        else:
            cover_margin = sc2 - sc1
            ats_result = "win" if cover_margin > spread_val else ("push" if cover_margin == spread_val else "loss")
    except:
        ats_result = "pending"

    return result, ml_result, ats_result


# ── Load model ────────────────────────────────────────────────────────────────
print(f"Loading model from {MODEL_PATH}...")
if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found at {MODEL_PATH}\n"
        f"Upload model_v10.pkl to model/ in the repo."
    )
with open(MODEL_PATH, "rb") as f:
    pkg = pickle.load(f)

model   = pkg["model"]
imputer = pkg["imputer"]
print(f"  Model loaded: {pkg.get('model_name','unknown')} | CV: {pkg.get('cv_score',0):.4f}")


# ── Main: predict upcoming games ───────────────────────────────────────────────
def generate_picks():
    # Load tournament schedule
    if not SCHEDULE.exists():
        print(f"  ERROR: {SCHEDULE} not found.")
        print("  Run fetch_data.py first to populate tourney_schedule.csv")
        return

    schedule = pd.read_csv(SCHEDULE)
    print(f"\n  Schedule: {len(schedule)} total games in tourney_schedule.csv")

    upcoming = schedule[~schedule["completed"].astype(bool)].copy()
    finished = schedule[schedule["completed"].astype(bool)].copy()
    print(f"  Upcoming/live: {len(upcoming)} | Finished: {len(finished)}")

    # Odds lookups
    h2h_lkp, spread_lkp = build_odds_lkp()

    output_rows = []

    # ── UPCOMING games: run model ──────────────────────────────────────────────
    no_data_count = 0
    for _, g in upcoming.iterrows():
        t1 = str(g.get("team1","")).strip()
        t2 = str(g.get("team2","")).strip()
        s1 = int(g.get("seed1") or 8)
        s2 = int(g.get("seed2") or 8)
        rnd_str  = str(g.get("round","R64"))
        region   = str(g.get("region",""))
        game_time= str(g.get("game_time",""))
        rnd_num  = ROUND_NUM_MAP.get(rnd_str, 64)

        if not t1 or not t2 or t1.upper() == "TBD" or t2.upper() == "TBD":
            continue

        # Get features
        f1 = get_features(t1)
        f2 = get_features(t2)

        # Build diff vector
        n_feats = len(FEAT_KEYS)
        diffs = [
            f1[i] - f2[i] if not (np.isnan(f1[i]) or np.isnan(f2[i])) else np.nan
            for i in range(n_feats)
        ]
        exp_d = diffs[6] if not np.isnan(diffs[6]) else 0.0
        row_vec = diffs + [
            float(s1 - s2),
            float(rnd_num),
            float(exp_d * (1.0 / max(rnd_num / 16, 1))),
        ]

        # Check data quality
        non_nan = sum(1 for x in row_vec if not np.isnan(x))
        if non_nan < 5:
            no_data_count += 1
            print(f"  WARNING: very few features for {t1} vs {t2} ({non_nan} non-NaN) — using seed fallback")
            # Seed-based fallback
            p1 = 0.5 + (s2 - s1) * 0.04
            p1 = max(0.51, min(0.97, p1))
            p2 = 1 - p1
        else:
            arr = np.array(row_vec, dtype=float).reshape(1, -1)
            prob = model.predict_proba(imputer.transform(arr))[0]
            p1, p2 = float(prob[1]), float(prob[0])

        pick    = t1 if p1 >= 0.5 else t2
        ml_conf = max(p1, p2) * 100
        pick_s  = s1 if pick == t1 else s2
        opp_s   = s2 if pick == t1 else s1

        # Odds + spread
        ml_odds  = get_ml_odds(pick, h2h_lkp)
        spread   = get_spread(t1, t2, s1, s2, ml_conf, spread_lkp)
        ats_pick = f"{pick} {spread}"
        ats_conf = min(ml_conf * 0.88 + np.random.uniform(-1, 1), 98)

        output_rows.append({
            "round":          rnd_str,
            "region":         region,
            "team1":          t1,
            "seed1":          s1,
            "team2":          t2,
            "seed2":          s2,
            "ml_pick":        pick,
            "ml_confidence":  round(ml_conf, 1),
            "ml_odds":        ml_odds,
            "ats_pick":       ats_pick,
            "ats_confidence": round(ats_conf, 1),
            "spread":         spread,
            "result":         "pending",
            "ml_result":      "pending",
            "ats_result":     "pending",
            "score1":         "",
            "score2":         "",
            "time":           game_time,
        })

    # ── FINISHED games: pass through with results ──────────────────────────────
    # Load existing espn_scores.csv to preserve prior picks/odds
    existing_picks = {}
    if OUT_CSV.exists():
        try:
            ex = pd.read_csv(OUT_CSV)
            for _, r in ex.iterrows():
                key = (norm(str(r.get("team1",""))), norm(str(r.get("team2",""))), str(r.get("round","")))
                existing_picks[key] = r.to_dict()
        except:
            pass

    for _, g in finished.iterrows():
        t1 = str(g.get("team1","")).strip()
        t2 = str(g.get("team2","")).strip()
        s1 = int(g.get("seed1") or 8)
        s2 = int(g.get("seed2") or 8)
        rnd_str  = str(g.get("round","R64"))
        region   = str(g.get("region",""))
        score1   = str(g.get("score1",""))
        score2   = str(g.get("score2",""))
        winner   = str(g.get("winner",""))
        game_time= str(g.get("game_time",""))

        if not t1 or not t2:
            continue

        # Try to recover pick from prior run
        key = (norm(t1), norm(t2), rnd_str)
        prior = existing_picks.get(key, {})

        ml_pick     = str(prior.get("ml_pick","")) or ""
        ml_conf     = float(prior.get("ml_confidence", 0) or 0)
        ml_odds     = str(prior.get("ml_odds",""))
        spread      = str(prior.get("spread",""))
        ats_pick    = str(prior.get("ats_pick",""))
        ats_conf    = float(prior.get("ats_confidence", 0) or 0)

        # If no prior pick exists, generate one now for recordkeeping
        if not ml_pick:
            f1 = get_features(t1)
            f2 = get_features(t2)
            rnd_num = ROUND_NUM_MAP.get(rnd_str, 64)
            n_feats = len(FEAT_KEYS)
            diffs = [
                f1[i] - f2[i] if not (np.isnan(f1[i]) or np.isnan(f2[i])) else np.nan
                for i in range(n_feats)
            ]
            exp_d = diffs[6] if not np.isnan(diffs[6]) else 0.0
            row_vec = diffs + [float(s1-s2), float(rnd_num), float(exp_d*(1.0/max(rnd_num/16,1)))]
            arr = np.array(row_vec, dtype=float).reshape(1,-1)
            prob = model.predict_proba(imputer.transform(arr))[0]
            p1 = float(prob[1])
            ml_pick = t1 if p1>=0.5 else t2
            ml_conf = max(p1, 1-p1)*100
            spread  = get_spread(t1, t2, s1, s2, ml_conf, spread_lkp)
            ats_pick= f"{ml_pick} {spread}"
            ats_conf= min(ml_conf*0.88, 98)
            ml_odds = ""

        # Compute results from actual scores
        result, ml_result, ats_result = compute_results(
            t1, t2, ml_pick, ats_pick, score1, score2, spread
        )

        output_rows.append({
            "round":          rnd_str,
            "region":         region,
            "team1":          t1,
            "seed1":          s1,
            "team2":          t2,
            "seed2":          s2,
            "ml_pick":        ml_pick,
            "ml_confidence":  round(ml_conf, 1),
            "ml_odds":        ml_odds,
            "ats_pick":       ats_pick,
            "ats_confidence": round(ats_conf, 1),
            "spread":         spread,
            "result":         result,
            "ml_result":      ml_result,
            "ats_result":     ats_result,
            "score1":         score1,
            "score2":         score2,
            "time":           game_time,
        })

    # ── Write output ───────────────────────────────────────────────────────────
    if not output_rows:
        print("  No games to output.")
        return

    ROUND_ORDER = {"R64":0,"R32":1,"S16":2,"E8":3,"FF":4,"CHAMP":5}
    out_df = pd.DataFrame(output_rows)
    out_df["_rord"] = out_df["round"].map(ROUND_ORDER).fillna(9)
    out_df = out_df.sort_values(["_rord","region","team1"]).drop(columns=["_rord"])
    out_df.to_csv(OUT_CSV, index=False)

    n_up   = len(out_df[out_df["result"] == "pending"])
    n_done = len(out_df[out_df["result"] != "pending"])
    n_wins = len(out_df[out_df["ml_result"] == "win"])
    n_loss = len(out_df[out_df["ml_result"] == "loss"])

    print(f"\n  ✅ espn_scores.csv written — {len(out_df)} games")
    print(f"     Upcoming: {n_up}  |  Finished: {n_done}  |  Record: {n_wins}-{n_loss}")
    if no_data_count:
        print(f"     WARNING: {no_data_count} games used seed fallback (missing KenPom data)")

    # Print summary
    print(f"\n  Upcoming picks:")
    upcoming_out = out_df[out_df["result"]=="pending"][["round","region","team1","seed1","team2","seed2","ml_pick","ml_confidence"]]
    if not upcoming_out.empty:
        print(upcoming_out.to_string(index=False))
    else:
        print("    (none — all games finished)")


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from datetime import datetime
    print(f"\n{'='*52}")
    print(f"Oracle Picks Generator — {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"{'='*52}\n")
    generate_picks()
