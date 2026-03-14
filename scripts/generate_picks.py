"""
generate_picks.py
-----------------
Runs the v7 GBM model on every matchup in the current bracket and writes
data/espn_scores.csv with all columns the dashboard expects.

Column contract (must match normalizeGame() in the dashboard):
  team1, seed1, team2, seed2, region, round, game_time, status, winner,
  score1, score2, ml_pick, ml_confidence, ml_odds, spread,
  ats_pick, ats_confidence, ml_result, ats_result, result

ml_odds  = American moneyline for the PICK TEAM (not necessarily the favorite).
           Derived from live_odds.csv h2h market.
spread   = consensus spread from the pick team's perspective
           (positive = underdog, negative = favorite).

Run order in GitHub Actions:
  1. fetch_data.py   → populates live_odds.csv + espn_raw.csv
  2. generate_picks.py → reads live_odds.csv + bracket.csv, writes espn_scores.csv

Fixes vs original:
  - round stored as string ('R64', 'R32', 'S16', 'E8', 'FF', 'CHAMP')
    to match dashboard filter logic
  - ml_result, ats_result, result columns computed from winner vs ml_pick
    so scorekeeping and record-tracking work on the dashboard
  - game_time passed through from bracket.csv to output
  - espn_raw.csv (not espn_scores.csv) used as the scores source, since
    fetch_data.py now writes raw ESPN data there
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
MODEL_PATH  = ROOT / "model" / "model_v7.pkl"
BRACKET_CSV = DATA_DIR / "bracket.csv"       # populated after Selection Sunday
ODDS_CSV    = DATA_DIR / "live_odds.csv"
RAW_CSV     = DATA_DIR / "espn_raw.csv"      # raw ESPN scores from fetch_data.py
OUT_CSV     = DATA_DIR / "espn_scores.csv"   # dashboard output — owned by this script

YEAR = 2026   # update each season

# ── Round label mapping ────────────────────────────────────────────────────────
# bracket.csv stores numeric round (64, 32, 16, 8, 4, 2)
# dashboard expects string keys ('R64', 'R32', 'S16', 'E8', 'FF', 'CHAMP')
ROUND_LABEL = {64: "R64", 32: "R32", 16: "S16", 8: "E8", 4: "FF", 2: "CHAMP"}
# Inverse map used when reading back existing espn_scores.csv
ROUND_NUM   = {"R64": 64, "R32": 32, "S16": 16, "E8": 8, "FF": 4, "CHAMP": 2}

# ── Load model ─────────────────────────────────────────────────────────────────
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)
print(f"Loaded model: {MODEL_PATH}")


# ── Team-name normalizer (must match train_v7.py) ──────────────────────────────
def norm(n):
    return str(n).lower().replace(" ","").replace(".","").replace("-","").replace("'","").replace("&","")[:10]

name_fixes = {
    "connecticut": "uconn", "albany": "ualbany",
    "collegeofch": "charlesto", "penn": "pennsylva",
}


# ── Feature lookups (same data sources as train_v7.py) ────────────────────────
# All CSVs live in the repo under data/lookups/ (committed once, not refreshed daily)
LOOKUP_DIR = ROOT / "data" / "lookups"

def _load(fname, low_memory=True):
    p = LOOKUP_DIR / fname
    if not p.exists():
        print(f"  WARNING: lookup file missing: {p}")
        return pd.DataFrame()
    return pd.read_csv(p, low_memory=low_memory)

kp      = _load("KenPom_Barttorvik.csv")
evan    = _load("EvanMiya.csv")
resume  = _load("Resumes.csv")
zrat    = _load("Z_Rating_Teams.csv")
hci     = _load("Heat_Check_Tournament_Index.csv")
hcr     = _load("Heat_Check_Ratings.csv")
kpp     = _load("KenPom_Preseason.csv")
ap      = _load("AP_Poll_Data.csv")
ht      = _load("INT___KenPom___Height.csv")
loc     = _load("Tournament_Locations.csv")
coach_s = _load("Coach_Results.csv")
coach_r = _load("REF___Current_NCAAM_Coaches.csv")
ats_p   = _load("ats_team_profiles.csv")
mega    = _load("DEV___March_Madness.csv", low_memory=False)


# ── Re-build lookups (copy of train_v7.py logic) ──────────────────────────────
tourney_mega = mega[mega["Post-Season Tournament"] == "March Madness"].copy() if not mega.empty else pd.DataFrame()

mega_cols = [
    "Pre-Tournament.AdjEM","Pre-Tournament.AdjOE","Pre-Tournament.AdjDE","Pre-Tournament.AdjTempo",
    "FG3Rate","OppFG3Rate","FTRate","TOPct","ORPct","EFG%","EFG%D",
    "Vulnerable Top 2 Seed?","Top 12 in AP Top 25 During Week 6?",
    "Active Coaching Length Index","CenterPts","PFPts","Conference",
    "Regular Season Conference Standing",
]
mega_lkp = {}
for _, row in tourney_mega.iterrows():
    try:
        yr = int(row["Season"]); tn = norm(str(row["Mapped ESPN Team Name"]))
        mega_lkp[(yr, tn)] = {c: row[c] for c in mega_cols if c in row.index}
    except: pass

conf_strength = {}
for yr in tourney_mega["Season"].unique() if not tourney_mega.empty else []:
    sub = tourney_mega[tourney_mega["Season"] == yr]
    if "Conference" in sub.columns and "Pre-Tournament.AdjEM" in sub.columns:
        for conf, val in sub.groupby("Conference")["Pre-Tournament.AdjEM"].mean().items():
            conf_strength[(int(yr), str(conf))] = float(val) if pd.notna(val) else np.nan

def fuzzy_mega(yr, team):
    tn = norm(team); fixed = name_fixes.get(tn[:10], name_fixes.get(tn[:9], tn))
    for n in [9, 7, 5]:
        for key, val in mega_lkp.items():
            if key[0] == yr and key[1][:n] == fixed[:n]: return val
    return {}

def build_lookup(df, year_col, team_col, value_cols):
    out = {}
    for _, row in df.iterrows():
        try: yr = int(row[year_col])
        except: continue
        out[(yr, norm(str(row[team_col])))] = {c: row.get(c, np.nan) for c in value_cols}
    return out

kp_lkp   = build_lookup(kp,  "YEAR","TEAM",["KADJ EM","KADJ O","KADJ D","BARTHAG","BADJ EM","TALENT","EXP","SEED"])
evan_lkp = build_lookup(evan,"YEAR","TEAM",["RELATIVE RATING","O RATE","D RATE"])
res_lkp  = build_lookup(resume,"YEAR","TEAM",["ELO","Q1 W","Q2 W","WAB RANK","B POWER"])
z_lkp    = build_lookup(zrat,"YEAR","TEAM",["Z RATING"])
hci_lkp  = build_lookup(hci, "YEAR","TEAM",["POWER","PATH","POOL VALUE"])
hcr_lkp  = build_lookup(hcr, "YEAR","TEAM",["EASY DRAW","TOUGH DRAW","DARK HORSE","UPSET ALERT","CINDERELLA"])
kpp_lkp  = build_lookup(kpp, "YEAR","TEAM",["PRESEASON KADJ EM","KADJ EM RANK CHANGE","KADJ EM CHANGE"])
ht_lkp   = build_lookup(ht,  "Season","TeamName",["AvgHeight","EffectiveHeight"])

ap_agg = ap[ap["WEEK"] >= 18].groupby(["YEAR","TEAM"]).agg(
    final_ap_rank=("AP RANK","last"),
    ap_times_ranked=("RANK?","sum"),
    ap_peak_rank=("AP RANK","min"),
).reset_index() if not ap.empty else pd.DataFrame(columns=["YEAR","TEAM","final_ap_rank","ap_times_ranked","ap_peak_rank"])

ap_lkp    = {(int(r["YEAR"]), norm(str(r["TEAM"]))): dict(r) for _, r in ap_agg.iterrows()}
ap_w6_lkp = {(int(r["YEAR"]), norm(str(r["TEAM"]))): float(r["AP RANK"]) if pd.notna(r["AP RANK"]) else 50
             for _, r in ap[ap["WEEK"] == 6].iterrows()} if not ap.empty else {}

coach_map = {}
for _, row in coach_r.iterrows():
    tn = norm(str(row["Join Team"])); cname = str(row["Current Coach"]).strip()
    match = coach_s[coach_s["COACH"].str.strip() == cname]
    if len(match):
        r = match.iloc[0]
        coach_map[tn[:7]] = {
            "pake": float(r["PAKE"]), "pase": float(r["PASE"]),
            "coach_f4_pct": float(str(r["F4%"]).replace("%","")) / 100,
        }

loc_dist = {}
for _, row in loc.iterrows():
    try:
        yr = int(row["YEAR"]); tn = norm(str(row["TEAM"])); rnd = row["CURRENT ROUND"]
        loc_dist[(yr, tn, rnd)] = {"distance_mi": float(row["DISTANCE (MI)"]), "tz_crossed": float(row["TIME ZONES CROSSED VALUE"])}
    except: pass

if not ats_p.empty:
    ats_p["team_norm"] = ats_p["team"].apply(norm)

def get_ats(team, yr):
    if ats_p.empty: return 0.5, 0.5
    sub = ats_p[ats_p["season"] == f"{yr-1}-{yr}"]; tn = norm(team)
    for n in [7, 5, 4]:
        m = sub[sub["team_norm"].str[:n] == tn[:n]]
        if len(m): return float(m.iloc[0]["ats_pct"]), float(m.iloc[0].get("dog_cover_pct", 0.5) or 0.5)
    return 0.5, 0.5

def fuzzy_lkp(lkp, yr, team):
    tn = norm(team)
    for n in [10, 7, 5, 4]:
        for key in lkp:
            if key[0] == yr and key[1][:n] == tn[:n]: return lkp[key]
    return {}

def sg(d, col, default=np.nan):
    try:
        v = d[col] if col in d else default
        if isinstance(v, pd.Series): v = v.iloc[0]
        return float(v) if pd.notna(v) else default
    except: return default

def bg(d, col):
    try:
        v = d[col] if col in d else False
        if isinstance(v, pd.Series): v = v.iloc[0]
        if isinstance(v, (bool, np.bool_)): return int(v)
        if isinstance(v, str): return int(v.lower() in ["true","yes"])
        return int(bool(v))
    except: return 0

def get_features(team, yr, round_num=64):
    tn = norm(team)
    k    = fuzzy_lkp(kp_lkp, yr, team);   e   = fuzzy_lkp(evan_lkp, yr, team)
    r    = fuzzy_lkp(res_lkp, yr, team);   z   = fuzzy_lkp(z_lkp, yr, team)
    h    = fuzzy_lkp(hci_lkp, yr, team);   hr  = fuzzy_lkp(hcr_lkp, yr, team)
    kpre = fuzzy_lkp(kpp_lkp, yr, team);   ht_ = fuzzy_lkp(ht_lkp, yr, team)
    ap_  = fuzzy_lkp(ap_lkp, yr, team);    c   = coach_map.get(tn[:7], {})
    ld   = loc_dist.get((yr, tn, round_num), {}); m = fuzzy_mega(yr, team)
    a_ats, a_dog = get_ats(team, yr)
    kadj_em = sg(k,"KADJ EM"); kadj_o = sg(k,"KADJ O"); kadj_d = sg(k,"KADJ D")
    em_change = sg(kpre,"KADJ EM CHANGE"); ap_final = sg(ap_,"final_ap_rank", 50)
    ap_w6_rank = ap_w6_lkp.get((yr, tn), 50)
    fg3_rate = sg(m,"FG3Rate"); opp_fg3_rate = sg(m,"OppFG3Rate")
    conf = str(m.get("Conference","")); conf_str = conf_strength.get((yr, conf), np.nan)
    return {
        "kadj_em":kadj_em,"barthag":sg(k,"BARTHAG"),"badj_em":sg(k,"BADJ EM"),
        "kadj_o":kadj_o,"kadj_d":kadj_d,"talent":sg(k,"TALENT"),"exp":sg(k,"EXP"),"seed":sg(k,"SEED"),
        "pt_adj_em":sg(m,"Pre-Tournament.AdjEM"),"pt_adj_oe":sg(m,"Pre-Tournament.AdjOE"),
        "pt_adj_de":sg(m,"Pre-Tournament.AdjDE"),"pt_tempo":sg(m,"Pre-Tournament.AdjTempo"),
        "fg3_rate":fg3_rate,"opp_fg3_rate":opp_fg3_rate,"ft_rate":sg(m,"FTRate"),
        "to_pct":sg(m,"TOPct"),"or_pct":sg(m,"ORPct"),"efg_pct":sg(m,"EFG%"),"efg_d":sg(m,"EFG%D"),
        "center_pts":sg(m,"CenterPts"),"pf_pts":sg(m,"PFPts"),
        "coach_tenure":sg(m,"Active Coaching Length Index"),
        "ap_week6":bg(m,"Top 12 in AP Top 25 During Week 6?"),"vuln_top2":bg(m,"Vulnerable Top 2 Seed?"),
        "evan_rel":sg(e,"RELATIVE RATING"),"evan_o":sg(e,"O RATE"),"evan_d":sg(e,"D RATE"),
        "elo":sg(r,"ELO"),"q1_wins":sg(r,"Q1 W"),"q2_wins":sg(r,"Q2 W"),
        "wab_rank":sg(r,"WAB RANK"),"b_power":sg(r,"B POWER"),
        "z_rating":sg(z,"Z RATING"),
        "hc_power":sg(h,"POWER"),"hc_path":sg(h,"PATH"),"hc_pool_val":sg(h,"POOL VALUE"),
        "easy_draw":bg(hr,"EASY DRAW"),"tough_draw":bg(hr,"TOUGH DRAW"),"dark_horse":bg(hr,"DARK HORSE"),
        "upset_alert":bg(hr,"UPSET ALERT"),"cinderella":bg(hr,"CINDERELLA"),
        "pre_kadj_em":sg(kpre,"PRESEASON KADJ EM"),"rank_change":sg(kpre,"KADJ EM RANK CHANGE"),
        "em_change":em_change,"avg_height":sg(ht_,"AvgHeight"),"eff_height":sg(ht_,"EffectiveHeight"),
        "ap_rank":ap_final,"ap_peak":sg(ap_,"ap_peak_rank",50),"ap_weeks":sg(ap_,"ap_times_ranked",0),
        "coach_pake":c.get("pake",0.0),"coach_pase":c.get("pase",0.0),"coach_f4pct":c.get("coach_f4_pct",0.0),
        "distance_mi":ld.get("distance_mi",0),"tz_crossed":ld.get("tz_crossed",0),
        "ats_pct":a_ats,"dog_cover":a_dog,
        "ap_momentum":ap_w6_rank - ap_final,
        "conf_strength":conf_str,
        "conf_standing":sg(m,"Regular Season Conference Standing"),
        "two_way":(kadj_o - kadj_d) if not (np.isnan(kadj_o) or np.isnan(kadj_d)) else np.nan,
        "three_pt_risk":(fg3_rate + opp_fg3_rate) if not (np.isnan(fg3_rate) or np.isnan(opp_fg3_rate)) else np.nan,
        "overperformed":em_change if not np.isnan(em_change) else 0.0,
    }

def build_feat_row(t1_feat, t2_feat, round_num):
    feat = {"round_num": round_num}
    for k in t1_feat:
        v1 = t1_feat.get(k, np.nan); v2 = t2_feat.get(k, np.nan)
        try: feat[f"{k}_diff"] = float(v1 or 0) - float(v2 or 0)
        except: feat[f"{k}_diff"] = 0.0
        feat[f"t1_{k}"] = v1; feat[f"t2_{k}"] = v2
    return feat

# Which columns the model was trained on (must match train_v7.py)
DIFF_COLS = [f"{k}_diff" for k in list(get_features("dummy", YEAR).keys())]
T1_ABS    = ["t1_seed","t1_ats_pct","t1_dog_cover","t1_upset_alert","t1_cinderella",
             "t1_easy_draw","t1_tough_draw","t1_coach_pake","t1_coach_pase",
             "t1_three_pt_risk","t1_vuln_top2","t1_dark_horse","t1_hc_path"]
FEAT_COLS = DIFF_COLS + T1_ABS + ["round_num"]


# ── Odds helpers ───────────────────────────────────────────────────────────────
def load_odds_lookup():
    """
    Returns two dicts keyed by norm(team_name):
      h2h[team]    = American ML odds for that team (most recent fetch, consensus)
      spread[team] = spread point value for that team (negative = favored)
    """
    if not ODDS_CSV.exists():
        print("  live_odds.csv not found — picks will have no odds")
        return {}, {}

    df = pd.read_csv(ODDS_CSV)

    # Keep only the most recent fetch date
    if "fetched_at" in df.columns:
        df = df[df["fetched_at"] == df["fetched_at"].max()]

    h2h_rows    = df[df["market"] == "h2h"]
    spread_rows = df[df["market"] == "spreads"]

    # Consensus ML per team: median across books
    h2h_lkp = {}
    for team, grp in h2h_rows.groupby("team"):
        prices = grp["price"].dropna().tolist()
        if prices:
            med = int(np.median(prices))
            # Format as American odds string
            h2h_lkp[norm(team)] = f"+{med}" if med > 0 else str(med)

    # Consensus spread per team: median across books
    spread_lkp = {}
    for team, grp in spread_rows.groupby("team"):
        pts = grp["point"].dropna().tolist()
        if pts:
            med = round(float(np.median(pts)), 1)
            spread_lkp[norm(team)] = med

    return h2h_lkp, spread_lkp


def get_pick_odds(pick_team, h2h_lkp):
    """Return American ML odds string for the pick team, e.g. '+350' or '-180'."""
    return h2h_lkp.get(norm(pick_team), "")


def get_spread_str(t1, t2, spread_lkp):
    """
    Return spread string from t1's perspective.
    e.g. if t1 is favored by 5.5 → '-5.5'; if underdog by 8 → '+8'
    """
    t1_pt = spread_lkp.get(norm(t1))
    t2_pt = spread_lkp.get(norm(t2))
    if t1_pt is not None:
        return f"{t1_pt:+.1f}".replace("+0.0","").replace("-0.0","")
    if t2_pt is not None:
        # flip sign
        return f"{-t2_pt:+.1f}".replace("+0.0","").replace("-0.0","")
    return ""


def get_ats_pick(pick_team, t1, t2, spread_lkp):
    """
    Returns e.g. 'Colorado St. +8.5' — the pick team and its spread.
    Spread is from the pick team's perspective.
    """
    pick_norm = norm(pick_team)
    t1_pt = spread_lkp.get(norm(t1))
    t2_pt = spread_lkp.get(norm(t2))

    if pick_norm == norm(t1) and t1_pt is not None:
        pt = t1_pt
    elif pick_norm == norm(t2) and t2_pt is not None:
        pt = t2_pt
    elif t1_pt is not None:
        pt = t1_pt if pick_norm == norm(t1) else -t1_pt
    else:
        return ""

    sign = "+" if pt > 0 else ""
    return f"{pick_team} {sign}{pt:.1f}"


# ── ATS confidence ─────────────────────────────────────────────────────────────
def ats_confidence(ml_conf, spread_lkp, pick_team, t1, t2):
    """
    Simple ATS confidence: start from ML confidence, discount when line is large
    (big favorites often don't cover), boost when line is tight (<3 pts).
    Returns percentage 0-100.
    """
    pick_norm = norm(pick_team)
    t1_pt = spread_lkp.get(norm(t1))
    t2_pt = spread_lkp.get(norm(t2))

    if pick_norm == norm(t1):
        pt = t1_pt if t1_pt is not None else None
    else:
        pt = t2_pt if t2_pt is not None else ((-t1_pt) if t1_pt is not None else None)

    if pt is None:
        return round(ml_conf * 0.95, 1)

    abs_pt = abs(pt)
    if abs_pt <= 3:
        adj = 1.02   # tight games → slight boost (more variance)
    elif abs_pt <= 7:
        adj = 0.98
    elif abs_pt <= 12:
        adj = 0.94
    else:
        adj = 0.88   # large favorites frequently don't cover

    return round(min(ml_conf * adj, 95.0), 1)


# ── Result reconciliation ──────────────────────────────────────────────────────
def compute_results(t1, t2, ml_pick, ats_pick, score1, score2, spread_str):
    """
    Given final scores and picks, return (result, ml_result, ats_result).

    result     = 'win'/'loss'/'pending'  — did ml_pick win the game?
    ml_result  = same as result (ML pick == game winner)
    ats_result = 'win'/'loss'/'push'/'pending' — did pick cover the spread?

    If scores are blank the game hasn't been played yet → all 'pending'.
    """
    # Scores must be present and non-empty to compute a result
    try:
        s1 = int(score1)
        s2 = int(score2)
        has_score = True
    except (ValueError, TypeError):
        has_score = False

    if not has_score:
        return "pending", "pending", "pending"

    # Who won the game?
    actual_winner = t1 if s1 > s2 else t2

    # ML result — did our pick match the actual winner?
    ml_result = "win" if norm(ml_pick) == norm(actual_winner) else "loss"
    result    = ml_result  # top-level result mirrors ML result

    # ATS result — did the pick team cover the spread?
    # spread_str is from t1's perspective (e.g. '-7.5' means t1 is -7.5)
    ats_result = "pending"
    try:
        if spread_str:
            t1_spread = float(spread_str)
            # Adjusted margin: t1 score + spread vs t2 score
            # t1 covers if (s1 + t1_spread) > s2
            # t2 covers if (s1 + t1_spread) < s2
            adjusted = s1 + t1_spread - s2

            pick_is_t1 = norm(ml_pick) == norm(t1)

            if abs(adjusted) < 0.5:
                ats_result = "push"
            else:
                t1_covered = adjusted > 0
                if pick_is_t1:
                    ats_result = "win" if t1_covered else "loss"
                else:
                    ats_result = "win" if not t1_covered else "loss"
    except (ValueError, TypeError):
        ats_result = "pending"

    return result, ml_result, ats_result


# ── Load ESPN raw scores for result lookup ─────────────────────────────────────
def load_raw_scores():
    """
    Load espn_raw.csv and build a lookup:
      (norm(winner), norm(loser)) → {winner_score, loser_score, margin}
    Used to back-fill scores and compute results for finished games.
    """
    if not RAW_CSV.exists():
        return {}
    df = pd.read_csv(RAW_CSV)
    lkp = {}
    for _, row in df.iterrows():
        w = norm(str(row.get("winner", "")))
        l = norm(str(row.get("loser",  "")))
        if w and l:
            lkp[(w, l)] = {
                "winner_score": int(row.get("winner_score", 0)),
                "loser_score":  int(row.get("loser_score",  0)),
            }
    return lkp


# ── Main prediction loop ───────────────────────────────────────────────────────
def generate_picks():
    if not BRACKET_CSV.exists():
        print(f"  bracket.csv not found at {BRACKET_CSV}")
        print("  Create data/bracket.csv after Selection Sunday with columns:")
        print("  team1,seed1,team2,seed2,region,round,game_time")
        return

    bracket = pd.read_csv(BRACKET_CSV)
    print(f"  Bracket: {len(bracket)} matchups")

    h2h_lkp, spread_lkp = load_odds_lookup()
    print(f"  Odds loaded: {len(h2h_lkp)} teams with ML odds, {len(spread_lkp)} with spreads")

    raw_scores = load_raw_scores()
    print(f"  Raw ESPN scores: {len(raw_scores)} completed games found")

    # Load existing espn_scores to preserve manually-set fields if needed
    existing = pd.DataFrame()
    if OUT_CSV.exists():
        existing = pd.read_csv(OUT_CSV)

    rows = []

    for _, g in bracket.iterrows():
        t1     = str(g["team1"]);   t2     = str(g["team2"])
        s1     = int(g["seed1"]);   s2     = int(g["seed2"])
        region = str(g.get("region", ""))
        rnd_num = int(g.get("round", 64))
        rnd_str = ROUND_LABEL.get(rnd_num, f"R{rnd_num}")  # 'R64', 'R32', etc.
        game_time = str(g.get("game_time", ""))

        # ── Look up existing row (keyed on string round label now) ─────────────
        ex_row = None
        if not existing.empty and "round" in existing.columns:
            mask = (
                (existing["team1"].astype(str).apply(norm) == norm(t1)) &
                (existing["team2"].astype(str).apply(norm) == norm(t2)) &
                (existing["round"].astype(str) == rnd_str)
            )
            if mask.any():
                ex_row = existing[mask].iloc[0]

        # ── Build features ─────────────────────────────────────────────────────
        try:
            f1 = get_features(t1, YEAR, rnd_num)
            f2 = get_features(t2, YEAR, rnd_num)
        except Exception as e:
            print(f"  Feature error {t1} vs {t2}: {e}")
            f1 = get_features("dummy", YEAR, rnd_num)
            f2 = get_features("dummy", YEAR, rnd_num)

        feat_row = build_feat_row(f1, f2, rnd_num)
        feat_df  = pd.DataFrame([feat_row])

        for col in FEAT_COLS:
            if col not in feat_df.columns:
                feat_df[col] = 0.0

        feat_df = feat_df[FEAT_COLS].fillna(0)

        # ── Model prediction ───────────────────────────────────────────────────
        try:
            prob = model.predict_proba(feat_df)[0]  # [p_loss, p_win] for t1
            p_t1 = float(prob[1])
        except Exception as e:
            print(f"  Predict error {t1} vs {t2}: {e}")
            p_t1 = 0.5

        pick    = t1 if p_t1 >= 0.5 else t2
        ml_conf = round(max(p_t1, 1 - p_t1) * 100, 1)

        # ── Odds ───────────────────────────────────────────────────────────────
        ml_odds  = get_pick_odds(pick, h2h_lkp)
        spread   = get_spread_str(t1, t2, spread_lkp)
        ats_pick = get_ats_pick(pick, t1, t2, spread_lkp)
        ats_conf = ats_confidence(ml_conf, spread_lkp, pick, t1, t2)

        # ── Scores and winner ──────────────────────────────────────────────────
        # Priority: existing espn_scores row → ESPN raw lookup → blank
        score1 = score2 = winner = ""
        status = "upcoming"

        if ex_row is not None:
            # Already have a processed row — preserve its score/winner/status
            score1  = ex_row.get("score1", "")
            score2  = ex_row.get("score2", "")
            winner  = str(ex_row.get("winner", ""))
            status  = str(ex_row.get("status", "upcoming"))
        else:
            # Try to match from raw ESPN scores
            t1n = norm(t1); t2n = norm(t2)
            raw = raw_scores.get((t1n, t2n)) or raw_scores.get((t2n, t1n))
            if raw:
                if raw_scores.get((t1n, t2n)):
                    score1 = raw["winner_score"]; score2 = raw["loser_score"]
                    winner = t1
                else:
                    score1 = raw["loser_score"];  score2 = raw["winner_score"]
                    winner = t2
                status = "final"

        # ── Compute result columns ─────────────────────────────────────────────
        # If we already have result columns stored (from a previous run), keep them.
        # Otherwise derive from scores + picks.
        if ex_row is not None and "ml_result" in ex_row and str(ex_row.get("ml_result","")) not in ("","nan","pending"):
            result     = str(ex_row.get("result",     "pending"))
            ml_result  = str(ex_row.get("ml_result",  "pending"))
            ats_result = str(ex_row.get("ats_result", "pending"))
        else:
            result, ml_result, ats_result = compute_results(
                t1, t2, pick, ats_pick, score1, score2, spread
            )

        rows.append({
            "team1":          t1,
            "seed1":          s1,
            "team2":          t2,
            "seed2":          s2,
            "region":         region,
            "round":          rnd_str,       # ← string label, e.g. 'R64'
            "game_time":      game_time,     # ← passed through from bracket.csv
            "status":         status,
            "winner":         winner,
            "score1":         score1,
            "score2":         score2,
            "ml_pick":        pick,
            "ml_confidence":  ml_conf,
            "ml_odds":        ml_odds,
            "spread":         spread,
            "ats_pick":       ats_pick,
            "ats_confidence": ats_conf,
            "result":         result,        # ← NEW: win/loss/pending
            "ml_result":      ml_result,     # ← NEW: win/loss/pending
            "ats_result":     ats_result,    # ← NEW: win/loss/push/pending
        })

        fav_label = "(fav)" if pick == (t1 if s1 < s2 else t2) else "(UPSET)"
        res_label = f"[{ml_result.upper()}]" if ml_result != "pending" else ""
        print(f"  #{s1} {t1:20} vs #{s2} {t2:20} → {pick} {ml_conf:.0f}% {ml_odds or '?':>6} {fav_label} {res_label}")

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUT_CSV, index=False)
    print(f"\n  ✅ Wrote {len(out_df)} picks → {OUT_CSV}")

    wins   = out_df[out_df["ml_result"] == "win"]
    losses = out_df[out_df["ml_result"] == "loss"]
    upsets = out_df[out_df.apply(
        lambda r: (int(r["seed1"]) > int(r["seed2"]) if norm(r["ml_pick"]) == norm(r["team1"])
                   else int(r["seed2"]) > int(r["seed1"])), axis=1)]
    print(f"  📊 Record: {len(wins)}-{len(losses)} ({len(out_df) - len(wins) - len(losses)} pending)")
    print(f"  🔥 {len(upsets)} upset picks")


if __name__ == "__main__":
    print(f"\n{'='*50}")
    print(f"Oracle v7 — Generate Picks ({YEAR})")
    print(f"{'='*50}\n")
    generate_picks()
