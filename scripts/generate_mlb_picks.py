"""
generate_mlb_picks.py
=====================
Sixth Sense Strategy — MLB Daily Picks Generator
Loads trained GBM models, runs inference on today's raw_games.json,
writes final picks to data/mlb/picks.csv.

Run after fetch_mlb_data.py:
    python generate_mlb_picks.py

Output schema (data/mlb/picks.csv):
    game_id, game_date, game_time, status,
    team1 (home), team2 (away), seed1, seed2, score1, score2,
    pitcher1, pitcher2, region, round,
    ml_pick, ml_confidence, ml_odds, ml_result,
    spread, ats_pick, ats_confidence, ats_result,
    total_line, ou_pick, ou_confidence, ou_odds, ou_result,
    result, winner
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from datetime import date

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_DIR  = "models/mlb"
DATA_DIR   = "data/mlb"
INPUT_FILE = f"{DATA_DIR}/raw_games.json"
OUTPUT_CSV = f"{DATA_DIR}/picks.csv"

TODAY = date.today().strftime("%Y%m%d")

# Confidence threshold for publishing a pick (below this = "no pick" / low conviction)
MIN_CONF = 52

# ── Feature columns must match train_mlb_model.py ────────────────────────────
# Note: training used team-level pitching; inference uses SP stats where available,
# falling back to team stats. We bridge this by blending SP + bullpen.
FEATURE_COLS = [
    "h_woba", "h_wrc", "h_bb_pct", "h_k_pct",
    "a_woba", "a_wrc", "a_bb_pct", "a_k_pct",
    "h_era", "h_fip", "h_xfip", "h_whip", "h_k9", "h_bb9",
    "a_era", "a_fip", "a_xfip", "a_whip", "a_k9", "a_bb9",
    "woba_diff", "era_diff", "fip_diff",
    "park_factor", "is_home", "season",
]

# Default total lines by park factor tier (used if Odds API not yet run)
def default_total(park_factor: int) -> float:
    if park_factor >= 110: return 9.5
    if park_factor >= 105: return 9.0
    if park_factor >= 102: return 8.5
    if park_factor >= 98:  return 8.5
    return 8.0


# ── Load models ───────────────────────────────────────────────────────────────

def load_models():
    print("[1/3] Loading trained models...")
    try:
        ml_model = joblib.load(f"{MODEL_DIR}/ml_model.pkl")
        rl_model = joblib.load(f"{MODEL_DIR}/rl_model.pkl")
        ou_model = joblib.load(f"{MODEL_DIR}/ou_model.pkl")
        print("  ✓ ml_model, rl_model, ou_model loaded")
        return ml_model, rl_model, ou_model
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Models not found in {MODEL_DIR}/\n"
            "Run train_mlb_model.py first."
        )


# ── Build feature row for inference ──────────────────────────────────────────

def build_feature_row(g: dict) -> dict:
    """
    Map raw_games.json fields → FEATURE_COLS expected by the model.

    SP stats are blended with team bullpen stats:
    - If SP stats available: use weighted blend (SP 60% + team 40%)
    - This approximates full-game expected ERA accounting for bullpen
    """
    def blend(sp_val, team_val, sp_weight=0.60):
        try:
            return float(sp_val) * sp_weight + float(team_val) * (1 - sp_weight)
        except (TypeError, ValueError):
            return float(team_val)

    # Blended ERA/FIP for home pitcher (SP + team bullpen)
    h_era_blend  = blend(g.get("h_sp_era"),  g.get("h_pit_era",  4.2))
    h_fip_blend  = blend(g.get("h_sp_fip"),  g.get("h_pit_fip",  4.1))
    h_xfip_blend = blend(g.get("h_sp_xfip"), g.get("h_pit_xfip", 4.1))
    h_whip_blend = blend(g.get("h_sp_whip"), g.get("h_pit_whip", 1.3), 0.55)
    h_k9_blend   = blend(g.get("h_sp_k9"),   g.get("h_pit_k9",  8.7), 0.65)
    h_bb9_blend  = blend(g.get("h_sp_bb9"),  g.get("h_pit_bb9", 3.1), 0.65)

    a_era_blend  = blend(g.get("a_sp_era"),  g.get("a_pit_era",  4.2))
    a_fip_blend  = blend(g.get("a_sp_fip"),  g.get("a_pit_fip",  4.1))
    a_xfip_blend = blend(g.get("a_sp_xfip"), g.get("a_pit_xfip", 4.1))
    a_whip_blend = blend(g.get("a_sp_whip"), g.get("a_pit_whip", 1.3), 0.55)
    a_k9_blend   = blend(g.get("a_sp_k9"),   g.get("a_pit_k9",  8.7), 0.65)
    a_bb9_blend  = blend(g.get("a_sp_bb9"),  g.get("a_pit_bb9", 3.1), 0.65)

    return {
        "h_woba":    float(g.get("h_woba",    0.317)),
        "h_wrc":     float(g.get("h_wrc",     100)),
        "h_bb_pct":  float(g.get("h_bb_pct",  8.5)),
        "h_k_pct":   float(g.get("h_k_pct",   22.0)),

        "a_woba":    float(g.get("a_woba",    0.317)),
        "a_wrc":     float(g.get("a_wrc",     100)),
        "a_bb_pct":  float(g.get("a_bb_pct",  8.5)),
        "a_k_pct":   float(g.get("a_k_pct",   22.0)),

        "h_era":     h_era_blend,
        "h_fip":     h_fip_blend,
        "h_xfip":    h_xfip_blend,
        "h_whip":    h_whip_blend,
        "h_k9":      h_k9_blend,
        "h_bb9":     h_bb9_blend,

        "a_era":     a_era_blend,
        "a_fip":     a_fip_blend,
        "a_xfip":    a_xfip_blend,
        "a_whip":    a_whip_blend,
        "a_k9":      a_k9_blend,
        "a_bb9":     a_bb9_blend,

        "woba_diff": float(g.get("woba_diff",   0.0)),
        "era_diff":  a_era_blend - h_era_blend,    # positive = home SP advantage
        "fip_diff":  a_fip_blend - h_fip_blend,

        "park_factor": float(g.get("park_factor", 100)),
        "is_home":     1.0,
        "season":      float(g.get("season", 2025)),
    }


# ── Confidence → display string ───────────────────────────────────────────────

def conf_to_int(prob: float) -> int:
    """Convert raw model probability to 0-100 integer confidence."""
    return max(50, min(99, int(round(prob * 100))))


def implied_prob_from_odds(odds_str: str) -> float:
    """American odds string → implied probability (0-1)."""
    try:
        o = int(str(odds_str).replace("+", "").strip())
        if o > 0:
            return 100 / (o + 100)
        else:
            return abs(o) / (abs(o) + 100)
    except (ValueError, TypeError):
        return 0.5


# ── Generate picks ────────────────────────────────────────────────────────────

def generate_picks(games: list, ml_model, rl_model, ou_model) -> list[dict]:
    print(f"[2/3] Generating picks for {len(games)} games...")
    picks = []

    for g in games:
        h_name = g.get("home_name", g.get("home_abbr", "Home"))
        a_name = g.get("away_name", g.get("away_abbr", "Away"))
        h_abbr = g.get("home_abbr", "")
        a_abbr = g.get("away_abbr", "")

        # Build feature vector
        feat = build_feature_row(g)
        X = pd.DataFrame([feat])[FEATURE_COLS].fillna(0)

        # Predictions
        ml_home_prob  = float(ml_model.predict_proba(X)[0][1])
        rl_cover_prob = float(rl_model.predict_proba(X)[0][1])  # home -1.5 covers
        ou_over_prob  = float(ou_model.predict_proba(X)[0][1])  # goes over

        # ── ML pick ──────────────────────────────────────────────────────────
        if ml_home_prob >= 0.5:
            ml_pick = h_name
            ml_conf = conf_to_int(ml_home_prob)
        else:
            ml_pick = a_name
            ml_conf = conf_to_int(1 - ml_home_prob)

        # ── Run line pick (always -1.5 / +1.5) ───────────────────────────────
        spread = "-1.5"
        if rl_cover_prob >= 0.5:
            rl_pick = f"{h_name} {spread}"
            rl_conf = conf_to_int(rl_cover_prob)
        else:
            rl_pick = f"{a_name} +1.5"
            rl_conf = conf_to_int(1 - rl_cover_prob)

        # ── Over/Under pick ───────────────────────────────────────────────────
        total_line = default_total(int(g.get("park_factor", 100)))
        if ou_over_prob >= 0.5:
            ou_pick = f"Over {total_line}"
            ou_conf = conf_to_int(ou_over_prob)
        else:
            ou_pick = f"Under {total_line}"
            ou_conf = conf_to_int(1 - ou_over_prob)

        # ── Status and scores ─────────────────────────────────────────────────
        status = g.get("status", "Scheduled")
        home_score = g.get("home_score", "")
        away_score = g.get("away_score", "")
        winner     = g.get("winner", "")

        # Determine result (pending until game finishes)
        ml_result  = "pending"
        ats_result = "pending"
        ou_result  = "pending"
        result     = "pending"

        if winner:
            # ML result
            if winner == h_abbr:
                ml_result = "win" if ml_pick == h_name else "loss"
                result    = "win" if ml_pick == h_name else "loss"
            elif winner == a_abbr:
                ml_result = "win" if ml_pick == a_name else "loss"
                result    = "win" if ml_pick == a_name else "loss"

            # Run line result
            try:
                hs, as_ = int(home_score), int(away_score)
                diff = hs - as_
                home_covered = diff >= 2
                if rl_pick.startswith(h_name):
                    ats_result = "win" if home_covered else "loss"
                else:
                    ats_result = "win" if not home_covered else "loss"
            except (ValueError, TypeError):
                pass

            # O/U result
            try:
                total = int(home_score) + int(away_score)
                went_over = total > total_line
                if "Over" in ou_pick:
                    ou_result = "win" if went_over else "loss"
                else:
                    ou_result = "win" if not went_over else "loss"
            except (ValueError, TypeError):
                pass

        # Division/region label for display
        # Will be enriched by scrape_mlb_odds.py with actual division data
        region = f"{a_abbr} @ {h_abbr}"

        row = {
            "game_id":        g.get("game_id", ""),
            "game_date":      g.get("game_date", TODAY),
            "game_time":      g.get("game_time", ""),
            "status":         status,
            "round":          "MLB",
            "region":         region,

            # Dashboard fields (mirrors NCAAM schema)
            "team1":          h_name,
            "team2":          a_name,
            "seed1":          "",
            "seed2":          "",
            "score1":         home_score,
            "score2":         away_score,
            "winner":         winner,

            # Probable starters (displayed in dashboard)
            "pitcher1":       g.get("pitcher_home", ""),
            "pitcher2":       g.get("pitcher_away", ""),

            # ML pick
            "ml_pick":        ml_pick if ml_conf >= MIN_CONF else "",
            "ml_confidence":  ml_conf,
            "ml_odds":        "",         # filled by scrape_mlb_odds.py
            "ml_result":      ml_result,

            # Run line / ATS
            "spread":         spread,
            "ats_pick":       rl_pick if rl_conf >= MIN_CONF else "",
            "ats_confidence": rl_conf,
            "ats_result":     ats_result,

            # Over/Under
            "total_line":     total_line,
            "ou_pick":        ou_pick if ou_conf >= MIN_CONF else "",
            "ou_confidence":  ou_conf,
            "ou_odds":        "",         # filled by scrape_mlb_odds.py
            "ou_result":      ou_result,

            # Overall result (based on ML pick — used by dashboard header stats)
            "result":         result,

            # Raw probabilities for downstream debugging
            "_ml_home_prob":  round(ml_home_prob, 4),
            "_rl_cover_prob": round(rl_cover_prob, 4),
            "_ou_over_prob":  round(ou_over_prob, 4),
        }

        picks.append(row)

    return picks


# ── Write CSV ─────────────────────────────────────────────────────────────────

FINAL_COLS = [
    "game_id", "game_date", "game_time", "status", "round", "region",
    "team1", "team2", "seed1", "seed2", "score1", "score2", "winner",
    "pitcher1", "pitcher2",
    "ml_pick", "ml_confidence", "ml_odds", "ml_result",
    "spread", "ats_pick", "ats_confidence", "ats_result",
    "total_line", "ou_pick", "ou_confidence", "ou_odds", "ou_result",
    "result",
    # Keep raw probs for Odds API edge calc
    "_ml_home_prob", "_rl_cover_prob", "_ou_over_prob",
]


def write_csv(picks: list):
    print(f"[3/3] Writing {len(picks)} picks to {OUTPUT_CSV}...")
    df = pd.DataFrame(picks)

    # Ensure all expected columns exist
    for col in FINAL_COLS:
        if col not in df.columns:
            df[col] = ""

    df = df[FINAL_COLS]
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"  ✓ Saved to {OUTPUT_CSV}")


# ── Summary ───────────────────────────────────────────────────────────────────

def print_summary(picks: list):
    print("\n" + "=" * 55)
    print("TODAY'S MLB PICKS SUMMARY")
    print("=" * 55)
    for p in picks:
        ml_arrow = "→" if p["ml_confidence"] >= 65 else "·"
        print(
            f"  {ml_arrow} {p['team2']:20s} @ {p['team1']:20s}  "
            f"ML: {p['ml_pick'][:16]:16s} {p['ml_confidence']:3d}%  "
            f"RL: {str(p['ats_pick'])[:20]:20s} {p['ats_confidence']:3d}%  "
            f"O/U: {str(p['ou_pick'])[:14]:14s} {p['ou_confidence']:3d}%"
        )
    print("=" * 55)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print(f"SIXTH SENSE STRATEGY — MLB PICKS  [{TODAY}]")
    print("=" * 55)

    # Load raw games
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(
            f"{INPUT_FILE} not found. Run fetch_mlb_data.py first."
        )

    with open(INPUT_FILE) as f:
        games = json.load(f)

    if not games:
        print("No games today. Writing empty picks.csv.")
        pd.DataFrame(columns=FINAL_COLS).to_csv(OUTPUT_CSV, index=False)
        return

    # Load models
    ml_model, rl_model, ou_model = load_models()

    # Generate
    picks = generate_picks(games, ml_model, rl_model, ou_model)

    # Write
    write_csv(picks)
    print_summary(picks)

    print(f"\n✓ Done. Next: run scrape_mlb_odds.py to fill ml_odds + ou_odds fields.")


if __name__ == "__main__":
    main()
