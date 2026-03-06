# scripts/21_bankroll_staking_and_governor_plan.py
"""
Step 21: Turn Step 20 adversarial MC into a deployable bankroll + staking + governor plan.

What this does
--------------
1) Loads Step 20 leaderboard + per-variant sims CSVs.
2) Picks a "calibration scenario" (default: month blocks, drag=0.005, badq=0.25, badx=3).
3) Converts unit-stake ruin probabilities into a recommended stake size for YOUR bankroll:
      If sims are run at 1.0 unit stake per bet,
      scaling stake by k scales the equity path by k.
      Ruin with real bankroll B under stake k  <=>  ruin with bankroll (B/k) under unit stake.

   So: pick the smallest "bankroll_required_unit" such that p_ruin <= target,
       then set k = B_real / bankroll_required_unit.

4) Suggests a simple "risk governor" using quantiles from the sims:
   - DD stop (pause / halve stake) based on pessimistic max_dd quantiles
   - 20D stop based on pessimistic worst_20d quantiles

Outputs
-------
- stake_plan_step21.csv (table of stake sizing across target ruin levels + scenarios)
- _SUMMARY_step21_bankroll_plan.json (machine-readable summary)
- Console printout with your recommended stake + governor thresholds

Assumptions
-----------
- Step 20 outputs exist:
    reports/bfsp/uk/greyhound/win/walkforward/step20_adversarial_mc/mc_step20_leaderboard.csv
    reports/bfsp/uk/greyhound/win/walkforward/step20_adversarial_mc/mc_sims_step20__*.csv
- The sims CSV includes columns: terminal_pnl, max_dd, worst_20d, ruin_200, ruin_300, ...
"""

from __future__ import annotations

from pathlib import Path
import json
import re
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# CONFIG (edit these)
# ---------------------------------------------------------------------
ROOT = Path("reports/bfsp/uk/greyhound/win/walkforward")
STEP20_DIR = ROOT / "step20_adversarial_mc"
LEADERBOARD = STEP20_DIR / "mc_step20_leaderboard.csv"

# Your real bankroll in "units" (if you treat 1 unit = £1, then this is £)
REAL_BANKROLL = 2000

# Target ruin probability for sizing (e.g., 0.01 for 1%, 0.05 for 5%)
TARGET_RUIN = 0.01

# Preferred calibration scenario (adversarial)
PREF_BLOCK_MODE = "month"     # "month" is more conservative than "quarter"
PREF_DRAG = 0.005             # per-bet drag (0.005 is a reasonable robustness penalty)

# Governor severity:
# - If drawdown exceeds DD_HARD_FRAC * bankroll, pause
# - If drawdown exceeds DD_SOFT_FRAC * bankroll, halve stake
DD_SOFT_FRAC = 0.50
DD_HARD_FRAC = 0.75

# Rolling-loss governor proxy from sims (uses worst_20d distribution):
# - If rolling 20D loss exceeds W20_SOFT_FRAC * bankroll, halve stake
# - If rolling 20D loss exceeds W20_HARD_FRAC * bankroll, pause
W20_SOFT_FRAC = 0.12
W20_HARD_FRAC = 0.18

# Quantiles used to set governor thresholds from sims (p05 is conservative)
Q_GOV = 0.05


# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------
def load_leaderboard() -> pd.DataFrame:
    if not LEADERBOARD.exists():
        raise FileNotFoundError(f"Missing Step 20 leaderboard: {LEADERBOARD}")
    df = pd.read_csv(LEADERBOARD)
    if "variant" not in df.columns and "variant" not in df.columns:
        # some versions call it "variant"
        pass
    # Normalize column name
    if "variant" not in df.columns and "variant" in df.columns:
        df = df.rename(columns={"variant": "variant"})
    return df


def find_variant_tag(df_lb: pd.DataFrame, drag: float, block_mode: str) -> str:
    """
    Find the Step 20 variant tag in the leaderboard that matches drag + block_mode.
    """
    # Example tag:
    #   drag=0.005|block=month|badq=0.25|badx=3
    # We match drag to 3dp to avoid floating quirks.
    drag_s = f"{drag:.3f}"
    mask = df_lb["variant"].astype(str).str.contains(fr"drag={drag_s}\|block={block_mode}", regex=True)
    sub = df_lb[mask].copy()
    if sub.empty:
        # fallback: looser match
        mask = df_lb["variant"].astype(str).str.contains(fr"drag={drag_s}", regex=True) & \
               df_lb["variant"].astype(str).str.contains(fr"block={block_mode}", regex=True)
        sub = df_lb[mask].copy()

    if sub.empty:
        raise ValueError(f"Could not find variant with drag={drag_s}, block={block_mode} in leaderboard.")

    # Pick best by robust_score if present, otherwise by terminal_p05
    if "robust_score" in sub.columns:
        sub = sub.sort_values("robust_score", ascending=False)
    elif "terminal_p05" in sub.columns:
        sub = sub.sort_values("terminal_p05", ascending=False)

    return str(sub.iloc[0]["variant"])


def safe_tag_to_filename_fragment(tag: str) -> str:
    # Must mirror Step 20 safe_tag logic:
    # safe_tag = tag.replace("=", "_").replace("|", "__").replace(":", "_")
    return tag.replace("=", "_").replace("|", "__").replace(":", "_")


def sims_path_for_tag(tag: str) -> Path:
    frag = safe_tag_to_filename_fragment(tag)
    # Step 20 saved: mc_sims_step20__{safe_tag}.csv
    p = STEP20_DIR / f"mc_sims_step20__{frag}.csv"
    if not p.exists():
        # Try to find it by glob in case formatting changed
        candidates = list(STEP20_DIR.glob("mc_sims_step20__*.csv"))
        if not candidates:
            raise FileNotFoundError(f"No sims CSVs found in {STEP20_DIR}")
        # pick the one that contains both drag and block fragments
        drag_m = re.search(r"drag=([0-9.]+)", tag)
        block_m = re.search(r"block=([a-z]+)", tag)
        drag_s = drag_m.group(1) if drag_m else ""
        block_s = block_m.group(1) if block_m else ""
        for c in candidates:
            name = c.name
            if drag_s.replace(".", "_") in name and f"block_{block_s}" in name:
                return c
        # fallback: first candidate
        return candidates[0]
    return p


def extract_ruin_columns(df: pd.DataFrame) -> list[int]:
    cols = [c for c in df.columns if c.startswith("ruin_")]
    bankrolls = []
    for c in cols:
        try:
            bankrolls.append(int(c.split("_", 1)[1]))
        except Exception:
            pass
    return sorted(set(bankrolls))


def ruin_probs_from_sims(df: pd.DataFrame) -> dict[int, float]:
    bankrolls = extract_ruin_columns(df)
    out = {}
    for b in bankrolls:
        col = f"ruin_{b}"
        out[b] = float(df[col].mean())
    return out


def bankroll_required_for_target(ruin_probs: dict[int, float], target: float) -> float:
    """
    Given p_ruin(bankroll) at discrete bankrolls, return the smallest bankroll
    that achieves p_ruin <= target using log-linear interpolation.

    If target is stricter than min achievable in table, we extrapolate.
    """
    bs = np.array(sorted(ruin_probs.keys()), dtype=float)
    ps = np.array([ruin_probs[int(b)] for b in bs], dtype=float)

    # Ensure monotonic-ish for interpolation (ruin should decrease with bankroll).
    # If noise violates monotonicity, we enforce a cumulative minimum from the right.
    ps_mono = np.minimum.accumulate(ps[::-1])[::-1]

    # If already below target at smallest bankroll
    if ps_mono[0] <= target:
        return float(bs[0])

    # If never below target, extrapolate beyond largest bankroll using last two points
    if ps_mono[-1] > target:
        # Use log(p) vs log(b) slope
        if len(bs) < 2:
            return float(bs[-1] * 2)
        b1, b2 = bs[-2], bs[-1]
        p1, p2 = ps_mono[-2], ps_mono[-1]
        # guard
        p1 = max(p1, 1e-9); p2 = max(p2, 1e-9); target_ = max(target, 1e-9)
        slope = (np.log(p2) - np.log(p1)) / (np.log(b2) - np.log(b1) + 1e-12)
        # log(target)=log(p2)+slope*(log(b)-log(b2))
        logb = np.log(b2) + (np.log(target_) - np.log(p2)) / (slope + 1e-12)
        return float(np.exp(logb))

    # Find first index where p <= target
    idx = int(np.where(ps_mono <= target)[0][0])
    if idx == 0:
        return float(bs[0])

    b_lo, b_hi = bs[idx - 1], bs[idx]
    p_lo, p_hi = ps_mono[idx - 1], ps_mono[idx]

    # Interpolate on log(p) vs log(b)
    p_lo = max(p_lo, 1e-9); p_hi = max(p_hi, 1e-9); target_ = max(target, 1e-9)

    x0, x1 = np.log(b_lo), np.log(b_hi)
    y0, y1 = np.log(p_lo), np.log(p_hi)
    # y = y0 + t*(y1-y0), solve for y=log(target)
    t = (np.log(target_) - y0) / (y1 - y0 + 1e-12)
    logb = x0 + t * (x1 - x0)
    return float(np.exp(logb))


def recommend_stake(real_bankroll: float, bankroll_required_unit: float) -> float:
    """
    stake_k such that ruin(real_bankroll, stake_k) ~= ruin(bankroll_required_unit, stake=1).
    So stake_k = real_bankroll / bankroll_required_unit.
    """
    if bankroll_required_unit <= 0:
        return 0.0
    return float(real_bankroll) / float(bankroll_required_unit)


def governor_thresholds_from_sims(df: pd.DataFrame, real_bankroll: float, stake_k: float) -> dict:
    """
    Convert unit-stake quantiles (max_dd, worst_20d) into real-money thresholds under stake_k.
    We also apply your fraction-based soft/hard gates as explicit caps.
    """
    maxdd_unit = df["max_dd"].to_numpy(dtype=float)
    w20_unit = df["worst_20d"].to_numpy(dtype=float)

    # These are negative numbers (losses). We convert to positive magnitudes.
    dd_mag_unit = -np.quantile(maxdd_unit, Q_GOV)   # p05 drawdown magnitude (conservative)
    w20_mag_unit = -np.quantile(w20_unit, Q_GOV)    # p05 worst-20D magnitude

    # Scale by stake k
    dd_mag_real = dd_mag_unit * stake_k
    w20_mag_real = w20_mag_unit * stake_k

    # Hard/soft caps based on fractions of bankroll
    dd_soft_cap = DD_SOFT_FRAC * real_bankroll
    dd_hard_cap = DD_HARD_FRAC * real_bankroll
    w20_soft_cap = W20_SOFT_FRAC * real_bankroll
    w20_hard_cap = W20_HARD_FRAC * real_bankroll

    return {
        "q_used": Q_GOV,
        "dd_p05_mag_unit": float(dd_mag_unit),
        "w20_p05_mag_unit": float(w20_mag_unit),
        "dd_p05_mag_real": float(dd_mag_real),
        "w20_p05_mag_real": float(w20_mag_real),
        "recommended_rules": {
            "soft_throttle": {
                "when_running_drawdown_exceeds": float(min(dd_mag_real, dd_soft_cap)),
                "or_when_rolling_20d_loss_exceeds": float(min(w20_mag_real, w20_soft_cap)),
                "action": "halve stake (k -> 0.5k) for next 20 betting days, then re-evaluate",
            },
            "hard_stop": {
                "when_running_drawdown_exceeds": float(min(dd_mag_real * 1.25, dd_hard_cap)),
                "or_when_rolling_20d_loss_exceeds": float(min(w20_mag_real * 1.25, w20_hard_cap)),
                "action": "pause betting for 20 betting days, then resume at half stake",
            },
        }
    }


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main():
    lb = load_leaderboard()

    # Pick preferred scenario tag
    pref_tag = find_variant_tag(lb, drag=PREF_DRAG, block_mode=PREF_BLOCK_MODE)
    sims_path = sims_path_for_tag(pref_tag)

    sims = pd.read_csv(sims_path)
    ruin_probs = ruin_probs_from_sims(sims)

    bankroll_req_unit = bankroll_required_for_target(ruin_probs, TARGET_RUIN)
    stake_k = recommend_stake(REAL_BANKROLL, bankroll_req_unit)

    gov = governor_thresholds_from_sims(sims, real_bankroll=REAL_BANKROLL, stake_k=stake_k)

    # Build a small table for multiple ruin targets (handy)
    target_grid = [0.10, 0.05, 0.02, 0.01, 0.005]
    rows = []
    for tgt in target_grid:
        br = bankroll_required_for_target(ruin_probs, tgt)
        k = recommend_stake(REAL_BANKROLL, br)
        rows.append({
            "variant": pref_tag,
            "target_ruin": tgt,
            "bankroll_required_unit_stake1": br,
            "recommended_stake_k": k,
            "recommended_unit_stake_for_£1_bankroll": k,  # interpret k as £ per 1-unit bet if 1 unit = £1
        })

    out_table = pd.DataFrame(rows)

    OUT_DIR = ROOT / "step21_bankroll_plan"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = OUT_DIR / "stake_plan_step21.csv"
    out_table.to_csv(csv_path, index=False)

    summary = {
        "real_bankroll": REAL_BANKROLL,
        "target_ruin": TARGET_RUIN,
        "picked_variant": pref_tag,
        "sims_csv": str(sims_path),
        "ruin_probs_unit_stake1": ruin_probs,
        "bankroll_required_unit_stake1": bankroll_req_unit,
        "recommended_stake_k": stake_k,
        "governor": gov,
        "stake_plan_csv": str(csv_path),
        "notes": {
            "interpretation": (
                "recommended_stake_k scales the unit-stake strategy. "
                "If you treated 1 unit = £1 previously, then stake_k ≈ £ per bet. "
                "If you bet in £0.10 increments, round stake_k accordingly."
            ),
            "why_month_blocks": "Month blocks preserve clustering and are more conservative than quarter blocks.",
            "drag": "Per-bet drag is applied daily as drag * n_bets to approximate commission/slippage/edge decay.",
        }
    }

    json_path = OUT_DIR / "_SUMMARY_step21_bankroll_plan.json"
    json_path.write_text(json.dumps(summary, indent=2))

    # Console printout (the bit you’ll actually use)
    print("\n" + "=" * 72)
    print("=== STEP 21: BANKROLL + STAKING + GOVERNOR PLAN ===")
    print(f"Real bankroll           : {REAL_BANKROLL}")
    print(f"Target ruin probability : {TARGET_RUIN:.4f}")
    print(f"Calibration scenario    : {pref_tag}")
    print(f"Sims used               : {sims_path}")
    print("-" * 72)
    print(f"Bankroll required (unit stake=1) to hit target ruin: {bankroll_req_unit:.2f}")
    print(f"Recommended stake scale k = REAL_BANKROLL / required: {stake_k:.4f}")
    print()
    print("Governor (suggested):")
    soft = gov["recommended_rules"]["soft_throttle"]
    hard = gov["recommended_rules"]["hard_stop"]
    print(f"  Soft throttle if running DD >= {soft['when_running_drawdown_exceeds']:.2f} "
          f"or 20D loss >= {soft['or_when_rolling_20d_loss_exceeds']:.2f}")
    print(f"    Action: {soft['action']}")
    print(f"  Hard stop if running DD >= {hard['when_running_drawdown_exceeds']:.2f} "
          f"or 20D loss >= {hard['or_when_rolling_20d_loss_exceeds']:.2f}")
    print(f"    Action: {hard['action']}")
    print("-" * 72)
    print(f"Saved: {csv_path}")
    print(f"Saved: {json_path}")
    print("=" * 72 + "\n")


if __name__ == "__main__":
    main()