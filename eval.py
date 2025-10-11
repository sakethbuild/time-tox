import os
import math
import hashlib
from collections import defaultdict
import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score

# ---------------- CONFIG ----------------
output_dir = "results"
os.makedirs(output_dir, exist_ok=True)

WINDOWS = ["screening_window", "1_month", "3_months", "6_months", "9_months", "12_months"]
CATEGORIES = ["core_treatment", "imaging_diagnostics", "labs", "clinic_visits"]

THRESH = {
    "totals_exact_match_min_pass": 0.95,
    "totals_within1_min_pass": 0.99,
    "totals_mae_max_pass": 0.1,
    "categories_within1_min_pass": 0.98,
    "categories_mae_max_pass": 0.2,
    "kappa_min_pass": 0.9,
    "kappa_min_warn": 0.8
}

# ---------------- Helpers ----------------
def safe_float(x):
    try:
        if pd.isna(x) or x == "":
            return 0.0
        if isinstance(x, str):
            x2 = x.replace(",", "").strip()
            return float(x2) if x2 != "" else 0.0
        return float(x)
    except:
        return 0.0

def mae(arr):
    arr = np.asarray(arr, dtype=float)
    return float(np.mean(np.abs(arr))) if arr.size>0 else float("nan")

def rmse(arr):
    arr = np.asarray(arr, dtype=float)
    return float(np.sqrt(np.mean(np.square(arr)))) if arr.size>0 else float("nan")

def pretty(v, digits=4):
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)

def load_and_prepare(path):
    df = pd.read_csv(path, dtype=str).fillna("")
    if "filename" not in df.columns or "intervention_type" not in df.columns:
        raise KeyError(f"CSV {path} missing required columns.")
    df["filename"] = df["filename"].astype(str).str.strip()
    df["intervention_type"] = df["intervention_type"].astype(str).str.strip()
    for w in WINDOWS:
        if w not in df.columns:
            df[w] = "0"
    for c in CATEGORIES:
        for w in WINDOWS:
            col = f"{c}_{w}"
            if col not in df.columns:
                df[col] = "0"
    numeric_cols = WINDOWS + [f"{c}_{w}" for c in CATEGORIES for w in WINDOWS]
    for col in numeric_cols:
        df[col] = df[col].apply(safe_float)
    return df

# ---------------- Load CSVs from runs folder ----------------
runs_dir = "runs"
path_run1 = os.path.join(runs_dir, "batch1.csv")
path_run2 = os.path.join(runs_dir, "batch2.csv")

# Verify files exist
if not os.path.exists(path_run1):
    raise FileNotFoundError(f"Could not find {path_run1}. Please ensure batch1.csv exists in the runs folder.")
if not os.path.exists(path_run2):
    raise FileNotFoundError(f"Could not find {path_run2}. Please ensure batch2.csv exists in the runs folder.")

print(f"Run1: {path_run1}, Run2: {path_run2}")

# ---------------- Load CSVs ----------------
df1 = load_and_prepare(path_run1)
df2 = load_and_prepare(path_run2)

# Create intervention identifiers (filename + intervention_type)
df1['intervention_id'] = df1['filename'] + '|' + df1['intervention_type']
df2['intervention_id'] = df2['filename'] + '|' + df2['intervention_type']

files1 = sorted(df1["filename"].unique().tolist())
files2 = sorted(df2["filename"].unique().tolist())

# Get unique interventions from both runs
interventions1 = set(df1['intervention_id'].unique())
interventions2 = set(df2['intervention_id'].unique())

intersection_interventions = interventions1 & interventions2
union_interventions = interventions1 | interventions2
only_in_run1 = interventions1 - interventions2
only_in_run2 = interventions2 - interventions1

intervention_coverage = len(intersection_interventions) / len(union_interventions) if len(union_interventions)>0 else 0.0

# Print diagnostic information
print(f"\n=== INTERVENTION MATCHING DIAGNOSTICS ===")
print(f"Total unique interventions in Run1: {len(interventions1)}")
print(f"Total unique interventions in Run2: {len(interventions2)}")
print(f"Overlapping interventions: {len(intersection_interventions)}")
print(f"Only in Run1: {len(only_in_run1)}")
print(f"Only in Run2: {len(only_in_run2)}")
print(f"Coverage: {intervention_coverage:.2%}\n")

if only_in_run1:
    print(f"Sample interventions only in Run1 (first 5):")
    for item in sorted(only_in_run1)[:5]:
        print(f"  - {item}")
    print()

if only_in_run2:
    print(f"Sample interventions only in Run2 (first 5):")
    for item in sorted(only_in_run2)[:5]:
        print(f"  - {item}")
    print()

sha_report = {}

# If no overlapping interventions, stop
if len(intersection_interventions) == 0:
    raise RuntimeError("No overlapping interventions between runs. Check filename and intervention_type values.")

# ---------------- Compare per-arm, per-cell ----------------
per_cell_rows = []
per_arm_rows = []
top_diffs = []
avg_abs_by_window = defaultdict(list)
avg_abs_by_cat_window = defaultdict(list)

# global accumulators for aggregated metrics
total_cells = 0
total_exact_matches = 0
total_within1 = 0
all_total_diffs = []
cat_cells = 0
cat_exact_matches = 0
cat_within1 = 0
all_cat_diffs = []
presence_flips = 0
reconciliation_failures = []  # list of tuples (filename, arm, run# , window, total, sum_cats)

# iterate each overlapping intervention
for intervention_id in sorted(intersection_interventions):
    # Parse intervention_id back to filename and intervention_type
    fname, intervention = intervention_id.split('|', 1)
    
    # select rows
    rows1 = df1[df1['intervention_id'] == intervention_id]
    rows2 = df2[df2['intervention_id'] == intervention_id]
    
    if len(rows1) == 0 or len(rows2) == 0:
        # shouldn't happen because intersection_interventions built from both, but guard
        print(f"Warning: Missing data for {intervention_id}")
        continue
    if len(rows1) > 1:
        print(f"Warning: multiple rows in run1 for {intervention_id} - using first row")
    if len(rows2) > 1:
        print(f"Warning: multiple rows in run2 for {intervention_id} - using first row")
    r1 = rows1.iloc[0]
    r2 = rows2.iloc[0]

    # Reconciliation for each run: sum(categories) == totals
    recon_fail_r1 = {}
    recon_fail_r2 = {}
    for w in WINDOWS:
        sum_r1 = sum(r1[f"{c}_{w}"] for c in CATEGORIES)
        tot_r1 = r1[w]
        if not math.isclose(sum_r1, tot_r1, rel_tol=0.0, abs_tol=1e-9):
            recon_fail_r1[w] = (tot_r1, sum_r1)
            reconciliation_failures.append((fname, intervention, 1, w, tot_r1, sum_r1))
        sum_r2 = sum(r2[f"{c}_{w}"] for c in CATEGORIES)
        tot_r2 = r2[w]
        if not math.isclose(sum_r2, tot_r2, rel_tol=0.0, abs_tol=1e-9):
            recon_fail_r2[w] = (tot_r2, sum_r2)
            reconciliation_failures.append((fname, intervention, 2, w, tot_r2, sum_r2))

    # Totals stability metrics
    total_diffs = []
    exact_count = 0
    within1_count = 0
    max_abs = -1
    max_abs_window = None
    for w in WINDOWS:
        v1 = r1[w]
        v2 = r2[w]
        d = v1 - v2
        total_diffs.append(d)
        all_total_diffs.append(d)
        avg_abs_by_window[w].append(abs(d))
        total_cells += 1
        if d == 0:
            exact_count += 1
            total_exact_matches += 1
        if abs(d) <= 1:
            within1_count += 1
            total_within1 += 1
        if abs(d) > max_abs:
            max_abs = abs(d)
            max_abs_window = w
        # presence flip
        p1 = 1 if v1 > 0 else 0
        p2 = 1 if v2 > 0 else 0
        if p1 != p2:
            presence_flips += 1
        per_cell_rows.append({
            "filename": fname, "intervention_type": intervention,
            "type": "TOTAL", "category": "TOTAL", "window": w,
            "run1": v1, "run2": v2,
            "diff": d, "abs_diff": abs(d),
            "exact_match": int(d==0), "within1": int(abs(d) <= 1)
        })
        top_diffs.append({"filename": fname, "intervention_type": intervention, "kind": "TOTAL", "category": "TOTAL", "window": w, "run1": v1, "run2": v2, "diff": d, "abs_diff": abs(d)})

    totals_exact_rate = exact_count / len(WINDOWS)
    totals_within1_rate = within1_count / len(WINDOWS)
    totals_mae = mae(total_diffs)
    totals_rmse = rmse(total_diffs)

    # Category stability metrics
    cat_diffs = []
    cat_exact_count = 0
    cat_within1_count = 0
    cat_shifts = 0
    cat_max_abs = -1
    cat_max_cell = None
    for c in CATEGORIES:
        for w in WINDOWS:
            col = f"{c}_{w}"
            v1 = r1[col]
            v2 = r2[col]
            d = v1 - v2
            cat_diffs.append(d)
            all_cat_diffs.append(d)
            avg_abs_by_cat_window[(c,w)].append(abs(d))
            cat_cells += 1
            if d == 0:
                cat_exact_count += 1
                cat_exact_matches += 1
            if abs(d) <= 1:
                cat_within1_count += 1
                cat_within1 += 1
            if abs(d) > cat_max_abs:
                cat_max_abs = abs(d)
                cat_max_cell = (c,w)
            # category shift: totals equal but distribution changed
            if math.isclose(r1[w], r2[w], rel_tol=0.0, abs_tol=1e-9) and not math.isclose(v1, v2, rel_tol=0.0, abs_tol=1e-9):
                cat_shifts += 1
            per_cell_rows.append({
                "filename": fname, "intervention_type": intervention,
                "type": "CATEGORY", "category": c, "window": w,
                "run1": v1, "run2": v2,
                "diff": d, "abs_diff": abs(d),
                "exact_match": int(d==0), "within1": int(abs(d) <= 1)
            })
            top_diffs.append({"filename": fname, "intervention_type": intervention, "kind": "CATEGORY", "category": c, "window": w, "run1": v1, "run2": v2, "diff": d, "abs_diff": abs(d)})

    categories_exact_rate = cat_exact_count / (len(CATEGORIES)*len(WINDOWS))
    categories_within1_rate = cat_within1_count / (len(CATEGORIES)*len(WINDOWS))
    categories_mae = mae(cat_diffs)
    categories_rmse = rmse(cat_diffs)

    # Compose per-arm summary and determine PASS/WARN/FAIL for this arm
    # Pass criteria (per spec)
    arm_pass = (
        (totals_exact_rate >= THRESH["totals_exact_match_min_pass"]) and
        (totals_within1_rate >= THRESH["totals_within1_min_pass"]) and
        (totals_mae <= THRESH["totals_mae_max_pass"]) and
        (categories_within1_rate >= THRESH["categories_within1_min_pass"]) and
        (categories_mae <= THRESH["categories_mae_max_pass"]) and
        # We'll check kappa per-arm below; here placeholder True and compute later
        True and
        (len(recon_fail_r1) == 0) and
        (len(recon_fail_r2) == 0) and
        (presence_flips == 0)  # presence flips global, but we include here conservatively; we'll better compute per-arm flips separately below
    )

    # Compute per-arm presence flips and per-arm kappa so the verdict is per-arm accurate
    presence_totals_r1 = [1 if r1[w] > 0 else 0 for w in WINDOWS]
    presence_totals_r2 = [1 if r2[w] > 0 else 0 for w in WINDOWS]
    # per-arm totals kappa
    try:
        kappa_totals_arm = cohen_kappa_score(presence_totals_r1, presence_totals_r2)
    except Exception:
        kappa_totals_arm = float("nan")
    # per-arm categories presence vectors
    presence_cat_r1 = []
    presence_cat_r2 = []
    per_arm_presence_flips = 0
    for c in CATEGORIES:
        for w in WINDOWS:
            a = 1 if r1[f"{c}_{w}"] > 0 else 0
            b = 1 if r2[f"{c}_{w}"] > 0 else 0
            presence_cat_r1.append(a)
            presence_cat_r2.append(b)
            if a != b:
                per_arm_presence_flips += 1
    try:
        kappa_cats_arm = cohen_kappa_score(presence_cat_r1, presence_cat_r2)
    except Exception:
        kappa_cats_arm = float("nan")

    # refine PASS/WARN/FAIL per arm using per-arm kappas and per-arm presence flips
    arm_status = "FAIL"
    # apply pass criteria
    if (
        (totals_exact_rate >= THRESH["totals_exact_match_min_pass"]) and
        (totals_within1_rate >= THRESH["totals_within1_min_pass"]) and
        (totals_mae <= THRESH["totals_mae_max_pass"]) and
        (categories_within1_rate >= THRESH["categories_within1_min_pass"]) and
        (categories_mae <= THRESH["categories_mae_max_pass"]) and
        (not per_arm_presence_flips) and
        (len(recon_fail_r1) == 0) and (len(recon_fail_r2) == 0) and
        (not math.isnan(kappa_totals_arm) and kappa_totals_arm >= THRESH["kappa_min_pass"])
    ):
        arm_status = "PASS"
    elif (
        (not per_arm_presence_flips) and
        (len(recon_fail_r1) == 0) and (len(recon_fail_r2) == 0) and
        (not math.isnan(kappa_totals_arm) and kappa_totals_arm >= THRESH["kappa_min_warn"])
    ):
        arm_status = "WARN"
    else:
        arm_status = "FAIL"


    per_arm_rows.append({
        "filename": fname,
        "intervention_type": intervention,
        # Totals summary
        "totals_exact_rate": totals_exact_rate,
        "totals_within1_rate": totals_within1_rate,
        "totals_mae": totals_mae,
        "totals_rmse": totals_rmse,
        "totals_max_abs": max_abs,
        "totals_max_abs_window": max_abs_window,
        # Categories summary
        "categories_exact_rate": categories_exact_rate,
        "categories_within1_rate": categories_within1_rate,
        "categories_mae": categories_mae,
        "categories_rmse": categories_rmse,
        "category_shifts": cat_shifts,
        # Presence/Reconciliation/Kappa
        "kappa_totals": float(kappa_totals_arm) if not math.isnan(kappa_totals_arm) else None,
        "kappa_categories": float(kappa_cats_arm) if not math.isnan(kappa_cats_arm) else None,
        "per_arm_presence_flips": per_arm_presence_flips,
        "recon_fail_run1": ";".join(f"{k}:{v[0]}!={v[1]}" for k,v in recon_fail_r1.items()) if recon_fail_r1 else "",
        "recon_fail_run2": ";".join(f"{k}:{v[0]}!={v[1]}" for k,v in recon_fail_r2.items()) if recon_fail_r2 else "",
        # verdict
        "status": arm_status
    })

# ---------------- Aggregate/global metrics ----------------
agg_totals_exact_rate = (total_exact_matches / total_cells) if total_cells>0 else float("nan")
agg_totals_within1_rate = (total_within1 / total_cells) if total_cells>0 else float("nan")
agg_totals_mae = mae(all_total_diffs)
agg_totals_rmse = rmse(all_total_diffs)

agg_cat_exact_rate = (cat_exact_matches / cat_cells) if cat_cells>0 else float("nan")
agg_cat_within1_rate = (cat_within1 / cat_cells) if cat_cells>0 else float("nan")
agg_cat_mae = mae(all_cat_diffs)
agg_cat_rmse = rmse(all_cat_diffs)

# --- New: Categorize interventions by total diffs ---
exact_match_interventions = []
within3_interventions = []
over3_interventions = []

for arm in per_arm_rows:
    # Sum of absolute diffs for totals across all windows
    total_abs_diffs = 0
    for w in WINDOWS:
        # Find corresponding per_cell row for this intervention, type TOTAL, window
        cell = next((row for row in per_cell_rows if row["filename"] == arm["filename"] and row["intervention_type"] == arm["intervention_type"] and row["type"] == "TOTAL" and row["window"] == w), None)
        if cell:
            total_abs_diffs += abs(cell["diff"])
    if total_abs_diffs == 0:
        exact_match_interventions.append(arm)
    elif total_abs_diffs <= 3:
        within3_interventions.append(arm)
    else:
        over3_interventions.append(arm)

# global kappas
# flatten presence vectors across all interventions/windows
presence_totals_y1 = []
presence_totals_y2 = []
presence_cat_y1 = []
presence_cat_y2 = []
for intervention_id in sorted(intersection_interventions):
    fname, intervention = intervention_id.split('|', 1)
    r1 = df1[df1['intervention_id'] == intervention_id].iloc[0]
    r2 = df2[df2['intervention_id'] == intervention_id].iloc[0]
    for w in WINDOWS:
        presence_totals_y1.append(1 if r1[w] > 0 else 0)
        presence_totals_y2.append(1 if r2[w] > 0 else 0)
    for c in CATEGORIES:
        for w in WINDOWS:
            presence_cat_y1.append(1 if r1[f"{c}_{w}"] > 0 else 0)
            presence_cat_y2.append(1 if r2[f"{c}_{w}"] > 0 else 0)

kappa_totals_global = cohen_kappa_score(presence_totals_y1, presence_totals_y2) if len(presence_totals_y1)>0 else float("nan")
kappa_categories_global = cohen_kappa_score(presence_cat_y1, presence_cat_y2) if len(presence_cat_y1)>0 else float("nan")

# Top diffs
top_diffs_sorted = sorted(top_diffs, key=lambda x: x["abs_diff"], reverse=True)
top_10 = top_diffs_sorted[:10]

# Tiny tables
avg_abs_by_window_out = {w: (np.mean(avg_abs_by_window[w]) if len(avg_abs_by_window[w])>0 else 0.0) for w in WINDOWS}
avg_abs_by_cat_window_out = {(c,w): (np.mean(avg_abs_by_cat_window[(c,w)]) if len(avg_abs_by_cat_window[(c,w)])>0 else 0.0) for c in CATEGORIES for w in WINDOWS}

# Overall PASS/WARN/FAIL for the whole comparison
overall_status = "FAIL"
if (
    (not reconciliation_failures) and
    (presence_flips == 0) and
    (agg_totals_exact_rate >= THRESH["totals_exact_match_min_pass"]) and
    (agg_totals_within1_rate >= THRESH["totals_within1_min_pass"]) and
    (agg_totals_mae <= THRESH["totals_mae_max_pass"]) and
    (agg_cat_within1_rate >= THRESH["categories_within1_min_pass"]) and
    (agg_cat_mae <= THRESH["categories_mae_max_pass"]) and
    (not math.isnan(kappa_totals_global) and kappa_totals_global >= THRESH["kappa_min_pass"])
):
    overall_status = "PASS"
elif (
    (not reconciliation_failures) and
    (presence_flips == 0) and
    (not math.isnan(kappa_totals_global) and kappa_totals_global >= THRESH["kappa_min_warn"])
):
    overall_status = "WARN"
else:
    overall_status = "FAIL"

# ---------------- Write outputs ----------------
per_cell_df = pd.DataFrame(per_cell_rows)
per_cell_df.to_csv(os.path.join(output_dir, "per_cell_diffs.csv"), index=False)

per_arm_df = pd.DataFrame(per_arm_rows)
per_arm_df.to_csv(os.path.join(output_dir, "per_arm_summary.csv"), index=False)

pd.DataFrame(top_diffs_sorted).to_csv(os.path.join(output_dir, "top_diffs.csv"), index=False)

by_window_df = pd.DataFrame([{"window": w, "avg_abs_delta": avg_abs_by_window_out[w]} for w in WINDOWS])
by_window_df.to_csv(os.path.join(output_dir, "avg_abs_by_window.csv"), index=False)

by_cat_window_df = pd.DataFrame([{"category": k[0], "window": k[1], "avg_abs_delta": v} for k,v in avg_abs_by_cat_window_out.items()])
by_cat_window_df.to_csv(os.path.join(output_dir, "avg_abs_by_category_window.csv"), index=False)

# human-readable summary
lines = []
lines.append("COMPARISON SUMMARY")
lines.append("==================")
lines.append(f"Run1: {path_run1}")
lines.append(f"Run2: {path_run2}")
lines.append("")
lines.append("INPUT & COVERAGE")
lines.append(f"Unique files Run1: {len(files1)}; Run2: {len(files2)}")
lines.append(f"Unique interventions Run1: {len(interventions1)}; Run2: {len(interventions2)}")
lines.append(f"Intervention coverage (intersection/union): {len(intersection_interventions)}/{len(union_interventions)} = {pretty(intervention_coverage)}")
lines.append(f"Only in Run1: {len(only_in_run1)}; Only in Run2: {len(only_in_run2)}")
lines.append("")
if sha_report:
    lines.append("SHA256 mismatches (heuristic):")
    for m in sha_report.get("mismatches", []):
        lines.append(str(m))
    lines.append("")
lines.append("TOTALS (aggregated)")
lines.append(f"Exact-match rate: {pretty(agg_totals_exact_rate)}")
lines.append(f"Within-1-day rate: {pretty(agg_totals_within1_rate)}")
lines.append(f"MAE: {pretty(agg_totals_mae)}; RMSE: {pretty(agg_totals_rmse)}")
lines.append("")
lines.append("CATEGORIES (aggregated)")
lines.append(f"Exact-match rate: {pretty(agg_cat_exact_rate)}")
lines.append(f"Within-1-day rate: {pretty(agg_cat_within1_rate)}")
lines.append(f"MAE: {pretty(agg_cat_mae)}; RMSE: {pretty(agg_cat_rmse)}")
lines.append("")
lines.append("PRESENCE/ABSENCE AGREEMENT")
lines.append(f"Cohen's kappa (totals): {pretty(kappa_totals_global)}")
lines.append(f"Cohen's kappa (categories): {pretty(kappa_categories_global)}")
lines.append(f"Presence flips (0<->>0 across all cells): {presence_flips}")
lines.append("")
lines.append("RECONCILIATION")
lines.append(f"Reconciliation failures (sum(categories) != total): {len(reconciliation_failures)}")
for rf in reconciliation_failures[:50]:
    lines.append(str(rf))
lines.append("")
lines.append("TOP 10 ABSOLUTE DIFFS")
for t in top_10:
    lines.append(f"{t['filename']} | {t['intervention_type']} | {t['kind']} | {t['category']} | {t['window']}: {t['run1']} -> {t['run2']} | Δ={t['diff']}")
lines.append("")
lines.append("OVERALL PASS/WARN/FAIL")
lines.append(f"Overall status: {overall_status}")
lines.append("PASS criteria (aggregated): totals exact>=95%, within1>=99%, MAE<=0.1; categories within1>=98%, MAE<=0.2; kappa>=0.9; no presence flips; reconciliation OK.")
lines.append("WARN criteria: kappa>=0.8 and no presence flips and no reconciliation failures.")
lines.append("")
lines.append("Files written:")
lines.append(" - " + os.path.join(output_dir, "per_cell_diffs.csv"))
lines.append(" - " + os.path.join(output_dir, "per_arm_summary.csv"))
lines.append(" - " + os.path.join(output_dir, "top_diffs.csv"))
lines.append(" - " + os.path.join(output_dir, "avg_abs_by_window.csv"))
lines.append(" - " + os.path.join(output_dir, "avg_abs_by_category_window.csv"))
lines.append(" - " + os.path.join(output_dir, "summary.txt"))

with open(os.path.join(output_dir, "summary.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

# Print a short console summary
print("\n".join(lines[:20]))
print("\nFull outputs written to:", os.path.abspath(output_dir))