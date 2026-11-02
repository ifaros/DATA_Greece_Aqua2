"""
Synthesize trajectories for untested (in-between) targets by pairing and
blending real trajectories from the two nearest tested neighbors along one
axis (x, y, or theta -- whichever one changes between the neighbors), then
scoring the synthetic trajectories with the same success/failure check used
for real runs.

Example: you have 10 real runs for (3,0,0) and 10 for (3,3,0). This script
will pair them up trial-by-trial (real run #1 of (3,0,0) with real run #1
of (3,3,0), #2 with #2, ...), and for each pair build a synthetic run for
targets in between -- e.g. (3,0.75,0), (3,1.5,0), (3,2.25,0) at 25/50/75%
of the way -- by blending the paired trajectories sample-by-sample. At the
midpoint (t=0.5) this blend is literally the average of the pair; off-center
points use the corresponding weighted linear blend.

If the two neighbors have different trial counts (e.g. 10 vs 6), only
min(n_A, n_B) pairs are used (matched in original trial order: #1 with #1,
#2 with #2, ...), so every intermediate target gets an equal number of
synthetic trajectories.

Because real trajectories differ in length/duration, each pair is first
resampled onto a common time grid before blending (holding each
trajectory's last known pose past its own end -- i.e. assuming the vehicle
held station once its run finished).

Usage:
    python3 make_results_xy.py <root_dir> <output.xlsx>

Example (reproduces the dataset used in the paper):
    python3 make_results_xy.py path/to/Greece data/results_xy.xlsx
"""

import sys
import re
from pathlib import Path
from itertools import groupby

import numpy as np
import openpyxl
from openpyxl.styles import Font, PatternFill

# ---------------- SETTINGS (edit if needed) ----------------

POS_TOL = 0.3            # meters
HEADING_TOL_DEG = 20.0   # degrees (+/-)
TIME_TOL_S = 60.0        # seconds, elapsed-time cutoff (matches real controller)
SAMPLE_RATE_HZ = 10.0    # used only if timestamps.npy is missing/corrupted

# Fractions of the way from neighbor A to neighbor B to synthesize.
# 0.5 = midpoint. Edit this list freely, e.g. [0.5] for midpoint only.
# Total synthetic targets produced = (number of axis pairs) x (len of this list).
# With 17 axis pairs: 4 fractions -> 68 synthetic targets, 5 fractions -> 85.
INTERP_FRACTIONS = [0.25, 0.5, 0.75]

# Which axis/axes to interpolate along. Currently y-axis AND x-axis:
#   y-axis: for every (x, theta) combo, interpolate (x,3,theta)<->(x,0,theta)
#           and (x,-3,theta)<->(x,0,theta), for x in {3,6,9}, theta in {0,+45,-45}.
#   x-axis: for every (y, theta) combo, interpolate (3,y,theta)<->(6,y,theta)
#           and (6,y,theta)<->(9,y,theta), for y in {-3,0,3}, theta in {0,+45,-45}.
# Add "theta" here too if you also want interpolation across heading.
# Pairs are only created where BOTH real neighbor targets actually exist in
# your data -- missing combinations are skipped automatically.
AXES_TO_INTERPOLATE = ["y", "x"]

X_FILE = "traj_x.npy"
Y_FILE = "traj_y.npy"
HEAD_FILE = "head_meas.npy"
TIME_FILE = "timestamps.npy"   # optional

FOLDER_RE = re.compile(r'^test_(?P<x>\d)(?P<yneg>n?)(?P<y>\d)(?P<theta>n?pi4)?_')

# -------------------------------------------------------------


def wrap_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def parse_folder_name(name):
    m = FOLDER_RE.match(name)
    if not m:
        return None
    x = float(m.group("x"))
    y = float(m.group("y"))
    if m.group("yneg") == "n":
        y = -y
    theta_str = m.group("theta")
    if theta_str is None:
        theta_deg = 0.0
    elif theta_str == "pi4":
        theta_deg = 45.0
    elif theta_str == "npi4":
        theta_deg = -45.0
    else:
        theta_deg = 0.0
    return x, y, theta_deg


def safe_load(path):
    return np.load(path, allow_pickle=False)


class TrajectoryError(Exception):
    pass


def load_trajectory(folder: Path):
    """Returns (elapsed, x, y, theta) or raises TrajectoryError with a reason."""
    x_path, y_path, h_path, t_path = (folder / X_FILE, folder / Y_FILE,
                                       folder / HEAD_FILE, folder / TIME_FILE)
    for p in (x_path, y_path, h_path):
        if not p.exists():
            raise TrajectoryError(f"MISSING FILE: {p.name}")
    try:
        x, y, theta = safe_load(x_path), safe_load(y_path), safe_load(h_path)
    except Exception as e:
        raise TrajectoryError(f"CORRUPTED NPY: {type(e).__name__}: {e}")

    n = min(len(x), len(y), len(theta))
    if n == 0:
        raise TrajectoryError("EMPTY ARRAYS")
    x, y, theta = x[:n], y[:n], theta[:n]

    elapsed = None
    if t_path.exists():
        try:
            ts = safe_load(t_path)
            n2 = min(n, len(ts))
            x, y, theta, ts = x[:n2], y[:n2], theta[:n2], ts[:n2]
            elapsed = ts - ts[0]
        except Exception:
            elapsed = None
    if elapsed is None:
        elapsed = np.arange(n) / SAMPLE_RATE_HZ

    return elapsed, x, y, theta


def check_success(elapsed, x, y, theta, x_target, y_target, theta_target_deg):
    """Same pos/heading/time threshold check used for real runs."""
    theta_target = np.deg2rad(theta_target_deg)
    heading_tol = np.deg2rad(HEADING_TOL_DEG)

    dist = np.sqrt((x - x_target) ** 2 + (y - y_target) ** 2)
    heading_error = np.abs(wrap_angle(theta - theta_target))

    pos_ok = dist <= POS_TOL
    heading_ok = heading_error <= heading_tol
    time_ok = elapsed <= TIME_TOL_S
    all_ok = pos_ok & heading_ok & time_ok

    idx_success = np.where(all_ok)[0]
    min_dist = float(dist.min())
    if len(idx_success) > 0:
        return 1, "success", float(elapsed[idx_success[0]]), min_dist
    return 0, "failure (time-limited)", None, min_dist


def resample_to_grid(elapsed, x, y, theta, common_t):
    """Resample one trajectory onto common_t, holding first/last pose past the ends."""
    xr = np.interp(common_t, elapsed, x, left=x[0], right=x[-1])
    yr = np.interp(common_t, elapsed, y, left=y[0], right=y[-1])
    theta_unwrapped = np.unwrap(theta)
    thr = np.interp(common_t, elapsed, theta_unwrapped,
                     left=theta_unwrapped[0], right=theta_unwrapped[-1])
    thr = wrap_angle(thr)
    return xr, yr, thr


def blend_pair(traj_a, traj_b, t):
    """
    Blend two already-resampled trajectories (same common_t grid) at
    fraction t (0 = pure A, 1 = pure B, 0.5 = average).
    traj_a, traj_b: (common_t, x, y, theta) tuples.
    """
    common_t, xa, ya, tha = traj_a
    _, xb, yb, thb = traj_b
    x_synth = (1 - t) * xa + t * xb
    y_synth = (1 - t) * ya + t * yb
    # circular-safe heading blend: shortest-path interpolation at every sample
    dtheta = wrap_angle(thb - tha)
    theta_synth = wrap_angle(tha + t * dtheta)
    return common_t, x_synth, y_synth, theta_synth


def find_axis_neighbors(targets):
    """
    targets: dict {(x,y,theta_deg): [folder, folder, ...]}
    Returns list of dicts describing each adjacent real-target pair that
    differs in exactly one coordinate, e.g.
      {"axis": "y", "fixed": (3, None, 0), "lo": (3,-3,0), "hi": (3,0,0)}
    """
    keys = list(targets.keys())
    pairs = []
    for axis_idx, axis_name in enumerate(["x", "y", "theta"]):
        if axis_name not in AXES_TO_INTERPOLATE:
            continue
        # group targets by the OTHER two coordinates held fixed
        def other_coords(k):
            return tuple(v for i, v in enumerate(k) if i != axis_idx)

        keys_sorted = sorted(keys, key=lambda k: (other_coords(k), k[axis_idx]))
        for _, group in groupby(keys_sorted, key=other_coords):
            group = sorted(group, key=lambda k: k[axis_idx])
            for lo, hi in zip(group, group[1:]):
                pairs.append({"axis": axis_name, "lo": lo, "hi": hi})
    return pairs


def build_target_key(lo, hi, axis, t):
    """Return the interpolated (x,y,theta_deg) target for fraction t along axis."""
    axis_idx = {"x": 0, "y": 1, "theta": 2}[axis]
    coords = list(lo)
    coords[axis_idx] = lo[axis_idx] + t * (hi[axis_idx] - lo[axis_idx])
    return tuple(coords)


def main():
    if len(sys.argv) < 3:
        print("Usage: python3 make_results_xy.py <root_dir> <output.xlsx>")
        sys.exit(1)

    root_dir = Path(sys.argv[1])
    out_path = Path(sys.argv[2])
    if not root_dir.is_dir():
        print(f"Root directory not found: {root_dir}")
        sys.exit(1)

    # ---- group real trial folders by (x, y, theta_deg) ----
    targets = {}
    skipped = []
    for folder in sorted(root_dir.rglob("test_*")):
        if not folder.is_dir():
            continue
        parsed = parse_folder_name(folder.name)
        if parsed is None:
            skipped.append(str(folder.relative_to(root_dir)))
            continue
        targets.setdefault(parsed, []).append(folder)

    if skipped:
        print(f"Skipped {len(skipped)} folder(s) not matching the naming pattern.")

    print(f"Found {len(targets)} distinct real targets:")
    for k, v in sorted(targets.items()):
        print(f"  {k}: {len(v)} trial(s)")

    # ---- real-target success probabilities (for reference) ----
    real_summary_rows = []
    real_run_cache = {}   # (target, folder) -> (elapsed,x,y,theta) or TrajectoryError
    for target, folders in targets.items():
        n_success, n_total, n_errors = 0, 0, 0
        for folder in folders:
            try:
                traj = load_trajectory(folder)
                real_run_cache[(target, folder)] = traj
                result, status, _, _ = check_success(*traj, *target)
                n_total += 1
                n_success += result
            except TrajectoryError as e:
                real_run_cache[(target, folder)] = e
                n_total += 1
                n_errors += 1
        prob = n_success / n_total if n_total else None
        real_summary_rows.append({
            "x_target": target[0], "y_target": target[1], "theta_target_deg": target[2],
            "n_trials": n_total, "n_success": n_success, "n_errors": n_errors,
            "probability": prob,
        })

    # ---- find neighbor pairs along each axis and synthesize in-between targets ----
    neighbor_pairs = find_axis_neighbors(targets)
    print(f"\nFound {len(neighbor_pairs)} adjacent real-target pair(s) to interpolate between.")

    synth_run_rows = []
    synth_summary_rows = []

    for pair in neighbor_pairs:
        axis, lo, hi = pair["axis"], pair["lo"], pair["hi"]
        folders_lo = targets[lo]
        folders_hi = targets[hi]
        n_use = min(len(folders_lo), len(folders_hi))
        if n_use == 0:
            continue

        # load + cache real trajectories for this pair (reuse cache from above)
        loaded_lo, loaded_hi = [], []
        for f in folders_lo[:n_use]:
            traj = real_run_cache.get((lo, f))
            loaded_lo.append((f, traj))
        for f in folders_hi[:n_use]:
            traj = real_run_cache.get((hi, f))
            loaded_hi.append((f, traj))

        for t in INTERP_FRACTIONS:
            synth_target = build_target_key(lo, hi, axis, t)
            xt, yt, tht = synth_target
            n_success = 0
            n_valid = 0
            for i in range(n_use):
                f_lo, traj_lo = loaded_lo[i]
                f_hi, traj_hi = loaded_hi[i]
                if isinstance(traj_lo, TrajectoryError) or isinstance(traj_hi, TrajectoryError):
                    reason = traj_lo if isinstance(traj_lo, TrajectoryError) else traj_hi
                    synth_run_rows.append({
                        "axis": axis, "t_fraction": t,
                        "x_target": xt, "y_target": yt, "theta_target_deg": tht,
                        "pair_index": i, "folder_lo": str(f_lo.name), "folder_hi": str(f_hi.name),
                        "result": 0, "status": f"SKIPPED (source error): {reason}",
                        "first_success_t_s": None, "min_dist_m": None,
                    })
                    continue

                elapsed_lo, x_lo, y_lo, th_lo = traj_lo
                elapsed_hi, x_hi, y_hi, th_hi = traj_hi

                t_end = max(elapsed_lo[-1], elapsed_hi[-1])
                common_t = np.arange(0, t_end + 1 / SAMPLE_RATE_HZ, 1 / SAMPLE_RATE_HZ)

                rl = resample_to_grid(elapsed_lo, x_lo, y_lo, th_lo, common_t)
                rh = resample_to_grid(elapsed_hi, x_hi, y_hi, th_hi, common_t)

                ct, xs, ys, ths = blend_pair((common_t, *rl), (common_t, *rh), t)
                result, status, first_t, min_dist = check_success(ct, xs, ys, ths, xt, yt, tht)

                synth_run_rows.append({
                    "axis": axis, "t_fraction": t,
                    "x_target": xt, "y_target": yt, "theta_target_deg": tht,
                    "pair_index": i, "folder_lo": str(f_lo.name), "folder_hi": str(f_hi.name),
                    "result": result, "status": status,
                    "first_success_t_s": first_t, "min_dist_m": min_dist,
                })
                n_valid += 1
                n_success += result

            prob = n_success / n_valid if n_valid else None
            synth_summary_rows.append({
                "axis": axis, "t_fraction": t,
                "x_target": xt, "y_target": yt, "theta_target_deg": tht,
                "from_lo": str(lo), "from_hi": str(hi),
                "n_synth_trajectories": n_valid, "n_success": n_success,
                "probability": prob,
            })
            print(f"  synthetic target {synth_target} (axis={axis}, t={t}): "
                  f"{n_success}/{n_valid} success -> p={prob}")

    # ---- write Excel: 3 sheets ----
    wb = openpyxl.Workbook()

    def write_sheet(ws, rows, headers, highlight_result_col=None):
        header_font = Font(name="Arial", bold=True)
        header_fill = PatternFill(start_color="DDDDDD", end_color="DDDDDD", fill_type="solid")
        for col, h in enumerate(headers, start=1):
            cell = ws.cell(row=1, column=col, value=h)
            cell.font = header_font
            cell.fill = header_fill
        for r, row in enumerate(rows, start=2):
            for c, key in enumerate(headers, start=1):
                cell = ws.cell(row=r, column=c, value=row.get(key))
                cell.font = Font(name="Arial")
                if highlight_result_col and key == highlight_result_col and row.get(key) is not None:
                    if row[key] == 1:
                        cell.fill = PatternFill(start_color="C6EFCE", end_color="C6EFCE", fill_type="solid")
                    elif row[key] == 0:
                        cell.fill = PatternFill(start_color="FFC7CE", end_color="FFC7CE", fill_type="solid")
        for col_cells in ws.columns:
            length = max(len(str(c.value)) if c.value is not None else 0 for c in col_cells)
            ws.column_dimensions[col_cells[0].column_letter].width = max(10, length + 2)

    ws1 = wb.active
    ws1.title = "real_targets_summary"
    write_sheet(ws1, real_summary_rows,
                ["x_target", "y_target", "theta_target_deg", "n_trials", "n_success",
                 "n_errors", "probability"])

    ws2 = wb.create_sheet("synthetic_targets_summary")
    write_sheet(ws2, synth_summary_rows,
                ["axis", "t_fraction", "x_target", "y_target", "theta_target_deg",
                 "from_lo", "from_hi", "n_synth_trajectories", "n_success", "probability"])

    ws3 = wb.create_sheet("synthetic_runs_detail")
    write_sheet(ws3, synth_run_rows,
                ["axis", "t_fraction", "x_target", "y_target", "theta_target_deg",
                 "pair_index", "folder_lo", "folder_hi", "result", "status",
                 "first_success_t_s", "min_dist_m"],
                highlight_result_col="result")

    wb.save(out_path)
    print(f"\nWrote {out_path}")
    print(f"  real_targets_summary: {len(real_summary_rows)} rows")
    print(f"  synthetic_targets_summary: {len(synth_summary_rows)} rows")
    print(f"  synthetic_runs_detail: {len(synth_run_rows)} rows")


if __name__ == "__main__":
    main()