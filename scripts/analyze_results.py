"""
Turn saved results into the tables and plots the README reports.

Inputs:  results/tables/baseline_cv.csv, results/robustness_records.csv.gz
Outputs: results/tables/*.csv, results/tables/*.md, results/plots/*.png

Every number here is computed from those files; nothing is typed in by hand.
Metrics are computed per fold, then reported as mean +/- std across the 5
subject-wise folds.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import accuracy_score, f1_score  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wrist_sensor_robustness import config  # noqa: E402
from wrist_sensor_robustness.corruptions import SEVERITY_LEVELS, SEVERITY_UNITS  # noqa: E402

TABLES = config.RESULTS_DIR / "tables"
PLOTS = config.RESULTS_DIR / "plots"

# Validated categorical palette (dataviz reference palette, light mode): blue, orange, aqua.
# Aqua is below 3:1 contrast on the light surface, so every series also carries a marker
# and a direct label - identity is never color-alone.
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID, AXIS = "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7"
MODEL_STYLE = {"gbm_features": (C1, "o", "GBM (features)"), "cnn_1d": (C2, "s", "1D CNN")}
CORR_TITLES = {"dropout": "Sample dropout", "noise": "Additive noise", "axis_swap": "Axis swap / sign flip",
               "rotation": "Orientation rotation", "rate_drift": "Sampling-rate drift", "clipping": "Saturation / clipping"}

MIX_CLEAN_FRACTION = 0.7  # deployment-stream scenario: 70% clean windows, 30% corrupted


def style_axes(ax):
    ax.set_facecolor("#fcfcfb")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def metrics(df, pred_col):
    return accuracy_score(df["true"], df[pred_col]), f1_score(df["true"], df[pred_col], average="macro", zero_division=0)


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def pm(mean, std, pct=True):
    return f"{100 * mean:.1f} ± {100 * std:.1f}" if pct else f"{mean:.3f} ± {std:.3f}"


# --------------------------------------------------------------------------- baseline
def baseline_table():
    cv = pd.read_csv(TABLES / "baseline_cv.csv")
    rows = []
    for model in ("gbm_features", "cnn_1d"):
        for split in ("subject", "random_window"):
            g = cv[(cv.model == model) & (cv.split == split)]
            rows.append({"model": MODEL_STYLE[model][2],
                         "split": "subject-wise (5-fold, by person)" if split == "subject" else "random windows (leaky)",
                         "accuracy %": pm(g.accuracy.mean(), g.accuracy.std(ddof=0)),
                         "macro-F1 %": pm(g.macro_f1.mean(), g.macro_f1.std(ddof=0))})
        s = cv[(cv.model == model) & (cv.split == "subject")]
        r = cv[(cv.model == model) & (cv.split == "random_window")]
        rows.append({"model": MODEL_STYLE[model][2], "split": "**leakage inflation**",
                     "accuracy %": f"+{100 * (r.accuracy.mean() - s.accuracy.mean()):.1f} pts",
                     "macro-F1 %": f"+{100 * (r.macro_f1.mean() - s.macro_f1.mean()):.1f} pts"})
    out = pd.DataFrame(rows)
    (TABLES / "baseline.md").write_text(md_table(out) + "\n")
    folds = cv[cv.split == "subject"].pivot(index="fold", columns="model", values="accuracy").round(3)
    (TABLES / "baseline_per_fold_accuracy.md").write_text(md_table(folds.reset_index()) + "\n")
    print(md_table(out))
    return cv


# ------------------------------------------------------------------------- robustness
def robustness(rec):
    rows = []
    for (kind, level, fold), g in rec.groupby(["corruption", "level", "fold"]):
        for model in ("gbm", "cnn"):
            acc, f1 = metrics(g, f"{model}_pred")
            rows.append({"corruption": kind, "level": level, "fold": fold, "model": model,
                         "accuracy": acc, "macro_f1": f1, "flag_rate": g["flagged"].mean()})
    per_fold = pd.DataFrame(rows)
    summ = (per_fold.groupby(["corruption", "level", "model"])[["accuracy", "macro_f1", "flag_rate"]]
            .agg(["mean", "std"]).reset_index())
    summ.columns = ["_".join(c).rstrip("_") for c in summ.columns]
    summ["severity"] = [str(SEVERITY_LEVELS[k][lv]) for k, lv in zip(summ.corruption, summ.level)]
    summ.to_csv(TABLES / "robustness.csv", index=False)

    # Compact table: macro-F1 at each level, plus drop from clean at the worst level.
    md_rows = []
    for kind in SEVERITY_LEVELS:
        for model, label in (("gbm", "GBM"), ("cnn", "CNN")):
            g = summ[(summ.corruption == kind) & (summ.model == model)].sort_values("level")
            f1 = g["macro_f1_mean"].to_numpy()
            md_rows.append({"corruption": CORR_TITLES[kind], "model": label,
                            **{f"L{lv}": f"{100 * v:.1f}" for lv, v in zip(g.level, f1)},
                            "drop at L5 (pts)": f"{100 * (f1[0] - f1[-1]):.1f}"})
    md = pd.DataFrame(md_rows)
    levels_md = pd.DataFrame([{"corruption": CORR_TITLES[k], "unit": SEVERITY_UNITS[k],
                               **{f"L{i}": str(v) for i, v in enumerate(lv)}} for k, lv in SEVERITY_LEVELS.items()])
    (TABLES / "robustness_macro_f1.md").write_text(md_table(md) + "\n")
    (TABLES / "severity_levels.md").write_text(md_table(levels_md) + "\n")

    fig, axes = plt.subplots(2, 3, figsize=(12, 6.6), sharey=True)
    for ax, kind in zip(axes.ravel(), SEVERITY_LEVELS):
        style_axes(ax)
        # axis_swap levels are distinct transforms, not an ordered severity: markers, no connecting line
        categorical = kind == "axis_swap"
        ends = {}
        for model, key in (("gbm_features", "gbm"), ("cnn_1d", "cnn")):
            color, marker, label = MODEL_STYLE[model]
            g = summ[(summ.corruption == kind) & (summ.model == key)].sort_values("level")
            m, s = 100 * g["macro_f1_mean"].to_numpy(), 100 * g["macro_f1_std"].to_numpy()
            if categorical:
                ax.errorbar(g.level, m, yerr=s, color=color, linestyle="none", marker=marker, markersize=7,
                            markeredgecolor="#fcfcfb", markeredgewidth=1.5, elinewidth=1.2, capsize=0, label=label)
            else:
                ax.fill_between(g.level, m - s, m + s, color=color, alpha=0.12, linewidth=0)
                ax.plot(g.level, m, color=color, linewidth=2, marker=marker, markersize=6,
                        markeredgecolor="#fcfcfb", markeredgewidth=1.5, label=label)
            ends[key] = (g.level.iloc[-1], m[-1])
        # direct end labels, nudged apart when the two series finish close together
        (xg, yg), (xc, yc) = ends["gbm"], ends["cnn"]
        nudge = 4 if abs(yg - yc) < 7 else 0
        for (x, yv), up in (((xg, yg), yg >= yc), ((xc, yc), yc > yg)):
            ax.annotate(f"{yv:.0f}", (x, yv), xytext=(7, nudge if up else -nudge), textcoords="offset points",
                        va="center", fontsize=8, color=INK)
        ax.set_title(CORR_TITLES[kind], fontsize=10, color=INK, loc="left")
        ax.set_xticks(range(6))
        ax.set_xticklabels([str(v) if v is not None else "clean" for v in SEVERITY_LEVELS[kind]],
                           fontsize=7, rotation=30 if kind == "axis_swap" else 0)
        ax.set_xlabel(SEVERITY_UNITS[kind], fontsize=8, color=MUTED)
        ax.set_ylim(0, 100)
        ax.set_xlim(-0.2, 5.6)
    for ax in axes[:, 0]:
        ax.set_ylabel("macro-F1 (%)", fontsize=9, color=MUTED)
    axes[0, 0].legend(frameon=False, fontsize=8, loc="lower left")
    fig.suptitle("Macro-F1 under simulated sensor corruption (subject-wise CV, mean ± std over 5 folds)",
                 fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(PLOTS / "robustness_macro_f1.png", dpi=150, facecolor="#fcfcfb")
    plt.close(fig)
    return summ


# --------------------------------------------------------------------------- validator
def detection(summ):
    g = summ[summ.model == "gbm"]  # flag rate is model-independent; take one copy
    mat = g.pivot(index="corruption", columns="level", values="flag_rate_mean").loc[list(SEVERITY_LEVELS)]
    out = mat.map(lambda v: f"{100 * v:.1f}").reset_index()
    out["corruption"] = out["corruption"].map(CORR_TITLES)
    out.columns = ["corruption"] + [f"L{c}" for c in mat.columns]
    (TABLES / "validator_flag_rate.md").write_text(md_table(out) + "\n")

    fig, ax = plt.subplots(figsize=(7, 3.6))
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("blue", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
    ax.imshow(100 * mat.to_numpy(), cmap=cmap, vmin=0, vmax=100, aspect="auto")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = 100 * mat.iloc[i, j]
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=8, color="#fcfcfb" if v > 55 else INK)
    ax.set_yticks(range(len(mat)))
    ax.set_yticklabels([CORR_TITLES[k] for k in mat.index], fontsize=8, color=INK)
    ax.set_xticks(range(mat.shape[1]))
    ax.set_xticklabels(["clean"] + [f"L{c}" for c in mat.columns[1:]], fontsize=8, color=MUTED)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Validator flag rate (%) by corruption and severity", fontsize=10, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(PLOTS / "validator_flag_rate.png", dpi=150, facecolor="#fcfcfb")
    plt.close(fig)


# ------------------------------------------------------------------------------ gating
def mixed_stream(rec, seed=config.RANDOM_SEED):
    """
    One version of each test window: clean with probability 0.7, otherwise a
    uniformly random (corruption, level 1-5). Seeded, so every gating method
    is compared on the identical stream.
    """
    rng = np.random.default_rng(seed)
    kinds = list(SEVERITY_LEVELS)
    base = rec[rec.level == 0][["fold", "window_idx"]].drop_duplicates().reset_index(drop=True)
    corrupted = rng.random(len(base)) >= MIX_CLEAN_FRACTION
    base["corruption"] = np.where(corrupted, np.array(kinds)[rng.integers(0, len(kinds), len(base))], "dropout")
    base["level"] = np.where(corrupted, rng.integers(1, 6, len(base)), 0)
    keyed = rec.set_index(["fold", "window_idx", "corruption", "level"])
    # level-0 rows are identical across corruption kinds (same clean window); any one will do
    stream = keyed.loc[list(base.itertuples(index=False, name=None))].reset_index()
    stream["is_corrupted"] = stream["level"] > 0
    return stream


def rule_failed(df) -> np.ndarray:
    """True where a physical rule fired (anything other than the learned anomaly score)."""
    reasons = df["reasons"].fillna("").str.split("|")
    return reasons.map(lambda rs: any(r and r != "anomalous_pattern" for r in rs)).to_numpy()


def keep_top(score: np.ndarray, n_reject: int, tiebreak: np.ndarray) -> np.ndarray:
    keep = np.ones(len(score), bool)
    keep[np.lexsort((tiebreak, score))[:n_reject]] = False
    return keep


def gating(rec):
    stream = mixed_stream(rec)
    stream["rule_failed"] = rule_failed(stream)
    rows, detail, curves = [], [], {}
    for model, label in (("gbm", "GBM (features)"), ("cnn", "1D CNN")):
        per_fold = []
        for fold, g in stream.groupby("fold"):
            correct = (g[f"{model}_pred"] == g["true"]).to_numpy()
            conf = g[f"{model}_conf"].to_numpy()
            tie = g["window_idx"].to_numpy()
            acc_all, f1_all = metrics(g, f"{model}_pred")
            keep_v = ~g["flagged"].to_numpy()
            reject = 1 - keep_v.mean()
            k = int(round(reject * len(g)))
            acc_v, f1_v = metrics(g[keep_v], f"{model}_pred")
            # Confidence gating at the SAME rejection rate, for a fair comparison.
            keep_c = keep_top(conf, k, tie)
            acc_c, f1_c = metrics(g[keep_c], f"{model}_pred")
            # Hybrid at the same rejection rate: physical-rule failures are always
            # rejected first, the remaining budget goes to the lowest-confidence windows.
            keep_h = keep_top(np.where(g["rule_failed"].to_numpy(), -1.0, conf), k, tie)
            acc_h, f1_h = metrics(g[keep_h], f"{model}_pred")
            corr = g["is_corrupted"].to_numpy()
            per_fold.append({
                "acc_all": acc_all, "f1_all": f1_all, "acc_val": acc_v, "f1_val": f1_v,
                "acc_conf": acc_c, "f1_conf": f1_c, "acc_hyb": acc_h, "f1_hyb": f1_h, "reject": reject,
                "reject_clean": 1 - keep_v[~corr].mean(), "reject_corrupt": 1 - keep_v[corr].mean(),
                "err_rate_rejected": 1 - correct[~keep_v].mean() if (~keep_v).any() else np.nan,
                "err_rate_kept": 1 - correct[keep_v].mean(),
                "corrupt_kept_val": corr[keep_v].mean(), "corrupt_kept_conf": corr[keep_c].mean(),
                "corrupt_kept_hyb": corr[keep_h].mean(),
            })
        pf = pd.DataFrame(per_fold)
        m, s = pf.mean(), pf.std(ddof=0)
        for gate, a, f, ck in (("none", "acc_all", "f1_all", None),
                               ("validator", "acc_val", "f1_val", "corrupt_kept_val"),
                               ("confidence", "acc_conf", "f1_conf", "corrupt_kept_conf"),
                               ("hybrid: rules + confidence", "acc_hyb", "f1_hyb", "corrupt_kept_hyb")):
            rows.append({
                "model": label, "gate": gate,
                "windows rejected %": "0.0" if gate == "none" else pm(m.reject, s.reject),
                "accuracy %": pm(m[a], s[a]), "macro-F1 %": pm(m[f], s[f]),
                "corrupted share of kept %": f"{100 * (m[ck] if ck else 0.3):.1f}" if ck else "30.0 (by construction)",
            })
        detail.append({"model": label,
                       "validator rejects: clean / corrupted windows %": f"{100 * m.reject_clean:.1f} / {100 * m.reject_corrupt:.1f}",
                       "model error rate: rejected vs kept %": f"{100 * m.err_rate_rejected:.1f} vs {100 * m.err_rate_kept:.1f}"})

        # Risk-coverage curves, pooled across folds: keep the top-q windows by each score.
        g = stream
        correct = (g[f"{model}_pred"] == g["true"]).to_numpy()
        curves[label] = {}
        conf_all = g[f"{model}_conf"].to_numpy()
        for name, score in (("validator quality score", g["quality_score"].to_numpy()),
                            ("model confidence", conf_all),
                            ("hybrid: rules, then confidence", np.where(g["rule_failed"].to_numpy(), -1.0, conf_all))):
            # tie-break by window index so the curve is deterministic
            order = np.lexsort((g["window_idx"].to_numpy(), -score))
            cum = np.cumsum(correct[order]) / np.arange(1, len(order) + 1)
            cov = np.arange(1, len(order) + 1) / len(order)
            curves[label][name] = (cov, cum)

    table = pd.DataFrame(rows)
    (TABLES / "gating.md").write_text(md_table(table) + "\n")
    table.to_csv(TABLES / "gating.csv", index=False)
    (TABLES / "gating_detail.md").write_text(md_table(pd.DataFrame(detail)) + "\n")
    print(md_table(table))
    print(md_table(pd.DataFrame(detail)))

    op_coverage = 100 * (1 - stream["flagged"].mean())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for ax, (label, cv) in zip(axes, curves.items()):
        style_axes(ax)
        for (name, (cov, acc)), color, marker in zip(cv.items(), (C1, C2, C3), ("o", "s", "^")):
            idx = np.searchsorted(cov, np.linspace(0.1, 1.0, 10)) - 1
            ax.plot(100 * cov, 100 * acc, color=color, linewidth=2, marker=marker, markevery=list(idx),
                    markersize=6, markeredgecolor="#fcfcfb", markeredgewidth=1.5, label=name)

        ax.set_title(f"{label}: accuracy on kept windows vs coverage", fontsize=10, color=INK, loc="left")
        ax.set_xlabel("coverage: % of windows kept (stream: 70% clean, 30% corrupted)", fontsize=8, color=MUTED)
        ax.set_xlim(40, 101)
        ax.set_ylim(55, 100)
        ax.axvline(op_coverage, color=MUTED, linewidth=0.8, linestyle=":")
        ax.annotate(f"validator flags {100 - op_coverage:.1f}% of windows", (op_coverage, 56), xytext=(-4, 4),
                    textcoords="offset points", ha="right", fontsize=7, color=MUTED)
    axes[0].set_ylabel("accuracy on kept windows (%)", fontsize=9, color=MUTED)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, ["rank by " + lab for lab in labels], frameon=False, fontsize=8, loc="lower center", ncol=3)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(PLOTS / "gating_risk_coverage.png", dpi=150, facecolor="#fcfcfb")
    plt.close(fig)

    # Where the validator's rejections come from.
    reasons = (stream[stream.flagged].assign(reason=lambda d: d.reasons.str.split("|")).explode("reason")
               .groupby(["corruption", "level", "reason"]).size().rename("n").reset_index())
    reasons.to_csv(TABLES / "gating_reject_reasons.csv", index=False)
    return stream


def harm_vs_detection(summ):
    """For each corruption at level 3: how much it hurts the GBM, and how often the validator catches it."""
    rows = []
    g = summ[summ.model == "gbm"]
    clean = g[g.level == 0]["macro_f1_mean"].iloc[0]
    for kind in SEVERITY_LEVELS:
        for lvl in (1, 3, 5):
            r = g[(g.corruption == kind) & (g.level == lvl)].iloc[0]
            rows.append({"corruption": CORR_TITLES[kind], "level": f"L{lvl} ({SEVERITY_LEVELS[kind][lvl]})",
                         "GBM macro-F1 loss (pts)": f"{100 * (clean - r.macro_f1_mean):.1f}",
                         "validator flag rate %": f"{100 * r.flag_rate_mean:.1f}"})
    out = pd.DataFrame(rows)
    (TABLES / "harm_vs_detection.md").write_text(md_table(out) + "\n")
    print(md_table(out))


def main():
    TABLES.mkdir(parents=True, exist_ok=True)
    PLOTS.mkdir(parents=True, exist_ok=True)
    print("== baseline ==")
    if (TABLES / "baseline_cv.csv").exists():
        baseline_table()
    else:
        print("baseline_cv.csv not found yet; run scripts/run_baseline.py")
    rec_path = config.RESULTS_DIR / "robustness_records.csv.gz"
    if not rec_path.exists():
        print("no robustness records yet; run scripts/run_robustness.py")
        return
    rec = pd.read_csv(rec_path, dtype={"severity": str})
    print("\n== robustness ==")
    summ = robustness(rec)
    print((TABLES / "robustness_macro_f1.md").read_text())
    print("== validator flag rate ==")
    detection(summ)
    print((TABLES / "validator_flag_rate.md").read_text())
    print("== harm vs detection ==")
    harm_vs_detection(summ)
    print("== gating (mixed stream) ==")
    gating(rec)


if __name__ == "__main__":
    main()
