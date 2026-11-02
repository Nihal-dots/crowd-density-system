"""Generate the LaTeX results fragment for the IEEE paper from the result files.

Reads research/results/ (eval_results.json, benchmark.json, train logs and
figures) and writes research/paper/generated_results.tex, to be pulled into
the paper with \\input{generated_results}. Measured numbers are only ever
read from the result files; anything that has not been produced yet is left
out and listed in the console report, so rerun this after each experiment.

Usage (from the project root):
    python research/make_paper_assets.py
"""

import csv
import json
import sys
from pathlib import Path

RESEARCH_DIR = Path(__file__).resolve().parent
RESULTS_DIR = RESEARCH_DIR / "results"
OUTPUT_PATH = RESEARCH_DIR / "paper" / "generated_results.tex"

# Literature values (not measured here): ShanghaiTech MAE / MSE as reported in the
# cited papers. Check them and the citation keys against your bibliography.
PUBLISHED = [
    ("MCNN", "zhang2016mcnn", {"A": (110.2, 173.2), "B": (26.4, 41.3)}),
    ("CSRNet", "li2018csrnet", {"A": (68.2, 115.0), "B": (10.6, 16.0)}),
]

included, missing = [], []


def load_json(name):
    path = RESULTS_DIR / name
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def tex(text):
    return str(text).replace("\\", "/").replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def figure(files, caption, label, width=r"\columnwidth", star=False):
    env = "figure*" if star else "figure"
    graphics = "\n".join(rf"  \includegraphics[width={width}]{{\resultsdir/{Path(f).as_posix()}}}" for f in files)
    return f"\\begin{{{env}}}[t]\n  \\centering\n{graphics}\n  \\caption{{{caption}}}\n  \\label{{{label}}}\n\\end{{{env}}}\n"


def table(caption, label, spec, header, rows, star=False):
    env = "table*" if star else "table"
    body = "\n".join("    " + row for row in rows)
    return (
        f"\\begin{{{env}}}[t]\n  \\caption{{{caption}}}\n  \\label{{{label}}}\n  \\centering\n"
        f"  \\begin{{tabular}}{{{spec}}}\n    \\hline\n    {header} \\\\\n    \\hline\n{body}\n    \\hline\n  \\end{{tabular}}\n\\end{{{env}}}\n"
    )


def accuracy_section(evaluation):
    """Measured MAE/MSE from evaluate.py next to published values."""
    parts = (evaluation or {}).get("parts", {})
    if evaluation and evaluation.get("partial"):
        missing.append("accuracy table: eval_results.json is a partial (--limit) run; rerun training/evaluate.py")
        parts = {}
    if not parts:
        missing.append("accuracy table: no measured results (run training/evaluate.py after training)")
        return ""
    for part in ("A", "B"):
        if part not in parts:
            missing.append(f"accuracy table: Part {part} not evaluated yet (shown as --)")

    def cells(values):
        return " & ".join(f"{v:.1f}" if v is not None else "--" for v in values)

    rows = [
        rf"{name}~\cite{{{key}}} & Published & {cells(values['A'] + values['B'])} \\"
        for name, key, values in PUBLISHED
    ]
    measured = [parts.get(p, {}).get(m) for p in ("A", "B") for m in ("mae", "mse")]
    rows.append(r"\hline")
    rows.append(rf"CSRNet (our implementation) & \textbf{{Measured}} & {cells(measured)} \\")
    epochs = ", ".join(f"Part {p}: epoch {info['checkpoint']['epoch']}, {info['num_images']} test images" for p, info in parts.items())
    included.append(f"accuracy table ({', '.join('Part ' + p for p in parts)})")
    return table(
        "Counting accuracy on ShanghaiTech. Rows marked Published are taken from the cited papers; "
        f"the Measured row was produced by our own training and evaluation ({epochs}).",
        "tab:accuracy", "llcccc",
        r"Method & Source & \multicolumn{2}{c}{Part A} & \multicolumn{2}{c}{Part B} \\" "\n"
        r"     &        & MAE & MSE & MAE & MSE",
        rows, star=True,
    )


def training_section():
    out = ""
    for part in ("A", "B"):
        curves, log = RESULTS_DIR / f"training_curves_{part}.png", RESULTS_DIR / f"train_log_{part}.csv"
        if not (curves.is_file() and log.is_file()):
            missing.append(f"training curves: Part {part} has not been trained")
            continue
        with open(log, newline="") as f:
            history = list(csv.DictReader(f))
        best = min(history, key=lambda row: float(row["mae"]))
        out += figure(
            [curves.name],
            f"Training loss and test MAE on Part {part} over {len(history)} epochs "
            f"(best MAE {float(best['mae']):.2f} at epoch {best['epoch']}).",
            f"fig:training_{part.lower()}",
        )
        included.append(f"training curves Part {part} ({len(history)} epochs)")
    return out


def examples_section(evaluation):
    out = ""
    for part, info in (evaluation or {}).get("parts", {}).items():
        if evaluation.get("partial"):
            break
        if (RESULTS_DIR / f"eval_examples_{part}.png").is_file():
            out += figure([f"eval_examples_{part}.png"],
                          f"Part {part} test examples from sparse to dense: image, ground-truth density and predicted density with counts.",
                          f"fig:examples_{part.lower()}", width=r"\textwidth", star=True)
            included.append(f"prediction examples Part {part}")
        if (RESULTS_DIR / f"eval_worst_{part}.png").is_file():
            out += figure([f"eval_worst_{part}.png"], f"The five Part {part} test images with the largest count error.",
                          f"fig:worst_{part.lower()}", width=r"\textwidth", star=True)
        rows = [rf"{tex(w['image'])} & {w['true_count']:.0f} & {w['predicted_count']:.1f} & {w['error']:+.1f} \\" for w in info["worst_images"]]
        out += table(f"Largest count errors on the Part {part} test split (mean count error {info['mean_count_error']:+.2f}).",
                     f"tab:worst_{part.lower()}", "lrrr", "Image & True & Predicted & Error", rows)
        included.append(f"failure cases Part {part}")
    if not out:
        missing.append("prediction examples and failure cases: need eval_results.json from training/evaluate.py")
    return out


def ablation_section(benchmark, key, name, label):
    experiment = (benchmark or {}).get(key)
    if not experiment:
        missing.append(f"{name} ablation: not run (experiments/run_benchmarks.py)")
        return ""
    settings = experiment["settings"]
    if settings.get("debug_limit"):
        missing.append(f"{name} ablation: benchmark.json holds a debug (--limit) run, not a real result")
        return ""
    rows = [rf"{tex(variant)} & {v['mae']:.2f} & {v['mse']:.2f} \\" for variant, v in experiment["variants"].items()]
    included.append(f"{name} ablation")
    return table(
        f"Ablation: {name} on Part {settings['part']} ({settings['epochs']} epochs, batch {settings['batch_size']}, identical settings).",
        label, "lcc", "Variant & MAE & MSE", rows,
    )


def latency_section(benchmark):
    devices = (benchmark or {}).get("latency", {}).get("devices", {})
    if not devices:
        missing.append("latency table: not run (experiments/run_benchmarks.py --experiments latency)")
        return ""
    names = {"cpu": "CPU", "cuda": "GPU"}
    rows = [
        rf"{names.get(device, device)} & {name} & {r['mean_ms']:.1f} $\pm$ {r['std_ms']:.1f} & {r['fps']:.2f} \\"
        for device, info in devices.items() for name, r in info["results"].items()
    ]
    hardware = "; ".join(f"{names.get(d, d)}: {tex(info['hardware'])}" for d, info in devices.items())
    runs = benchmark["latency"]["settings"]["timed_runs"]
    if "cuda" not in devices:
        missing.append("latency table: GPU rows missing (rerun the latency experiment on a GPU machine)")
    included.append(f"latency table ({', '.join(names.get(d, d) for d in devices)})")
    return table(f"Inference latency, mean $\\pm$ std over {runs} runs after warm-up ({hardware}).",
                 "tab:latency", "llcc", "Device & Resolution & Latency (ms) & FPS", rows)


def alerts_section(benchmark):
    experiment = (benchmark or {}).get("alerts")
    if not experiment or not experiment.get("videos"):
        missing.append("alert table: not run (experiments/run_benchmarks.py --experiments alerts)")
        return ""
    rows = [
        rf"{tex(Path(name).stem)} & {v['frames_processed']} & {v['without_hysteresis']['WARNING']} & {v['without_hysteresis']['CRITICAL']} "
        rf"& {v['with_hysteresis']['WARNING']} & {v['with_hysteresis']['CRITICAL']} \\"
        for name, v in experiment["videos"].items()
    ]
    s = experiment["settings"]
    included.append(f"alert table ({len(rows)} videos, checkpoint epoch {s['checkpoint_epoch']})")
    return table(
        f"WARNING (W) and CRITICAL (C) events on Indian footage without and with hysteresis "
        f"({s['consecutive_frames']} consecutive frames to raise, clear below {s['clear_ratio']}$T$; {tex(s['threshold_note'])}).",
        "tab:alerts", "lrcccc",
        r"Video & Frames & \multicolumn{2}{c}{No hysteresis} & \multicolumn{2}{c}{Hysteresis} \\" "\n"
        r"     &        & W & C & W & C",
        rows,
    )


def dashboard_section():
    shots = sorted(p.name for p in RESULTS_DIR.glob("dashboard_*.png"))
    if not shots:
        missing.append("dashboard screenshots: save them as research/results/dashboard_<name>.png")
        return ""
    included.append(f"dashboard screenshots ({len(shots)})")
    return figure(shots, "The monitoring dashboard running on Indian crowd footage.", "fig:dashboard", width=r"\textwidth", star=True)


def main():
    evaluation, benchmark = load_json("eval_results.json"), load_json("benchmark.json")
    sections = [
        accuracy_section(evaluation),
        training_section(),
        examples_section(evaluation),
        ablation_section(benchmark, "dilation_ablation", "back-end dilation", "tab:ablation_dilation"),
        ablation_section(benchmark, "kernel_ablation", "density map kernel", "tab:ablation_kernel"),
        latency_section(benchmark),
        alerts_section(benchmark),
        dashboard_section(),
    ]
    header = (
        "% Generated by research/make_paper_assets.py from research/results/ - do not edit by hand.\n"
        "% Needs \\usepackage{graphicx}. Set \\resultsdir to the results folder relative to the paper.\n"
        "\\providecommand{\\resultsdir}{../results}\n\n"
    )
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(header + "\n".join(s for s in sections if s), encoding="utf-8")

    print(f"Wrote {OUTPUT_PATH}")
    print("\nIncluded:")
    for item in included or ["(nothing yet)"]:
        print(f"  + {item}")
    print("\nMissing:")
    for item in missing or ["(nothing)"]:
        print(f"  - {item}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
