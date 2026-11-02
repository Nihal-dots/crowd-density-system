"""Build research/results/result.pdf from the predictions cached by evaluate_all.py.

Crowd counting is regression, so the headline metrics are MAE / RMSE of the
per-image count. For the classification view (confusion matrix, accuracy,
precision, recall, F1) the true and predicted counts are both binned into the
same fixed crowd-density levels (LEVELS below), and an alert view scores the
binary decision "count >= T" over a range of thresholds T. Every number in the
PDF is computed here from the cached predictions; nothing is typed in by hand
except the published baselines in LITERATURE, which are cited as such.

Usage (from the project root, after research/evaluate_all.py):
    python research/make_results_pdf.py
"""

import json
import math
import sys
from datetime import date
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from PIL import Image  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.enums import TA_CENTER  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet  # noqa: E402
from reportlab.lib.units import cm  # noqa: E402
from reportlab.platypus import (  # noqa: E402
    Image as RLImage, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)
from sklearn.metrics import (  # noqa: E402
    accuracy_score, cohen_kappa_score, confusion_matrix, f1_score, precision_recall_fscore_support,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from training.dataset import RoboflowDataset, ShanghaiTechDataset  # noqa: E402
from training.inspect_dataset import RESULTS_DIR  # noqa: E402

CACHE_DIR = RESULTS_DIR / "eval_cache"
FIG_DIR = RESULTS_DIR / "pdf_figures"
PDF_PATH = RESULTS_DIR / "result.pdf"
CHECKPOINT = PROJECT_ROOT / "experiments" / "checkpoints" / "all" / "best_model.pth"

SPLITS = {  # cache name -> (label, dataset factory for the qualitative figure)
    "shanghaitech_A_test": ("ShanghaiTech A", lambda: ShanghaiTechDataset("A", "test")),
    "shanghaitech_B_test": ("ShanghaiTech B", lambda: ShanghaiTechDataset("B", "test")),
    "roboflow_test": ("Roboflow", lambda: RoboflowDataset("test")),
}
# Crowd-density levels by people per image: [lower, upper)
LEVELS = [("Low", 0, 50), ("Medium", 50, 150), ("High", 150, 500), ("Very high", 500, math.inf)]
LEVEL_NAMES = [name for name, _, _ in LEVELS]
# Published results on the ShanghaiTech test sets: (method, source, A MAE, A RMSE, B MAE, B RMSE)
LITERATURE = [
    ("MCNN", "Zhang et al., CVPR 2016", 110.2, 173.2, 26.4, 41.3),
    ("CSRNet", "Li et al., CVPR 2018", 68.2, 115.0, 10.6, 16.0),
]
BOOTSTRAP = 2000

# Reference palette (dataviz skill): categorical slots 1-3, ink, sequential blue
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
INK, INK_2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#ffffff"
BLUES = LinearSegmentedColormap.from_list(
    "blues", ["#f4f8fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
    "axes.edgecolor": AXIS, "axes.labelcolor": INK_2, "axes.titlecolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
})


# ---------------------------------------------------------------- metrics

def level_range(lower, upper):
    return f"{lower}+" if math.isinf(upper) else f"{lower}-{upper - 1}"


def level_of(counts):
    edges = [lower for _, lower, _ in LEVELS[1:]]
    return np.digitize(counts, edges)


def counting_metrics(true, pred, rng):
    err = pred - true
    abs_err = np.abs(err)
    rel = abs_err / true
    idx = rng.integers(0, len(true), size=(BOOTSTRAP, len(true)))
    boot = abs_err[idx].mean(axis=1)
    ss_res, ss_tot = (err**2).sum(), ((true - true.mean())**2).sum()
    return {
        "n": len(true),
        "mae": abs_err.mean(),
        "mae_lo": np.percentile(boot, 2.5),
        "mae_hi": np.percentile(boot, 97.5),
        "rmse": math.sqrt((err**2).mean()),
        "bias": err.mean(),
        "mape": 100 * rel.mean(),
        "r2": 1 - ss_res / ss_tot,
        "pearson": float(np.corrcoef(true, pred)[0, 1]),
        "within10": 100 * (rel <= 0.10).mean(),
        "within20": 100 * (rel <= 0.20).mean(),
        "count_acc": 100 * np.clip(1 - rel, 0, None).mean(),
        "true_mean": true.mean(),
        "pred_mean": pred.mean(),
    }


def classification_metrics(true, pred):
    yt, yp = level_of(true), level_of(pred)
    labels = list(range(len(LEVELS)))
    present = sorted(set(yt) | set(yp))  # a level absent from both would add a meaningless 0 to the macro mean
    p, r, f, s = precision_recall_fscore_support(yt, yp, labels=labels, zero_division=0)
    return {
        "cm": confusion_matrix(yt, yp, labels=labels),
        "accuracy": 100 * accuracy_score(yt, yp),
        "within_one": 100 * (np.abs(yt - yp) <= 1).mean(),
        "macro_f1": 100 * f1_score(yt, yp, labels=present, average="macro", zero_division=0),
        "weighted_f1": 100 * f1_score(yt, yp, labels=labels, average="weighted", zero_division=0),
        "kappa": cohen_kappa_score(yt, yp, labels=labels, weights="quadratic"),
        "precision": 100 * p, "recall": 100 * r, "f1": 100 * f, "support": s,
    }


def alert_scores(true, pred, thresholds):
    """Precision / recall / F1 of the binary decision count >= T, for each T."""
    out = []
    for t in thresholds:
        yt, yp = true >= t, pred >= t
        tp, fp, fn = (yt & yp).sum(), (~yt & yp).sum(), (yt & ~yp).sum()
        prec = tp / (tp + fp) if tp + fp else np.nan
        rec = tp / (tp + fn) if tp + fn else np.nan
        f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else np.nan
        out.append((t, yt.sum(), prec, rec, f1))
    return np.array(out, dtype=float)


# ---------------------------------------------------------------- figures

def save(fig, name):
    path = FIG_DIR / name
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_training(history, best_epoch):
    epochs = [h["epoch"] for h in history]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.2))
    ax1.plot(epochs, [h["train_loss"] for h in history], color=SERIES[0], lw=2)
    ax1.set(title="Training loss per image", xlabel="Epoch", ylabel="Sum-reduced MSE loss")
    best = next(h for h in history if h["epoch"] == best_epoch)
    ax2.plot(epochs, [h["mae"] for h in history], color=SERIES[0], lw=2,
             label=f"MAE (best {best['mae']:.1f}, epoch {best_epoch})")
    ax2.plot(epochs, [h["mse"] for h in history], color=MUTED, lw=2, label="RMSE")
    ax2.scatter([best_epoch], [best["mae"]], s=40, color=SERIES[0], edgecolor=SURFACE, lw=1.5, zorder=3)
    ax2.set(title="Validation error (combined validation set)", xlabel="Epoch", ylabel="Heads per image")
    ax2.legend(loc="upper right")
    for ax in (ax1, ax2):
        ax.set_xlim(0.5, epochs[-1] + 0.5)
    fig.tight_layout()
    return save(fig, "training_curves.png")


def fig_scatter(data, metrics):
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
    for ax, (key, d), color in zip(axes, data.items(), SERIES):
        upper = max(d["true"].max(), d["pred"].max()) * 1.05
        ax.plot([0, upper], [0, upper], color=MUTED, lw=1, ls="--", label="perfect prediction")
        ax.scatter(d["true"], d["pred"], s=12, color=color, alpha=0.55, edgecolor="none")
        ax.set(xlim=(0, upper), ylim=(0, upper), aspect="equal", xlabel="True count", ylabel="Predicted count",
               title=d["label"])
        m = metrics[key]
        ax.text(0.04, 0.96, f"MAE {m['mae']:.1f}\nRMSE {m['rmse']:.1f}\nR$^2$ {m['r2']:.3f}", transform=ax.transAxes,
                va="top", fontsize=8, color=INK_2)
    fig.tight_layout()
    return save(fig, "scatter.png")


def fig_error_by_level(data):
    """Box plot of the relative counting error per density level and dataset."""
    fig, ax = plt.subplots(figsize=(9, 3.3))
    width = 0.24
    handles = []
    for i, (d, color) in enumerate(zip(data.values(), SERIES)):
        lv = level_of(d["true"])
        rel = 100 * (d["pred"] - d["true"]) / d["true"]
        groups = [(k, rel[lv == k]) for k in range(len(LEVELS)) if (lv == k).sum() >= 3]
        bp = ax.boxplot([g for _, g in groups], positions=[k + (i - 1) * width for k, _ in groups], widths=width * 0.85,
                        patch_artist=True, showfliers=False, medianprops={"color": INK, "lw": 1.2},
                        whiskerprops={"color": color}, capprops={"color": color},
                        boxprops={"facecolor": color, "edgecolor": color, "alpha": 0.75})
        handles.append((bp["boxes"][0], d["label"]))
    ax.axhline(0, color=INK_2, lw=1)
    ax.set_xticks(range(len(LEVELS)), [f"{n}\n({level_range(lo, hi)})" for n, lo, hi in LEVELS])
    ax.set(ylabel="Relative error (%)", title="Signed counting error by crowd-density level (boxes: IQR, whiskers: 1.5 IQR)")
    ax.legend([h for h, _ in handles], [lbl for _, lbl in handles], loc="upper center", ncols=3, bbox_to_anchor=(0.5, -0.2))
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    return save(fig, "error_by_level.png")


def draw_cm(ax, cm, title, normalize=True, small=False):
    shown = cm / cm.sum(axis=1, keepdims=True).clip(min=1) if normalize else cm
    ax.imshow(shown, cmap=BLUES, vmin=0, vmax=1 if normalize else cm.max())
    ax.grid(False)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = shown[i, j]
            dark = value > (0.55 if normalize else 0.55 * cm.max())
            text = f"{100 * value:.0f}%\n({cm[i, j]})" if normalize else f"{cm[i, j]}"
            ax.text(j, i, text, ha="center", va="center", fontsize=6.5 if small else 8,
                    color="white" if dark else INK)
    ax.set_xticks(range(len(LEVELS)), LEVEL_NAMES, fontsize=7 if small else 8)
    ax.set_yticks(range(len(LEVELS)), LEVEL_NAMES, fontsize=7 if small else 8)
    ax.set(xlabel="Predicted level", ylabel="True level", title=title)
    for spine in ax.spines.values():
        spine.set_visible(False)


def fig_confusion_overall(cm):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
    draw_cm(ax1, cm, "Counts", normalize=False)
    draw_cm(ax2, cm, "Row-normalised (= recall per level)")
    fig.tight_layout()
    return save(fig, "confusion_overall.png")


def fig_confusion_per_dataset(data, cls):
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6))
    for ax, (key, d) in zip(axes, data.items()):
        draw_cm(ax, cls[key]["cm"], f"{d['label']} (acc. {cls[key]['accuracy']:.1f}%)", small=True)
    fig.tight_layout()
    return save(fig, "confusion_per_dataset.png")


def fig_prf(c):
    fig, ax = plt.subplots(figsize=(9, 3.2))
    x = np.arange(len(LEVELS))
    width = 0.26
    for i, (name, values, color) in enumerate(zip(("Precision", "Recall", "F1 score"),
                                                  (c["precision"], c["recall"], c["f1"]), (AXIS, MUTED, SERIES[0]))):
        bars = ax.bar(x + (i - 1) * width, values, width * 0.92, color=color, label=name)
        ax.bar_label(bars, fmt="%.0f", fontsize=7, color=INK_2, padding=2)
    ax.set_xticks(x, [f"{n}\n(n = {s})" for n, s in zip(LEVEL_NAMES, c["support"])])
    ax.set(ylim=(0, 110), ylabel="%", title="Per-level precision, recall and F1 (all test images)")
    ax.legend(loc="upper center", ncols=3, bbox_to_anchor=(0.5, -0.22))
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    return save(fig, "prf_per_level.png")


def fig_alert(data, thresholds):
    fig, ax = plt.subplots(figsize=(9, 3.2))
    for d, color in zip(data.values(), SERIES):
        s = alert_scores(d["true"], d["pred"], thresholds)
        keep = s[:, 1] >= 5  # at least 5 true positives for a meaningful F1
        ax.plot(s[keep, 0], 100 * s[keep, 4], color=color, lw=2, label=d["label"])
    ax.set(xscale="log", ylim=(0, 102), xlabel="Alert threshold T (people per image, log scale)", ylabel="F1 (%)",
           title="Over-crowding alert: F1 of the decision 'count >= T'")
    ax.set_xticks([25, 50, 100, 200, 500, 1000], ["25", "50", "100", "200", "500", "1000"])
    ax.legend(loc="lower left")
    fig.tight_layout()
    return save(fig, "alert_f1.png")


def fig_examples(data):
    rows = []
    for key, d in data.items():
        dataset = SPLITS[key][1]()
        order = np.argsort(d["true"])
        maps = np.load(CACHE_DIR / f"{key}_maps.npz")
        for q in (0.25, 0.9):
            i = int(order[round(q * (len(order) - 1))])
            image_path, density_path = dataset.samples[i]
            with h5py.File(density_path, "r") as f:
                gt = f["density"][:]
            rows.append((d["label"], image_path, gt, maps[image_path.stem].astype(np.float32), d["true"][i], d["pred"][i]))
    fig, axes = plt.subplots(len(rows), 3, figsize=(8.5, 2.35 * len(rows)))
    for row_axes, (label, image_path, gt, pred, true_n, pred_n) in zip(axes, rows):
        with Image.open(image_path) as img:
            row_axes[0].imshow(img.convert("RGB"))
        row_axes[0].set_title(label, fontsize=8)
        row_axes[1].imshow(gt, cmap="inferno")
        row_axes[1].set_title(f"Ground truth: {true_n:.0f}", fontsize=8)
        row_axes[2].imshow(pred, cmap="inferno")
        row_axes[2].set_title(f"Predicted: {pred_n:.1f} ({pred_n - true_n:+.1f})", fontsize=8)
        for ax in row_axes:
            ax.axis("off")
    fig.tight_layout()
    return save(fig, "examples.png")


def fig_latency(data):
    fig, ax = plt.subplots(figsize=(9, 3))
    for d, color in zip(data.values(), SERIES):
        ax.scatter(d["mpix"], d["latency"] / 1000, s=10, alpha=0.5, color=color, edgecolor="none", label=d["label"])
    ax.set(xlabel="Image size (megapixels)", ylabel="Seconds per image",
           title="CPU inference time per image (full resolution, batch 1)")
    ax.legend(loc="upper left", markerscale=2)
    fig.tight_layout()
    return save(fig, "latency.png")


# ---------------------------------------------------------------- PDF

def build_pdf(ctx):
    styles = getSampleStyleSheet()
    body = ParagraphStyle("body", parent=styles["BodyText"], fontName="Helvetica", fontSize=9.5, leading=13.5,
                          spaceAfter=6)
    h1 = ParagraphStyle("h1", parent=styles["Heading1"], fontName="Helvetica-Bold", fontSize=14, spaceBefore=10,
                        spaceAfter=6, textColor=colors.HexColor(INK))
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=11, spaceBefore=8,
                        spaceAfter=4, textColor=colors.HexColor(INK))
    caption = ParagraphStyle("caption", parent=body, fontSize=8.5, leading=11, textColor=colors.HexColor(INK_2),
                             spaceAfter=10)
    title = ParagraphStyle("title", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=19, leading=23,
                           alignment=TA_CENTER, spaceAfter=4)
    subtitle = ParagraphStyle("subtitle", parent=body, alignment=TA_CENTER, textColor=colors.HexColor(INK_2))
    cell = ParagraphStyle("cell", parent=body, fontSize=8.5, leading=10.5, spaceAfter=0)

    page_w = A4[0] - 4 * cm
    story = []
    fig_no = [0]
    tab_no = [0]

    def figure(path, text, width=page_w):
        fig_no[0] += 1
        with Image.open(path) as img:
            w, h = img.size
        width = min(width, 21 * cm * w / h)  # keep tall figures on one page with their caption
        story.append(KeepTogether([
            RLImage(str(path), width=width, height=width * h / w),
            Paragraph(f"<b>Figure {fig_no[0]}.</b> {text}", caption),
        ]))

    def table(text, rows, col_widths=None, bold_last=False, align="CENTER"):
        tab_no[0] += 1
        rows = [[Paragraph(str(c), cell) if isinstance(c, str) and len(c) > 28 else c for c in r] for r in rows]
        t = Table(rows, colWidths=col_widths, repeatRows=1, hAlign="CENTER")
        style = [
            ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 8.5),
            ("FONT", (0, 1), (-1, -1), "Helvetica", 8.5),
            ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor(INK)),
            ("LINEABOVE", (0, 0), (-1, 0), 1, colors.HexColor(INK)),
            ("LINEBELOW", (0, 0), (-1, 0), 0.6, colors.HexColor(INK)),
            ("LINEBELOW", (0, -1), (-1, -1), 1, colors.HexColor(INK)),
            ("ALIGN", (1, 0), (-1, -1), align),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f4f1")]),
            ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]
        if bold_last:
            style += [("FONT", (0, -1), (-1, -1), "Helvetica-Bold", 8.5),
                      ("LINEABOVE", (0, -1), (-1, -1), 0.6, colors.HexColor(AXIS))]
        t.setStyle(TableStyle(style))
        story.append(KeepTogether([Paragraph(f"<b>Table {tab_no[0]}.</b> {text}", caption), t, Spacer(1, 10)]))

    m, c, data, info = ctx["metrics"], ctx["cls"], ctx["data"], ctx["info"]
    keys = list(data)
    pooled = m["pooled"]
    cp = c["pooled"]

    # Title + summary
    story += [
        Paragraph("Crowd Density Estimation with CSRNet", title),
        Paragraph("Experimental results on ShanghaiTech Part A, Part B and the Roboflow crowd dataset", subtitle),
        Paragraph(f"Model: CSRNet (VGG-16 front-end, dilated back-end), checkpoint epoch {info['epoch']} "
                  f"&nbsp;|&nbsp; Report generated {date.today():%d %B %Y}", subtitle),
        Spacer(1, 12),
    ]
    tiles = [
        [f"{m[k]['mae']:.1f}" for k in keys] + [f"{cp['accuracy']:.1f}%", f"{cp['macro_f1']:.1f}%"],
        [f"MAE {data[k]['label']}" for k in keys] + ["Density-level accuracy", "Macro F1"],
    ]
    tile_table = Table(tiles, colWidths=[page_w / 5] * 5)
    tile_table.setStyle(TableStyle([
        ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 17), ("FONT", (0, 1), (-1, 1), "Helvetica", 7.5),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor(INK)), ("TEXTCOLOR", (0, 1), (-1, 1), colors.HexColor(INK_2)),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"), ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor(AXIS)),
        ("LINEAFTER", (0, 0), (-2, -1), 0.6, colors.HexColor(GRID)),
        ("TOPPADDING", (0, 0), (-1, 0), 8), ("BOTTOMPADDING", (0, 1), (-1, 1), 8),
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f7f9fd")),
    ]))
    story += [tile_table, Spacer(1, 12)]

    story.append(Paragraph("1. Summary", h1))
    story.append(Paragraph(
        f"A single CSRNet model was trained jointly on ShanghaiTech Part A, ShanghaiTech Part B, the Roboflow crowd "
        f"dataset and 100 pseudo-labelled phone photographs, and was evaluated on {pooled['n']} annotated test images "
        f"it never trained on. It reaches an MAE of {m[keys[0]]['mae']:.1f} on ShanghaiTech Part A, "
        f"{m[keys[1]]['mae']:.1f} on Part B and {m[keys[2]]['mae']:.1f} on the Roboflow test split, with "
        f"R<super>2</super> = {pooled['r2']:.3f} between predicted and true counts over all test images. "
        f"When counts are grouped into four crowd-density levels (Low, Medium, High, Very high), the predicted level "
        f"matches the true level for {cp['accuracy']:.1f}% of the images (macro F1 {cp['macro_f1']:.1f}%, "
        f"quadratic-weighted Cohen's kappa {cp['kappa']:.3f}) and is within one level for {cp['within_one']:.1f}%.",
        body))

    # Setup
    story.append(Paragraph("2. Experimental setup", h1))
    story.append(Paragraph("2.1 Datasets", h2))
    story.append(Paragraph(
        "ShanghaiTech Part A contains dense crowd images collected from the internet; Part B contains sparser street "
        "scenes from Shanghai at a fixed 1024x768 resolution. Both are annotated with one point per head. The Roboflow "
        "crowd dataset (640x640 images) is annotated with one bounding box per head, whose centre is used as the head "
        "point. Ground-truth density maps use geometry-adaptive Gaussian kernels (Part A, Roboflow) and a fixed kernel "
        "(Part B), and each map sums to the annotated head count. The 100 phone photographs have no annotations; they "
        "were pseudo-labelled by the Part A model and used for training only, never for evaluation.", body))
    table("Datasets and splits. Count statistics are computed on the test split.", [
        ["Dataset", "Train", "Validation", "Test", "Test count range", "Test mean count"],
        ["ShanghaiTech Part A", 300, "-", m[keys[0]]["n"], f"{data[keys[0]]['true'].min():.0f} - {data[keys[0]]['true'].max():.0f}", f"{m[keys[0]]['true_mean']:.1f}"],
        ["ShanghaiTech Part B", 400, "-", m[keys[1]]["n"], f"{data[keys[1]]['true'].min():.0f} - {data[keys[1]]['true'].max():.0f}", f"{m[keys[1]]['true_mean']:.1f}"],
        ["Roboflow crowd", ctx["roboflow_train"], ctx["roboflow_valid"], m[keys[2]]["n"], f"{data[keys[2]]['true'].min():.0f} - {data[keys[2]]['true'].max():.0f}", f"{m[keys[2]]['true_mean']:.1f}"],
        ["Phone photos (pseudo-labelled)", 100, "-", "-", "-", "-"],
    ], col_widths=[4.6 * cm, 1.6 * cm, 2 * cm, 1.5 * cm, 3.2 * cm, 2.8 * cm])

    story.append(Paragraph("2.2 Model and training", h2))
    times = [h["time_s"] for h in ctx["history"]]
    story.append(Paragraph(
        "CSRNet uses the first ten convolutional layers of VGG-16 as a front-end and six dilated 3x3 convolutions "
        "(dilation 2) as a back-end, followed by a 1x1 convolution that outputs a density map at 1/8 of the input "
        "resolution; the predicted count is the sum of the map. The network was first trained on ShanghaiTech Part A "
        "from ImageNet weights, then fine-tuned jointly on all training sets.", body))
    table("Training configuration of the evaluated model.", [
        ["Setting", "Value"],
        ["Initialisation", "ShanghaiTech Part A best checkpoint"],
        ["Optimiser", "Adam, learning rate 1e-5, weight decay 1e-4"],
        ["Loss", "Pixel-wise MSE between density maps (sum reduction)"],
        ["Batch / crop", "8 images, random 256x256 crops, random horizontal flip"],
        ["Precision", "Mixed precision (AMP) on an NVIDIA T4 GPU (Google Colab)"],
        ["Epochs", f"{len(ctx['history'])} completed; best validation MAE at epoch {info['epoch']}"],
        ["Time per epoch", f"{np.mean(times):.0f} s (mean)"],
        ["Model selection", "Lowest MAE on ShanghaiTech A + B test and Roboflow valid"],
    ], col_widths=[4 * cm, 11.5 * cm], align="LEFT")

    story.append(Paragraph("2.3 Evaluation metrics", h2))
    story.append(Paragraph(
        "For N test images with true count <i>y</i> and predicted count <i>p</i>: MAE = mean |<i>p</i> - <i>y</i>| "
        "and RMSE = sqrt(mean (<i>p</i> - <i>y</i>)<super>2</super>) (reported as \"MSE\" in most crowd-counting "
        "papers). Bias is the mean signed error (positive = over-counting). Counting accuracy is mean "
        "max(0, 1 - |<i>p</i> - <i>y</i>| / <i>y</i>). The 95% confidence interval of the MAE comes from a bootstrap "
        f"with {BOOTSTRAP} resamples. For the classification metrics, both <i>y</i> and <i>p</i> are mapped to the "
        "density levels "
        "Low (&lt;50 people), Medium (50-149), High (150-499) and Very high (&ge;500); accuracy, precision, recall, "
        "F1 and the confusion matrix are then computed on these levels. Because the levels are ordered, the "
        "quadratic-weighted Cohen's kappa is also reported.", body))

    # Training curves
    story.append(Paragraph("3. Training behaviour", h1))
    figure(ctx["figs"]["training"], "Training loss and validation error of the joint fine-tuning run. The marked "
           "point is the checkpoint evaluated in this report.")

    # Counting
    story.append(PageBreak())
    story.append(Paragraph("4. Counting results", h1))
    rows = [["Test set", "Images", "MAE", "MAE 95% CI", "RMSE", "Bias", "R²", "r", "Count acc.", "±20%"]]
    for k, label in [(k, data[k]["label"]) for k in keys] + [("pooled", "All test images")]:
        x = m[k]
        rows.append([label, x["n"], f"{x['mae']:.2f}", f"{x['mae_lo']:.1f} - {x['mae_hi']:.1f}", f"{x['rmse']:.2f}",
                     f"{x['bias']:+.1f}", f"{x['r2']:.3f}", f"{x['pearson']:.3f}", f"{x['count_acc']:.1f}%",
                     f"{x['within20']:.1f}%"])
    table("Counting performance of the final model on the held-out test splits. r is the Pearson correlation; "
          "\"±20%\" is the share of images whose predicted count is within 20% of the true count.", rows,
          col_widths=[3.1 * cm, 1.2 * cm, 1.2 * cm, 2 * cm, 1.3 * cm, 1.2 * cm, 1.2 * cm, 1.2 * cm, 1.9 * cm, 1.3 * cm],
          bold_last=True)
    figure(ctx["figs"]["scatter"], "Predicted versus true count for every test image. Points on the dashed line "
           "are perfect predictions; points above it are over-counts.")
    figure(ctx["figs"]["error_by_level"], "Distribution of the signed relative error (predicted - true) / true, "
           "split by the true crowd-density level. Levels with fewer than three images in a dataset are omitted.")
    pt, pp = ctx["pooled_true"], ctx["pooled_pred"]
    lv = level_of(pt)
    median_rel = [100 * np.median((pp[lv == k] - pt[lv == k]) / pt[lv == k]) for k in range(len(LEVELS))]
    story.append(Paragraph(
        "The error depends on crowd density. Over all test images, the median relative error is "
        + ", ".join(f"{v:+.0f}% for {LEVEL_NAMES[k]}" for k, v in enumerate(median_rel))
        + " scenes: the model tends to over-count sparse scenes and under-count the densest ones, the typical "
        "regression-to-the-mean behaviour of density-map counters.", body))

    lit = [["Method", "Source", "Part A MAE", "Part A RMSE", "Part B MAE", "Part B RMSE"]]
    for name, src, a1, a2, b1, b2 in LITERATURE:
        lit.append([name, src, f"{a1:.1f}", f"{a2:.1f}", f"{b1:.1f}", f"{b2:.1f}"])
    lit.append(["This work (joint model)", "this report", f"{m[keys[0]]['mae']:.1f}", f"{m[keys[0]]['rmse']:.1f}",
                f"{m[keys[1]]['mae']:.1f}", f"{m[keys[1]]['rmse']:.1f}"])
    table("Comparison with published results on the ShanghaiTech test sets. Published numbers are taken from the "
          "cited papers, which train a separate model per part; this work uses one model for all datasets.", lit,
          col_widths=[3.6 * cm, 3.8 * cm, 2 * cm, 2.1 * cm, 2 * cm, 2.1 * cm], bold_last=True)
    csr = next(row for row in LITERATURE if row[0] == "CSRNet")
    story.append(Paragraph(
        f"Compared with the published single-dataset CSRNet, the joint model's MAE is "
        f"{m[keys[0]]['mae'] - csr[2]:+.1f} on Part A and {m[keys[1]]['mae'] - csr[4]:+.1f} on Part B. "
        "Sharing one set of weights across datasets with very different densities, image sizes and annotation "
        "styles trades some per-dataset accuracy for a single model that covers sparse street scenes, dense "
        "crowds and the Roboflow images at once.", body))

    # Classification
    story.append(Paragraph("5. Crowd-density level classification", h1))
    story.append(Paragraph(
        "A monitoring system acts on the crowd level rather than on the exact count, so the counts were also "
        "evaluated as a four-class problem. Each test image is assigned a true level from its annotated count and a "
        "predicted level from the model's count; Figure 4 shows the resulting confusion matrix over all "
        f"{pooled['n']} test images.", body))
    figure(ctx["figs"]["confusion_overall"], "Confusion matrix of the crowd-density levels over all test images: "
           "absolute counts (left) and normalised per true level (right). The diagonal holds the correct predictions.")
    off = cp["cm"] * (1 - np.eye(len(LEVELS), dtype=int))
    i, j = np.unravel_index(off.argmax(), off.shape)
    adjacent = 100 * sum(off[a, b] for a in range(len(LEVELS)) for b in range(len(LEVELS)) if abs(a - b) == 1) / off.sum()
    story.append(Paragraph(
        f"{adjacent:.1f}% of the misclassified images land in a neighbouring level. The most frequent confusion is "
        f"true {LEVEL_NAMES[i]} predicted as {LEVEL_NAMES[j]} ({off[i, j]} images, {100 * off[i, j] / cp['cm'][i].sum():.0f}% "
        f"of the {LEVEL_NAMES[i]} images), consistent with the over-counting of sparse scenes seen in Figure 3.", body))
    rows = [["Level", "People", "Images", "Precision", "Recall", "F1 score"]]
    for i, (name, lo, hi) in enumerate(LEVELS):
        rows.append([name, level_range(lo, hi), int(cp["support"][i]),
                     f"{cp['precision'][i]:.1f}%", f"{cp['recall'][i]:.1f}%", f"{cp['f1'][i]:.1f}%"])
    rows.append(["Macro average", "", int(cp["support"].sum()), f"{np.mean(cp['precision']):.1f}%",
                 f"{np.mean(cp['recall']):.1f}%", f"{cp['macro_f1']:.1f}%"])
    table("Per-level precision, recall and F1 over all test images.", rows,
          col_widths=[3.2 * cm, 2.2 * cm, 2 * cm, 2.4 * cm, 2.4 * cm, 2.4 * cm], bold_last=True)
    figure(ctx["figs"]["prf"], "Per-level precision, recall and F1 score (all test images).")

    rows = [["Test set", "Images", "Accuracy", "Within one level", "Macro F1", "Weighted F1", "Weighted kappa"]]
    for k, label in [(k, data[k]["label"]) for k in keys] + [("pooled", "All test images")]:
        x = c[k]
        rows.append([label, int(x["support"].sum()), f"{x['accuracy']:.1f}%", f"{x['within_one']:.1f}%",
                     f"{x['macro_f1']:.1f}%", f"{x['weighted_f1']:.1f}%", f"{x['kappa']:.3f}"])
    table("Density-level classification metrics per test set. Macro F1 averages the levels that occur in that "
          "test set's ground truth or predictions.", rows,
          col_widths=[3.3 * cm, 1.5 * cm, 1.8 * cm, 2.6 * cm, 1.8 * cm, 2.1 * cm, 2.4 * cm], bold_last=True)
    figure(ctx["figs"]["confusion_per_dataset"], "Row-normalised confusion matrices per test set.")

    story.append(Paragraph("6. Over-crowding alert detection", h1))
    story.append(Paragraph(
        "The deployed system raises an alert when the estimated count of a zone reaches a threshold. Treating "
        "\"true count &ge; T\" as the positive class, Figure 7 shows the F1 score of this binary decision for every "
        "threshold T (only thresholds with at least five positive images in a dataset are plotted).", body))
    figure(ctx["figs"]["alert"], "F1 score of the over-crowding alert as a function of the alert threshold T.")
    rows = [["Threshold T", "Positive images", "Precision", "Recall", "F1 score"]]
    for t, pos, p, r, f in alert_scores(ctx["pooled_true"], ctx["pooled_pred"], [50, 100, 150, 250, 500]):
        rows.append([f"{t:.0f}", f"{pos:.0f}", f"{100 * p:.1f}%", f"{100 * r:.1f}%", f"{100 * f:.1f}%"])
    table("Alert detection at selected thresholds over all test images.", rows,
          col_widths=[3 * cm, 3 * cm, 2.6 * cm, 2.6 * cm, 2.6 * cm])

    # Qualitative
    story.append(PageBreak())
    story.append(Paragraph("7. Qualitative results", h1))
    figure(ctx["figs"]["examples"], "Test images (left), ground-truth density maps (middle) and predicted density "
           "maps (right). For each dataset, one image at the 25th and one at the 90th percentile of the true count is "
           "shown.", width=page_w * 0.92)

    # Speed
    story.append(PageBreak())
    story.append(Paragraph("8. Inference speed", h1))
    lat_rows = [["Test set", "Image size", "Median time / image", "Images / second"]]
    for k in keys:
        d = data[k]
        med = np.median(d["latency"])
        lat_rows.append([d["label"], f"{np.median(d['mpix']):.2f} MP (median)", f"{med / 1000:.2f} s", f"{1000 / med:.2f}"])
    table(f"CPU inference time measured during evaluation ({ctx['cpu']}, {info['threads']} threads, PyTorch "
          f"{info['torch_version']}, float32, batch 1).", lat_rows,
          col_widths=[3.6 * cm, 4 * cm, 4 * cm, 3.4 * cm])
    figure(ctx["figs"]["latency"], "CPU inference time per test image against image size.")

    story.append(Paragraph("9. Notes on the evaluation", h1))
    for text in [
        "All numbers in this report are computed by research/make_results_pdf.py from the model's predictions on the "
        "test images; only the published baselines in Table 4 are taken from the literature.",
        "The ShanghaiTech test sets were also used to choose the best training epoch, which is the usual practice in "
        "crowd-counting papers but makes their scores slightly optimistic. Roboflow model selection used the "
        "separate validation split, so the Roboflow test score is fully held out.",
        "The confusion matrix, accuracy and F1 score are not native to crowd counting; they depend on the chosen "
        "level boundaries (50, 150 and 500 people per image), which were fixed from the ground-truth count "
        "distribution before the model's predictions were examined.",
        "The phone photographs are not part of any evaluation because they have no human annotations.",
    ]:
        story.append(Paragraph(f"&bull; {text}", body))

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(colors.HexColor(MUTED))
        canvas.drawString(2 * cm, 1.2 * cm, "Crowd Density Estimation with CSRNet - experimental results")
        canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"Page {doc.page}")
        canvas.restoreState()

    doc = SimpleDocTemplate(str(PDF_PATH), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm,
                            bottomMargin=2 * cm, title="Crowd Density Estimation with CSRNet - Results",
                            author="Crowd Density System")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)


# ---------------------------------------------------------------- main

def main():
    import platform
    import torch

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    info = json.loads((CACHE_DIR / "checkpoint.json").read_text(encoding="utf-8"))
    history = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)["history"]

    data = {}
    for key, (label, _) in SPLITS.items():
        records = json.loads((CACHE_DIR / f"{key}.json").read_text(encoding="utf-8"))
        data[key] = {
            "label": label,
            "true": np.array([r["true_count"] for r in records]),
            "pred": np.array([r["predicted_count"] for r in records]),
            "latency": np.array([r["latency_ms"] for r in records]),
            "mpix": np.array([r["height"] * r["width"] / 1e6 for r in records]),
        }
    pooled_true = np.concatenate([d["true"] for d in data.values()])
    pooled_pred = np.concatenate([d["pred"] for d in data.values()])

    rng = np.random.default_rng(0)
    metrics = {k: counting_metrics(d["true"], d["pred"], rng) for k, d in data.items()}
    metrics["pooled"] = counting_metrics(pooled_true, pooled_pred, rng)
    cls = {k: classification_metrics(d["true"], d["pred"]) for k, d in data.items()}
    cls["pooled"] = classification_metrics(pooled_true, pooled_pred)

    figs = {
        "training": fig_training(history, info["epoch"]),
        "scatter": fig_scatter(data, metrics),
        "error_by_level": fig_error_by_level(data),
        "confusion_overall": fig_confusion_overall(cls["pooled"]["cm"]),
        "confusion_per_dataset": fig_confusion_per_dataset(data, cls),
        "prf": fig_prf(cls["pooled"]),
        "alert": fig_alert(data, np.unique(np.geomspace(20, 1500, 120).round())),
        "examples": fig_examples(data),
        "latency": fig_latency(data),
    }
    roboflow_raw = PROJECT_ROOT / "data" / "raw" / "crowd density tensorflow"
    build_pdf({
        "metrics": metrics, "cls": cls, "data": data, "info": info, "history": history, "figs": figs,
        "pooled_true": pooled_true, "pooled_pred": pooled_pred, "cpu": platform.processor() or "CPU",
        "roboflow_train": len(list((roboflow_raw / "train").glob("*.jpg"))),
        "roboflow_valid": len(list((roboflow_raw / "valid").glob("*.jpg"))),
    })

    summary = {k: {kk: (float(vv) if np.isscalar(vv) else vv.tolist()) for kk, vv in v.items()} for k, v in metrics.items()}
    summary_cls = {k: {kk: (float(vv) if np.isscalar(vv) else vv.tolist()) for kk, vv in v.items()} for k, v in cls.items()}
    (RESULTS_DIR / "result_metrics.json").write_text(
        json.dumps({"checkpoint": info, "levels": LEVELS[:-1] + [("Very high", 500, None)], "counting": summary,
                    "classification": summary_cls}, indent=2), encoding="utf-8")
    print(f"Saved {PDF_PATH}")
    print(f"Saved {RESULTS_DIR / 'result_metrics.json'} and figures in {FIG_DIR}")
    for k in list(data) + ["pooled"]:
        print(f"{k:22s} MAE {metrics[k]['mae']:7.2f}  RMSE {metrics[k]['rmse']:7.2f}  "
              f"level acc {cls[k]['accuracy']:5.1f}%  macro F1 {cls[k]['macro_f1']:5.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
