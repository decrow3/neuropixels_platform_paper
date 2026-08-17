#!/usr/bin/env python3
"""Render the plain-language multi-site V1–Allen V1 difference report as PDF."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import sys

import markdown
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
REPORT_DIR = ROOT / "reports" / "multisite_v1_allen_v1_differences"
SOURCE = REPORT_DIR / "MULTISITE_V1_ALLEN_V1_DIFFERENCES.md"
SUMMARY_FIGURE = REPORT_DIR / "summary_gap_figure.png"
DETAILED_GRATING_FIGURE = REPORT_DIR / "detailed_grating_diagnostic.png"
BROADER_V1_FIGURE = REPORT_DIR / "broader_v1_comparison.png"
HTML_OUTPUT = REPORT_DIR / "MULTISITE_V1_ALLEN_V1_DIFFERENCES.html"
PDF_OUTPUT = REPORT_DIR / "MULTISITE_V1_ALLEN_V1_DIFFERENCES.pdf"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--html-output", type=Path, default=HTML_OUTPUT)
    parser.add_argument("--pdf-output", type=Path, default=PDF_OUTPUT)
    parser.add_argument("--skip-pdf", action="store_true")
    return parser.parse_args()


def add_value_labels(ax: plt.Axes, bars, *, decimals: int = 3) -> None:
    for bar in bars:
        value = float(bar.get_height())
        offset = 0.008 if value >= 0 else -0.008
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + offset,
            f"{value:+.{decimals}f}" if decimals == 3 else f"{value:.{decimals}f}",
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=9.5,
            fontweight="bold",
        )


def build_summary_figure(path: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.8))

    grating_labels = [
        "Multi-site\nV1 raw",
        "Multi-site V1\nphase matched",
        "Allen BO\nrep. session",
        "Allen FC\nrep. session",
    ]
    grating_values = np.array([-0.098, 0.019, 0.088, 0.123])
    grating_colors = ["#D95F02", "#1B9E77", "#6F63A6", "#B07AA1"]
    bars = axes[0].bar(
        np.arange(4), grating_values, color=grating_colors, width=0.7
    )
    axes[0].axhline(0, color="#333333", linewidth=0.8)
    axes[0].set_xticks(np.arange(4), grating_labels)
    axes[0].set_ylim(-0.14, 0.16)
    axes[0].set_ylabel("log modulation score")
    axes[0].set_title("Grating response after common-window matching")
    axes[0].grid(axis="y", alpha=0.18)
    add_value_labels(axes[0], bars)
    axes[0].annotate(
        "known starting phase\nrecovers +0.117",
        xy=(1, 0.019),
        xytext=(0.35, 0.14),
        arrowprops={"arrowstyle": "->", "color": "#333333", "lw": 1.0},
        ha="center",
        va="top",
        fontsize=9,
    )

    decay_labels = [
        "Multi-site V1\n300 flashes",
        "Multi-site V1\n150 flashes",
        "Allen BO\n150 flashes",
        "Allen FC\n150 flashes",
    ]
    decay_values = np.array([47.53, 45.92, 43.88, 43.21])
    decay_colors = ["#D95F02", "#1B9E77", "#6F63A6", "#B07AA1"]
    bars = axes[1].bar(np.arange(4), decay_values, color=decay_colors, width=0.7)
    axes[1].set_xticks(np.arange(4), decay_labels)
    axes[1].set_ylim(40, 50.5)
    axes[1].set_ylabel("mean response-decay time (ms)")
    axes[1].set_title("Flash response after matching trial count")
    axes[1].grid(axis="y", alpha=0.18)
    for bar in bars:
        value = float(bar.get_height())
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.18,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="bold",
        )
    axes[1].annotate(
        "matching flash count\nshortens multi-site V1 by 1.61 ms",
        xy=(1, 45.92),
        xytext=(0.35, 50.1),
        arrowprops={"arrowstyle": "->", "color": "#333333", "lw": 1.0},
        ha="center",
        va="top",
        fontsize=9,
    )

    fig.suptitle(
        "Known protocol differences explain part—but not all—of the absolute gap",
        fontsize=14,
        fontweight="bold",
        y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94), w_pad=2.2)
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def build_supporting_figures() -> None:
    """Re-render report copies with descriptive cohort labels."""
    from scripts.mousev2_grating_corrected_welch_bridge import make_figure
    from scripts.v1_dataset_bridge import _make_figure

    corrected_dir = ROOT / "data" / "imports" / "mousev2_grating_corrected_welch_bridge_v1"
    make_figure(
        pd.read_csv(corrected_dir / "session_summary.csv"),
        pd.read_csv(corrected_dir / "tf_center_summary.csv"),
        pd.read_csv(corrected_dir / "analysis_centers.csv"),
        DETAILED_GRATING_FIGURE,
    )

    bridge_dir = ROOT / "artifacts" / "figure3" / "06b_v1_dataset_bridge"
    _make_figure(
        pd.read_csv(bridge_dir / "session_metric_summary.csv"),
        pd.read_csv(bridge_dir / "timescale_coverage.csv"),
        pd.read_csv(bridge_dir / "tf_session_summary.csv"),
        BROADER_V1_FIGURE,
    )


def html_document(body: str) -> str:
    css = r"""
@page {
  size: Letter;
  margin: 0.62in 0.66in 0.66in 0.66in;
  @bottom-center {
    content: "Multi-site V1–Allen V1 differences  •  " counter(page) " / " counter(pages);
    font-family: "Liberation Sans", Arial, sans-serif;
    font-size: 8pt;
    color: #6b7280;
  }
}
@page:first {
  @bottom-center { content: none; }
}
* { box-sizing: border-box; }
html { background: #ffffff; }
body {
  margin: 0;
  color: #172033;
  background: #ffffff;
  font-family: "Liberation Sans", Arial, sans-serif;
  font-size: 10.3pt;
  line-height: 1.43;
}
h1, h2, h3 {
  color: #123b5d;
  break-after: avoid-page;
  page-break-after: avoid;
}
h1 {
  margin: 0 0 0.18in 0;
  padding-bottom: 0.08in;
  border-bottom: 2px solid #1f7898;
  font-size: 21pt;
  line-height: 1.12;
}
h2 {
  margin: 0.18in 0 0.08in;
  font-size: 14pt;
  line-height: 1.18;
}
h3 { font-size: 11.5pt; }
p { margin: 0.06in 0 0.12in; }
ul, ol { margin: 0.05in 0 0.13in 0.25in; padding-left: 0.14in; }
li { margin-bottom: 0.055in; }
strong { color: #0f2f47; }
blockquote {
  margin: 0.16in 0;
  padding: 0.13in 0.17in;
  border-left: 5px solid #1f7898;
  background: #edf6f8;
  color: #17384b;
  break-inside: avoid-page;
  page-break-inside: avoid;
}
blockquote p { margin: 0.02in 0; }
table {
  width: 100%;
  border-collapse: collapse;
  margin: 0.10in 0 0.18in;
  font-size: 8.5pt;
  line-height: 1.28;
  break-inside: avoid-page;
  page-break-inside: avoid;
}
thead { display: table-header-group; }
th {
  color: #ffffff;
  background: #245e7a;
  font-weight: bold;
  text-align: left;
  padding: 0.065in 0.07in;
  border: 1px solid #d4dde3;
}
td {
  vertical-align: top;
  padding: 0.06in 0.07in;
  border: 1px solid #d4dde3;
}
tr:nth-child(even) td { background: #f4f7f9; }
img {
  display: block;
  max-width: 100%;
  max-height: 7.25in;
  width: auto;
  height: auto;
  margin: 0.10in auto 0.10in;
  object-fit: contain;
  break-inside: avoid-page;
  page-break-inside: avoid;
}
em { color: #4b5563; }
th em { color: #ffffff; }
code {
  font-family: "Liberation Mono", monospace;
  font-size: 8.8pt;
  color: #24364b;
  background: #f2f4f7;
  padding: 0.01in 0.025in;
  border-radius: 2px;
  overflow-wrap: anywhere;
}
.page-break {
  break-before: page;
  page-break-before: always;
  height: 0;
}
.cover {
  min-height: 9.0in;
  padding: 0.72in 0.55in 0.35in;
  display: flex;
  flex-direction: column;
  justify-content: center;
  text-align: left;
  background: linear-gradient(150deg, #f5fafb 0%, #ffffff 54%, #edf3f7 100%);
  border-top: 12px solid #1f7898;
  border-bottom: 3px solid #1f7898;
}
.cover h1 {
  border: 0;
  color: #103d5a;
  font-size: 33pt;
  line-height: 1.05;
  margin-bottom: 0.22in;
}
.cover h2 {
  color: #2c657d;
  font-size: 18pt;
  font-weight: normal;
  margin: 0 0 0.42in;
}
.cover > p { max-width: 6.1in; font-size: 11pt; }
.cover blockquote { margin-top: 0.42in; font-size: 12pt; }
a { color: #155f82; text-decoration: none; }
"""
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>How the multi-site V1 recordings and Allen V1 differ</title>
  <style>{css}</style>
</head>
<body>
{body}
</body>
</html>
"""


def render_html(source: Path, output: Path) -> None:
    body = markdown.markdown(
        source.read_text(encoding="utf-8"),
        extensions=["tables", "md_in_html", "sane_lists", "smarty"],
        output_format="html5",
    )
    output.write_text(html_document(body), encoding="utf-8")


def render_pdf(html_path: Path, pdf_path: Path) -> None:
    chrome = shutil.which("google-chrome") or shutil.which("chromium")
    if chrome is None:
        raise RuntimeError("Google Chrome or Chromium is required to render the PDF")
    command = [
        chrome,
        "--headless=new",
        "--disable-gpu",
        "--disable-dev-shm-usage",
        "--no-sandbox",
        "--allow-file-access-from-files",
        "--user-data-dir=/tmp/multisite-v1-allen-v1-report-chrome",
        "--no-pdf-header-footer",
        f"--print-to-pdf={pdf_path}",
        html_path.resolve().as_uri(),
    ]
    subprocess.run(command, cwd=html_path.parent, check=True)


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    html_output = args.html_output.resolve()
    pdf_output = args.pdf_output.resolve()
    html_output.parent.mkdir(parents=True, exist_ok=True)
    pdf_output.parent.mkdir(parents=True, exist_ok=True)
    build_summary_figure(SUMMARY_FIGURE)
    build_supporting_figures()
    render_html(source, html_output)
    if not args.skip_pdf:
        render_pdf(html_output, pdf_output)
        print(f"Rendered report: {pdf_output}")
    else:
        print(f"Rendered HTML: {html_output}")


if __name__ == "__main__":
    main()
