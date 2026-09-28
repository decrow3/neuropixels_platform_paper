"""Preserve the print-size figure and export a proportional 20 cm submission copy.

Run after build_figure4_integrated.py. PyMuPDF is preferred; Ghostscript and
Poppler provide a dependency-free system fallback.
"""
from pathlib import Path
import re
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
stem = ROOT / 'artifacts/figure4_rerun/v8_final/Figure4_CDE_rerun_with_schematic'
width = 20 / 2.54 * 72
target = stem.with_name(stem.name + '_elife_20cm')
try:
    import pymupdf as fitz
except ModuleNotFoundError:
    fitz = None

if fitz is not None:
    source = fitz.open(stem.with_suffix('.pdf'))
    assert len(source) == 1
    height = width * source[0].rect.height / source[0].rect.width
    out = fitz.open()
    page = out.new_page(width=width, height=height)
    page.show_pdf_page(page.rect, source, 0)
    out.save(target.with_suffix('.pdf'), garbage=4, deflate=True)
    page.get_pixmap(dpi=300, colorspace=fitz.csRGB).save(target.with_suffix('.png'))
else:
    if not shutil.which('gs') or not shutil.which('pdfinfo') or not shutil.which('pdftocairo'):
        raise RuntimeError('Export requires PyMuPDF or gs + pdfinfo + pdftocairo')
    info = subprocess.run(
        ['pdfinfo', str(stem.with_suffix('.pdf'))],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    match = re.search(r'^Page size:\s+([0-9.]+) x ([0-9.]+) pts$', info, re.MULTILINE)
    if not match:
        raise RuntimeError('Could not determine source PDF page size')
    source_width, source_height = map(float, match.groups())
    height = width * source_height / source_width
    subprocess.run(
        [
            'gs', '-q', '-dBATCH', '-dNOPAUSE', '-sDEVICE=pdfwrite',
            '-dFIXEDMEDIA', '-dPDFFitPage', f'-dDEVICEWIDTHPOINTS={width:.6f}',
            f'-dDEVICEHEIGHTPOINTS={height:.6f}',
            f'-sOutputFile={target.with_suffix(".pdf")}', str(stem.with_suffix('.pdf')),
        ],
        check=True,
    )
    subprocess.run(
        [
            'pdftocairo', '-png', '-singlefile', '-r', '300',
            str(target.with_suffix('.pdf')), str(target),
        ],
        check=True,
    )
print(target.with_suffix('.pdf'))
