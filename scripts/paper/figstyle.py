"""Shared style and helpers for the paper's data figures. Imported by figures.py.

Every figure is drawn at its final print size, so the font sizes are the printed sizes. The widths are those of the
two-column template: a single column is 3.03 in and the full text width is 6.30 in. The body font is the first serif
family of FONT_PREFERENCE that is installed, converted once to TrueType outlines if it is an OpenType/CFF font, so the
PDF carries valid embedded fonts; without any of them, matplotlib's default serif is used.
"""

import os
from pathlib import Path
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results"
OUT = Path(os.environ.get("FIGURE_DIR", REPO / "figures"))

SINGLE = 3.03
FULL = 6.30
LABEL, TICK, LEGEND, TITLE, ANNOT, MIN_PT = 10, 8.5, 8.5, 10, 7.5, 7.0

# Colors, with the role each one plays in every chart.
ATTACK = "#DD8452"         # the universal suffix / the attack: the key result
SEARCH = "#4C72B0"         # per-document search
CONTROL = "#8C8C8C"        # controls and baselines
CONTROL_LIGHT = "#BBBBBB"  # background points
CLEAN = "#55A868"          # correct / clean
FLIPPED = "#C44E52"        # flipped
OTHER = "#8172B3"          # a second series that is neither search nor control
INK = "#222222"            # text
MUTED = "#5A5A5A"          # secondary text
RULE = "#9A9A9A"           # thin rules

# key, display name, short name, mean document length in o200k_base tokens (paper order).
DATASETS = [
    ("trec", "TREC-QC", "TREC", 15), ("ag_news", "AG News", "AG", 50), ("dbpedia_14", "DBpedia-14", "DBP", 66),
    ("yahoo_answers", "Yahoo! Answers", "Yahoo", 86), ("patent", "Patent abstracts", "Patent", 109),
    ("arxiv_abstracts", "arXiv abstracts", "arXivA", 261), ("20newsgroups", "20 Newsgroups", "20NG", 283),
    ("lexglue_ecthr", "LexGLUE ECtHR", "ECtHR", 2163), ("arxiv_fulltext", "arXiv full text", "arXivF", 2831),
]

FONT_PREFERENCE = ["Times New Roman", "Times", "Nimbus Roman", "TeX Gyre Termes", "DejaVu Serif"]


def _fontconfig_families():
    try:
        out = subprocess.run(["fc-list", ":", "family"], capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return set()
    return {name.strip() for line in out.splitlines() for name in line.split(",")}


def _otf_to_ttf(src, dst):
    """Convert an OpenType/CFF font to TrueType (quadratic) outlines, keeping its names and metrics."""
    from fontTools.ttLib import TTFont, newTable
    from fontTools.pens.cu2quPen import Cu2QuPen
    from fontTools.pens.ttGlyphPen import TTGlyphPen
    font = TTFont(src)
    order = font.getGlyphOrder()
    glyph_set = font.getGlyphSet()
    glyphs = {}
    for name in order:
        pen = TTGlyphPen(glyph_set)
        glyph_set[name].draw(Cu2QuPen(pen, 1.0, reverse_direction=True))
        glyphs[name] = pen.glyph()
    font["loca"] = newTable("loca")
    font["glyf"] = glyf = newTable("glyf")
    glyf.glyphOrder, glyf.glyphs = order, glyphs
    del font["CFF "]
    if "VORG" in font:
        del font["VORG"]
    glyf.compile(font)
    for name, glyph in glyphs.items():
        if hasattr(glyph, "xMin"):
            font["hmtx"][name] = (font["hmtx"][name][0], glyph.xMin)
    maxp = newTable("maxp")
    maxp.tableVersion = 0x00010000
    for field in ("maxZones", "maxTwilightPoints", "maxStorage", "maxFunctionDefs", "maxInstructionDefs",
                  "maxStackElements", "maxSizeOfInstructions", "maxComponentElements", "maxPoints", "maxContours",
                  "maxCompositePoints", "maxCompositeContours", "maxComponentDepth"):
        setattr(maxp, field, 0)
    maxp.maxZones, maxp.numGlyphs = 1, len(order)
    font["maxp"] = maxp
    font["head"].glyphDataFormat = 0
    post = font["post"]
    post.formatType, post.extraNames, post.mapping = 2.0, [], {}
    font.sfntVersion = "\x00\x01\x00\x00"
    font.save(dst)


def _setup_font():
    installed = _fontconfig_families()
    family = next((f for f in FONT_PREFERENCE if f in installed), "DejaVu Serif")
    entries = [f for f in font_manager.fontManager.ttflist if f.name == family]
    cff = [f for f in entries if font_manager.is_opentype_cff_font(f.fname)]
    if cff:
        cache = Path(matplotlib.get_cachedir()) / "flipjev_ttf"
        cache.mkdir(parents=True, exist_ok=True)
        converted = []
        for entry in cff:
            dst = cache / (Path(entry.fname).stem + ".ttf")
            if not dst.exists():
                try:
                    _otf_to_ttf(entry.fname, dst)
                except Exception:  # keep the CFF font rather than fail if the conversion is unavailable
                    continue
            converted.append(dst)
        if converted:
            font_manager.fontManager.ttflist = [f for f in font_manager.fontManager.ttflist if f not in cff]
            for dst in converted:
                font_manager.fontManager.addfont(str(dst))
    elif not entries and family != "DejaVu Serif":
        family = "DejaVu Serif"
    return family


FONT = _setup_font()

STYLE = {
    "font.family": "serif", "font.serif": [FONT], "mathtext.fontset": "stix",
    "font.size": TICK, "axes.labelsize": LABEL, "axes.titlesize": TITLE, "axes.titleweight": "bold",
    "figure.labelsize": LABEL, "figure.titlesize": TITLE,
    "xtick.labelsize": TICK, "ytick.labelsize": TICK, "legend.fontsize": LEGEND, "legend.title_fontsize": LEGEND,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#808080", "grid.alpha": 0.25, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False, "legend.handlelength": 1.6, "legend.handletextpad": 0.5, "legend.borderaxespad": 0.3,
    "legend.columnspacing": 1.2, "legend.labelspacing": 0.35, "legend.borderpad": 0.2,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "savefig.dpi": 300, "figure.dpi": 150,
    "lines.linewidth": 1.8, "lines.markersize": 5,
    "axes.linewidth": 0.7, "axes.edgecolor": "#333333", "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": "#333333", "ytick.color": "#333333", "xtick.labelcolor": INK, "ytick.labelcolor": INK,
    "xtick.major.width": 0.7, "ytick.major.width": 0.7, "xtick.major.size": 3, "ytick.major.size": 3,
    "xtick.major.pad": 2.5, "ytick.major.pad": 2.5, "axes.labelpad": 3, "axes.titlepad": 4,
    "errorbar.capsize": 2,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white", "savefig.edgecolor": "white",
}


def apply():
    """Reset matplotlib to its defaults and apply the paper style."""
    plt.rcdefaults()
    plt.rcParams.update(STYLE)


apply()


def thousands(x, _pos=None):
    return f"{x:,.0f}"


def save(fig, name):
    """Write figures/<name>.pdf and figures/<name>.png (300 dpi)."""
    OUT.mkdir(parents=True, exist_ok=True)
    pdf, png = OUT / f"{name}.pdf", OUT / f"{name}.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=300)
    print(f"wrote {pdf} and {png.name}")
    plt.close(fig)
