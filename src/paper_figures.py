"""Publication figure export for the paper (vector PDF, high raster DPI)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

FIG_EXT = ".pdf"
SAVE_DPI = 600

_CONFIGURED = False


def configure_matplotlib_for_publication() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    plt.rcParams.update(
        {
            "savefig.format": "pdf",
            "savefig.dpi": SAVE_DPI,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    _CONFIGURED = True


def paper_figure_path(root: Path, *parts: str) -> Path:
    return root.joinpath(*parts).with_suffix(FIG_EXT)


def save_publication_figure(fig, path: Path | str, **kwargs) -> Path:
    configure_matplotlib_for_publication()
    out = Path(path).with_suffix(FIG_EXT)
    out.parent.mkdir(parents=True, exist_ok=True)
    opts = {
        "format": "pdf",
        "bbox_inches": "tight",
        "facecolor": "white",
        "edgecolor": "none",
        "dpi": SAVE_DPI,
    }
    opts.update(kwargs)
    fig.savefig(out, **opts)
    return out
