"""Shared chart styling, following the dataviz skill's palette (references/palette.md).

Static matplotlib PNGs for a research reproduction don't need the skill's
full interactive/dark-mode HTML machinery, but reuse its validated color
roles: CARLA is highlighted with the categorical "orange" slot against a
neutral sequential-blue ramp for the other strategies/ranks, and chart
chrome (ink/gridline/surface) follows the palette's light-mode values.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

# --- palette.md roles ---
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"

CARLA_COLOR = "#eb6834"  # categorical slot 2 (orange) -- reserved for CARLA only
DEFECTOR_COLOR = "#e34948"  # categorical slot 8 (red) -- reserved for uncooperative strategies

# Sequential blue ramp (light -> dark), palette.md "Sequential hue"
SEQUENTIAL_BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def blues(n: int) -> list[str]:
    """n evenly spaced steps from the sequential blue ramp."""
    idx = np.linspace(0, len(SEQUENTIAL_BLUES) - 1, n)
    return [SEQUENTIAL_BLUES[int(round(i))] for i in idx]


def style_axes(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.figure.set_facecolor(SURFACE)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(BASELINE)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9)
    ax.xaxis.label.set_color(INK_PRIMARY)
    ax.yaxis.label.set_color(INK_PRIMARY)
    ax.title.set_color(INK_PRIMARY)
    ax.grid(axis="y", color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def series_color(name: str) -> str:
    """CARLA gets the reserved highlight color; Defector/ZD-Extortion get
    the "uncooperative" red (the paper calls these out specifically as the
    strategies CARLA does worst against); everything else is neutral."""
    if name == "CARLA":
        return CARLA_COLOR
    if name in ("Defector", "ZD-Extortion"):
        return DEFECTOR_COLOR
    return "#6da7ec"


def save_fig(fig, path) -> None:
    fig.savefig(path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
