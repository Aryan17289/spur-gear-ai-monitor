"""Styling utilities for matplotlib charts"""
import matplotlib.pyplot as plt

def style_ax(ax, fig):
    """Apply consistent dark theme styling to matplotlib axes"""
    ax.set_facecolor("#0D1018")
    fig.patch.set_facecolor("#111620")
    ax.tick_params(colors="#A8B8CC", labelsize=10)
    ax.grid(axis="x", color="#182030", linewidth=0.7, zorder=1)
    for sp in ax.spines.values():
        sp.set_color("#243048")
        sp.set_linewidth(0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

def bar_label(ax, bars, values, fmt="{:+.4f}"):
    """Add value labels to horizontal bar chart"""
    x_max = max(abs(v) for v in values) if values else 1
    offset = x_max * 0.018 or 0.0005
    for bar, val in zip(bars, values):
        ha = "left" if val >= 0 else "right"
        xp = val + (offset if val >= 0 else -offset)
        ax.text(xp, bar.get_y() + bar.get_height()/2,
                fmt.format(val), va="center", ha=ha,
                fontsize=9, color="#5A6A80", fontfamily="monospace")
