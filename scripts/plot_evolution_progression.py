"""Plot generation progression of ShinkaEvolve candidate scoring functions."""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Data ─────────────────────────────────────────────────────────────────────

RANDOM_BASELINE = 0.439  # Expected NDCG@5 for random ranking

candidates = [
    # Gen 1: 8 original delta features
    ("v0 Baseline\n(weighted Δs)",          "gen1", 0.5129),
    ("v1 L2 Norm",                          "gen1", 0.5136),
    ("v2 LogP Dominant",                    "gen1", 0.5019),
    ("v3 Interactions\n(LogP×TPSA)",        "gen1", 0.5039),
    ("v4 Signed Deltas",                    "gen1", 0.5078),
    ("v5 Threshold\n(sweet spot)",          "gen1", 0.5021),
    ("v6 Dissimilarity\nFocus",             "gen1", 0.5067),
    ("v7 Rank-Based",                       "gen1", 0.5057),
    ("v8 Max Feature",                      "gen1", 0.5096),
    # Gen 2: + 4 env context features (12 total)
    ("v0 12-feat\nBaseline",                "gen2", 0.5129),
    ("v1 Ridge\nOptimal",                   "gen2", 0.5100),
    ("v2 Env Only",                         "gen2", 0.4131),
    ("v3 Env + Deltas",                     "gen2", 0.5086),
    ("v4 Multiplicative\n(env × Δ)",        "gen2", 0.5087),
    ("v5 Mean Delta\nDirect",               "gen2", 0.4876),
    ("v6 Env-Gated\nThreshold",             "gen2", 0.5022),
    ("v7 Kitchen Sink",                     "gen2", 0.5062),
]

labels = [c[0] for c in candidates]
gens = [c[1] for c in candidates]
scores = np.array([c[2] for c in candidates])

# ── Colors ───────────────────────────────────────────────────────────────────

gen_colors = {"gen1": "#3B82F6", "gen2": "#F59E0B"}
bar_colors = [gen_colors[g] for g in gens]

# Highlight best overall
best_idx = int(np.argmax(scores))

# ── Plot ─────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(16, 7))

x = np.arange(len(candidates))
bars = ax.bar(x, scores, color=bar_colors, edgecolor="white", linewidth=0.8,
              width=0.75, zorder=3)

# Highlight the best bar
bars[best_idx].set_edgecolor("#16A34A")
bars[best_idx].set_linewidth(2.5)

# Reference lines
ax.axhline(RANDOM_BASELINE, color="#94A3B8", linestyle="--", linewidth=1.2,
           label=f"Random baseline ({RANDOM_BASELINE})", zorder=2)
ax.axhline(0.52, color="#EF4444", linestyle=":", linewidth=1.2,
           label="2D feature ceiling (~0.52)", zorder=2)

# Score labels on bars
for i, (bar, score) in enumerate(zip(bars, scores)):
    weight = "bold" if i == best_idx else "normal"
    color = "#16A34A" if i == best_idx else "#1E293B"
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
            f"{score:.4f}", ha="center", va="bottom", fontsize=7.5,
            fontweight=weight, color=color, rotation=0)

# Generation separator
sep_x = 8.5  # between gen1 (0-8) and gen2 (9-16)
ax.axvline(sep_x, color="#CBD5E1", linestyle="-", linewidth=1.5, zorder=1)
ax.text(4, 0.405, "Generation 1\n8 Δ-descriptor features",
        ha="center", va="bottom", fontsize=10, color="#3B82F6",
        fontstyle="italic", alpha=0.8)
ax.text(12.5, 0.405, "Generation 2\n+ 4 env context features",
        ha="center", va="bottom", fontsize=10, color="#F59E0B",
        fontstyle="italic", alpha=0.8)

# Axes
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7.5)
ax.set_ylabel("NDCG@5 (leave-one-target-out)", fontsize=11)
ax.set_title("ShinkaEvolve: Generation Progression of Scoring Functions",
             fontsize=14, fontweight="bold", pad=15)
ax.set_ylim(0.40, 0.535)
ax.set_xlim(-0.6, len(candidates) - 0.4)

# Grid
ax.yaxis.grid(True, alpha=0.3, zorder=0)
ax.set_axisbelow(True)

# Legend
gen1_patch = mpatches.Patch(color="#3B82F6", label="Gen 1: property deltas only")
gen2_patch = mpatches.Patch(color="#F59E0B", label="Gen 2: + env context features")
best_patch = mpatches.Patch(facecolor="white", edgecolor="#16A34A", linewidth=2,
                            label=f"Best: {labels[best_idx].replace(chr(10), ' ')} ({scores[best_idx]:.4f})")
handles = [gen1_patch, gen2_patch, best_patch]
# Add reference lines to legend
handles.extend(ax.get_legend_handles_labels()[0])
ax.legend(handles=handles, loc="upper right", fontsize=8.5,
          framealpha=0.9, edgecolor="#E2E8F0")

# Annotation: key insight
ax.annotate(
    "Env context features have low\nwithin-group variance → no\nNDCG improvement",
    xy=(11, 0.4131), xytext=(14.5, 0.44),
    fontsize=8, color="#B45309",
    arrowprops=dict(arrowstyle="->", color="#B45309", lw=1.2),
    ha="center", va="bottom",
    bbox=dict(boxstyle="round,pad=0.4", facecolor="#FEF3C7", edgecolor="#F59E0B",
              alpha=0.9),
)

plt.tight_layout()
plt.savefig("outputs/evolve/generation_progression.png", dpi=180,
            bbox_inches="tight", facecolor="white")
plt.close()
print("Saved: outputs/evolve/generation_progression.png")
