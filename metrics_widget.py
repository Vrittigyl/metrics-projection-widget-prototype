import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display


def make_fake_timeseries():
    """Creates fake brain signal data."""
    np.random.seed(42)
    return np.random.randn(500, 10)   # 500 timepoints, 10 regions


def make_fake_region_positions():
    """Creates (x, y) positions for 10 brain regions on a 2D plot."""
    angles = np.linspace(0, 2 * np.pi, 10, endpoint=False)
    x = np.cos(angles)
    y = np.sin(angles)
    return x, y


REGION_NAMES = [
    "Frontal-L", "Frontal-R", "Parietal-L", "Parietal-R",
    "Temporal-L", "Temporal-R", "Occipital-L", "Occipital-R",
    "Cingulate-L", "Cingulate-R"
]


def compute_metric(timeseries, metric_name):
    """
    Takes the timeseries (time x regions) and computes
    ONE number per region — that number is what gets colored.
    metric_name options:
      "Mean"     → average signal per region
      "Variance" → how much the signal fluctuates per region
    """
    if metric_name == "Mean":
        return np.mean(timeseries, axis=0)      # shape: (10,)

    if metric_name == "Variance":
        return np.var(timeseries, axis=0)       # shape: (10,)


def draw_plot(metric_name, colormap):
    """
    Draws the actual visualization:
    - Left plot: brain regions as colored dots
    - Right plot: histogram of metric values across regions
    """
    ts       = make_fake_timeseries()
    x, y     = make_fake_region_positions()
    values   = compute_metric(ts, metric_name)   # one number per region

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # ── Left: brain map ──
    scatter = ax1.scatter(x, y, c=values, cmap=colormap, s=300, zorder=3)

    # label each dot with the region name
    for i, name in enumerate(REGION_NAMES):
        ax1.annotate(name, (x[i], y[i]), fontsize=7,
                     ha="center", va="bottom", xytext=(0, 12),
                     textcoords="offset points")

    fig.colorbar(scatter, ax=ax1, label=metric_name)
    ax1.set_title(f"Brain Regions colored by {metric_name}")
    ax1.set_xticks([])
    ax1.set_yticks([])

    # ── Right: histogram ──
    ax2.hist(values, bins=5, color="steelblue", edgecolor="white")
    ax2.set_xlabel(metric_name)
    ax2.set_ylabel("Number of regions")
    ax2.set_title(f"Distribution of {metric_name}")

    plt.tight_layout()
    plt.show()

def show_widget():
    metric_dropdown = widgets.Dropdown(
        options=["Mean", "Variance"],
        value="Mean",
        description="Metric:",
    )
    colormap_dropdown = widgets.Dropdown(
        options=["viridis"],
        value="viridis",
        description="Colormap:",
    )

    output = widgets.Output()   # a box that holds the plot

    def on_change(change):
        """Runs every time a dropdown value changes — clears old plot, draws new one."""
        with output:
            output.clear_output(wait=True)
            draw_plot(metric_dropdown.value, colormap_dropdown.value)

    # attach on_change to both dropdowns
    metric_dropdown.observe(on_change, names="value")
    colormap_dropdown.observe(on_change, names="value")

    # show everything
    display(widgets.HBox([metric_dropdown, colormap_dropdown]))
    display(output)

    # draw the first plot immediately
    with output:
        draw_plot(metric_dropdown.value, colormap_dropdown.value)
