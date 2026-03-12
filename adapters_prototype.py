import numpy as np
import importlib
import inspect
import ipywidgets as widgets
import matplotlib.pyplot as plt
from IPython.display import display

from tvb.datatypes.time_series import TimeSeries
import tvb.adapters.analyzers as adapters


# Discover adapter metrics dynamically
def discover_adapter_metrics():
    metrics = {}

    for adapter_name in adapters.ALL_ANALYZERS:

        try:
            module = importlib.import_module(
                f"tvb.adapters.analyzers.{adapter_name.lower()}"
            )

            for name, obj in inspect.getmembers(module, inspect.isclass):

                if name.endswith("Adapter"):

                    label = name.replace("Adapter", "")
                    metrics[label] = obj

        except Exception:
            continue

    return metrics


METRICS = discover_adapter_metrics()


# Helpers
def make_fake_timeseries():

    np.random.seed(42)
    return np.random.randn(1000, 76)


def make_region_positions(n):

    angles = np.linspace(0, 2*np.pi, n, endpoint=False)

    return np.column_stack([
        np.cos(angles),
        np.sin(angles)
    ])


def wrap_timeseries(data):

    if isinstance(data, TimeSeries):
        return data

    data2 = np.stack([data, data], axis=1)

    ts = TimeSeries(
        data=data2[:, :, :, None],
        sample_period=1.0
    )

    ts.configure()

    return ts



# Run metric through adapter
def compute_metric(data, metric_name):

    AdapterClass = METRICS[metric_name]

    ts = wrap_timeseries(data)
    adapter = AdapterClass()

    params = {"time_series": ts}

    try:
        result = adapter.launch(params)
    except Exception:
        return np.zeros(data.shape[1])

    values = np.array(result)

    if values.size == 1:
        values = np.repeat(values, data.shape[1])

    return values



# Plot
def draw_plot(data, metric_name, cmap):

    values = compute_metric(data, metric_name)

    n = len(values)

    coords = make_region_positions(n)

    fig, (ax1, ax2) = plt.subplots(1,2,figsize=(10,4))

    scatter = ax1.scatter(
        coords[:,0],
        coords[:,1],
        c=values,
        cmap=cmap,
        s=200
    )

    ax1.set_title(metric_name)
    ax1.set_xticks([])
    ax1.set_yticks([])

    fig.colorbar(scatter, ax=ax1)

    ax2.hist(values, bins=10)
    ax2.set_title("Distribution")

    plt.tight_layout()
    plt.show()



# Widget
class MetricsProjectionWidget:

    def __init__(self, data=None):

        if data is None:
            data = make_fake_timeseries()

        self.data = data

        self.metric_dd = widgets.Dropdown(
            options=list(METRICS.keys()),
            description="Metric:"
        )

        self.cmap_dd = widgets.Dropdown(
            options=["viridis", "plasma"],
            value="viridis",
            description="Colormap:"
        )

        self.output = widgets.Output()

        self.metric_dd.observe(self.update, names="value")
        self.cmap_dd.observe(self.update, names="value")

    def update(self, change=None):

        with self.output:
            self.output.clear_output(wait=True)

            draw_plot(
                self.data,
                self.metric_dd.value,
                self.cmap_dd.value
            )

    def show(self):

        display(
            widgets.VBox([
                widgets.HBox([self.metric_dd, self.cmap_dd]),
                self.output
            ])
        )

        self.update()
