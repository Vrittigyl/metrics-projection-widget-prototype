import numpy as np
import pkgutil
import importlib
import inspect
import ipywidgets as widgets
import matplotlib.pyplot as plt
from IPython.display import display


# Optional dependencies
try:
    import bct
    BCT_AVAILABLE = True
except ImportError:
    BCT_AVAILABLE = False

try:
    import tvb.analyzers as analyzers
    from tvb.datatypes.time_series import TimeSeries
    TVB_AVAILABLE = True
except ImportError:
    TVB_AVAILABLE = False



# Discover TVB metrics dynamically
def discover_tvb_metrics():

    metrics = {}

    if not TVB_AVAILABLE:
        return metrics

    for _, module_name, _ in pkgutil.iter_modules(analyzers.__path__):

        if not module_name.startswith("metric_"):
            continue

        module = importlib.import_module(f"tvb.analyzers.{module_name}")

        for name, obj in inspect.getmembers(module):

            if inspect.isfunction(obj) and name.startswith("compute_"):
                label = name.replace("compute_", "").replace("_metric", "")
                label = label.replace("_", " ").title()

                metrics[f"TVB: {label}"] = obj

    return metrics



# Discover BCT metrics dynamically
def discover_bct_metrics():

    metrics = {}

    if not BCT_AVAILABLE:
        return metrics

    test = np.random.rand(10,10)
    test = (test + test.T)/2

    for name, fn in inspect.getmembers(bct, inspect.isfunction):
        try:
            result = fn(test)

            if isinstance(result, np.ndarray) and result.shape == (10,):
                label = name.replace("_"," ").title()
                metrics[f"BCT: {label}"] = fn

        except Exception:
            continue

    return metrics



# Unified metric registry
METRICS = {}
METRICS.update(discover_tvb_metrics())
METRICS.update(discover_bct_metrics())



# Helper functions
def make_fake_timeseries():

    np.random.seed(42)
    return np.random.randn(600, 10)


def make_region_positions(n):

    angles = np.linspace(0, 2*np.pi, n, endpoint=False)

    return np.column_stack([
        np.cos(angles),
        np.sin(angles)
    ])


def wrap_timeseries(data):

    if isinstance(data, TimeSeries):
        return data

    # duplicate signal to create two state variables
    data2 = np.stack([data, data], axis=1)

    ts = TimeSeries(
        data=data2[:, :, :, None],
        sample_period=1.0
    )

    ts.configure()

    return ts



# Compute metric
def call_tvb_metric(fn, ts):
    """
    Try to call a TVB metric without hardcoding parameter names.
    Add required parameters only when the function raises KeyError.
    """

    params = {"time_series": ts}

    # possible optional parameters with defaults
    defaults = {
        "start_point": 500,
        "segment": 4,
    }

    for key, value in defaults.items():
        try:
            return fn(params)
        except KeyError as e:
            missing = str(e).strip("'")
            if missing in defaults:
                params[missing] = defaults[missing]
            else:
                raise

    return fn(params)

def compute_metric(data, metric_name):

    fn = METRICS[metric_name]

    if metric_name.startswith("TVB"):

        ts = wrap_timeseries(data)

        result = call_tvb_metric(fn, ts)

        # handle dict outputs
        if isinstance(result, dict):
            result = list(result.values())[0]

        values = np.array(result)

        if values.size == 1:
            values = np.repeat(values, data.shape[1])

        return values

    else:

        fc = np.abs(np.corrcoef(data.T))
        return fn(fc)

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

    ax2.hist(values, bins=6)

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
            options=["viridis","plasma"],
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
