import matplotlib
import mplhep as hep
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.ticker import AutoMinorLocator

larger = 28
large = 26
med = 20

matplotlib.use("Agg")
plt.style.use(hep.style.ROOT)
_params = {
    "axes.titlesize": larger,
    "legend.fontsize": med,
    "figure.figsize": (16, 10),
    "axes.labelsize": larger,
    "xtick.labelsize": large,
    "ytick.labelsize": large,
    "figure.titlesize": large,
    "xtick.bottom": True,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 12,
    "ytick.major.size": 12,
    "xtick.minor.size": 8,
    "ytick.minor.size": 8,
    "ytick.left": True,
}
plt.rcParams.update(_params)


def plot_effs(xaxis, yaxis, error=None, path=None):
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))

    if error is None:
        ax.plot(xaxis, yaxis, color="black")
    else:
        error = np.swapaxes(error, 0, 1)
        ax.errorbar(xaxis, yaxis, yerr=error, fmt="o", capsize=2.0)

    # ax.set_title("Expected b-Tagging Performance Run 3".format(label), fontsize=24)
    ax.set_xlabel("offline b-jet value")
    ax.yaxis.set_ticks_position("both")
    ax.xaxis.set_ticks_position("both")
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.set_ylabel("$N_{passing}/N_{total}$")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, "both", linestyle="dashed", alpha=1.0)
    hep.cms.label(loc=0, lumi=None, year=None, rlabel="")
    fig.savefig(path)


def plot_multiple_effs(xaxis, yaxes, errors=None, names=None, path=None):
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))

    if not isinstance(names, list):
        names = [f"Path {i}" for i in range(len(yaxes))]

    if errors is None:
        for name, yaxis in zip(names, yaxes):
            ax.plot(xaxis, yaxis, label=name)
    else:
        for name, yaxis, error in zip(names, yaxes, errors):
            error = np.swapaxes(error, 0, 1)
            ax.errorbar(xaxis, yaxis, yerr=error, fmt="o", label=name, capsize=2.0)

    # ax.set_title("Expected b-Tagging Performance Run 3".format(label), fontsize=24)
    ax.set_xlabel("offline b-jet value")
    ax.yaxis.set_ticks_position("both")
    ax.xaxis.set_ticks_position("both")
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.set_ylabel("$N_{passing}/N_{total}$")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, "both", linestyle="dashed", alpha=1.0)
    ax.legend()
    hep.cms.text("Data Preliminary", loc=0)
    fig.savefig(path)
