import matplotlib as plt


def trim_axes(ax):
    # remove the upper and the right borders of the axis
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def prep4ai():
    plt.rcParams['pdf.fonttype'] = 42  # AI can detect text now
    plt.rcParams['font.sans-serif'] = "Arial"
