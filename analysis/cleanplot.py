
def trim_axes(ax, xlim, ylim):
    # remove the upper and the right borders of the axis
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # limit the axes
    ax.set_ylim(ymin=ylim[0], ymax=ylim[1])
    ax.set_xlim(xmin=xlim[0], xmax=xlim[1])