import numpy as np
import matplotlib as plt
from matplotlib.patches import Polygon


def trim_axes(ax):
    # remove the upper and the right borders of the axis
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_position(('outward', 5))


def prep4ai():
    plt.rcParams['pdf.fonttype'] = 42  # AI can detect text now
    plt.rcParams['font.sans-serif'] = "Arial"


def add_snr_colorbar(fig, ax, im):
    cbar_ax = ax.inset_axes([1, .05, .04, .85])
    fig.colorbar(im, cax=cbar_ax)
    cbar_ax.set(title='SNR')


def add_shades(ax, n_blocks, n_trials):
    n_trials_per_block = int(n_trials / n_blocks)
    # +++ TEST +++
    assert n_trials_per_block * n_blocks == n_trials
    # ++++++++++++
    block_onsets = np.array(range(1, n_trials + 1, n_trials_per_block))
    shade_onsets = block_onsets.reshape(-1, 2)[:, 1]
    ymin, ymax = ax.get_ylim()
    for ionset in shade_onsets:
        vertices = np.array([(ionset - .5, ymin),
                             (ionset + .5 + n_trials_per_block - 1, ymin),
                             (ionset + .5 + n_trials_per_block - 1, ymax),
                             (ionset - .5, ymax)])
        poly = Polygon(vertices, facecolor='k', alpha=.1)
        ax.add_patch(poly)
