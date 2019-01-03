# ==============================================================================
# Author:   Robin Chan, University of Wuppertal
# Contact:  rchan@uni-wuppertal.de
# GitHub:   https://github.com/robin-chan
# ==============================================================================

"""
Create visualization of priors (of training set) as heatmaps
Output: pdf with heatmap
"""


import os
import numpy as np
from labels import *
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import rc
from globals import *

#####################################################################################
#
# Setup
#
#####################################################################################

os.chdir(work_dir + "out/graphics")
if not os.path.exists("priors-heat"):os.makedirs("priors-heat")
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

priors = np.load(work_dir + "out/priors-array/priors.npy")
smooth_priors = np.load(work_dir + "out/priors-array/priors_smooth.npy")

#####################################################################################
#
# Main
#
#####################################################################################

print("########################################################################")
print("Create priors heatmaps:")

for k in class_indices:
    heat = priors[:, :, k] / 19800
    smooth = smooth_priors[:, :, k] / 19800
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.23, 2.1))
    divider1 = make_axes_locatable(ax1)
    divider2 = make_axes_locatable(ax2)
    cax1 = divider1.append_axes("right", size="3%", pad=0.1)
    cax2 = divider2.append_axes("right", size="3%", pad=0.1)
    ax1.grid(False)
    ax2.grid(False)
    ax1.text(0.5, 1.05, labels[k].name, ha="center", transform=ax1.transAxes)
    ax2.text(0.5, 1.05, "Smoothed", ha="center", transform=ax2.transAxes)
    ax1.tick_params(axis='both', which='both', length=0)
    ax2.tick_params(axis='both', which='both', length=0)
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax1.get_yticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)
    plt.setp(ax2.get_yticklabels(), visible=False)
    heatplot1 = ax1.imshow(heat, cmap='RdBu_r', interpolation='None')
    heatplot2 = ax2.imshow(smooth, cmap='RdBu_r', interpolation='None')
    plt.tight_layout()
    fig.colorbar(heatplot1, cax=cax1, format="%.3f")
    fig.colorbar(heatplot2, cax=cax2, format="%.3f")
    fig.savefig("priors-heat/" +str(k) + "_" + str(labels[k].name) + ".pdf", bbox_inches='tight')

print("DONE")