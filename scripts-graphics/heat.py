# ==============================================================================
# Author:   Robin Chan, University of Wuppertal
# Contact:  rchan@uni-wuppertal.de
# GitHub:   https://github.com/robin-chan
# ==============================================================================

"""
Create heatmaps of non-detection at object-level and at pixel-level
Output: prediction images with class colors
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.measure import label
from mpl_toolkits.axes_grid1 import make_axes_locatable
from labels import labels
from globals import *

#####################################################################################
#
# function: create and save heatmap
#
#####################################################################################

def save_heat(array1,array2,title,filename,vmin,vmax):
    fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(6, 2.15))
    divider1 = make_axes_locatable(ax1)
    divider2 = make_axes_locatable(ax2)
    cax1 = divider1.append_axes("right", size="3%", pad=0.1)
    cax2 = divider2.append_axes("right", size="3%", pad=0.1)
    ax1.grid(False)
    ax2.grid(False)
    ax1.text(0.5, 1.08, "Bayes",  ha="center",transform=ax1.transAxes,fontsize=17)
    ax2.text(0.5, 1.08, "ML", ha="center",transform=ax2.transAxes,fontsize=17)
    ax1.tick_params(axis='both', which='both', length=0)
    ax2.tick_params(axis='both', which='both', length=0)
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax1.get_yticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)
    plt.setp(ax2.get_yticklabels(), visible=False)
    heatplot1 = ax1.imshow(array1, cmap='RdBu_r',interpolation='None', vmin=vmin, vmax=vmax)
    heatplot2 = ax2.imshow(array2, cmap='RdBu_r',interpolation='None', vmin=vmin, vmax=vmax)
    plt.tight_layout()
    cbar1=fig.colorbar(heatplot1, cax=cax1, ticks=range(0,int(vmax)+1,2), format="%.0f")
    cbar2=fig.colorbar(heatplot2, cax=cax2, ticks=range(0,int(vmax)+1,2), format="%.0f")
    cbar1.ax.tick_params(labelsize=14)
    cbar2.ax.tick_params(labelsize=14)
    #fig.suptitle("Non-detection " + title) ### optional: add title to graphic
    #fig.subplots_adjust(top=0.80)
    fig.savefig("heat/" + filename + ".pdf", bbox_inches='tight',transparent=True)

#####################################################################################
#
# Setup
#
#####################################################################################

# ------------------------------------------------------------------------------------
# Choose data folder
# ------------------------------------------------------------------------------------

folder = work_dir + "out/graphics/"

# ------------------------------------------------------------------------------------
# Initialization
# ------------------------------------------------------------------------------------

os.chdir(work_dir + "out/graphics/")
os.environ['PATH'] = os.environ['PATH'] + ':/Library/TeX/texbin'
plt.rc('font', size=10, family='serif')
plt.rc('text', usetex=True)
if not os.path.exists("heat"): os.makedirs("heat")
gt_list     = sorted(os.listdir(ground_truth_dir))
ml_list     = sorted(os.listdir(work_dir + "out/predictions/ML/"))
bayes_list  = sorted(os.listdir(work_dir + "out/predictions/B/"))
n = len(gt_list)
shape = resolution[0]*resolution[1] ### height times width, number of pixel-positions

#####################################################################################
#
# Main
#
#####################################################################################

print("########################################################################")
for k in class_indices:

    # ----------------------------------------------------------------------------------------
    # ML heatmaps
    # ----------------------------------------------------------------------------------------

    print("Creating ML heatmaps of class " + labels[k].name)
    heatmap_px_ml   = np.zeros(shape)
    heatmap_obj_ml  = np.zeros(shape)

    for im in range(n):

        # load components
        ml_comp = np.load("merged-comp-arrays/ML-" + labels[k].name + "/" + ml_list[im] + ".npy").flatten()
        gt_array = np.asarray(Image.open(ground_truth_dir + gt_list[im]).resize((resolution[0], resolution[1])))
        gt_comp = label(((gt_array == labels[k].color).all(axis=2)).astype(int)).flatten()

        # count non-detected pixels (false negatives)
        count = np.multiply((ml_comp != 0).astype(int), (gt_comp != 0).astype(int))
        count = count + (gt_comp != 0).astype(int)
        count[count != 1] = 0
        heatmap_px_ml += count

        # count non-detected objects (false negatives object-wise, no overlap with prediction)
        for inst in range(1, np.max(gt_comp)+1):
            check = np.multiply(ml_comp, (gt_comp == inst).astype(int))
            if np.count_nonzero(check) == 0:
                heatmap_obj_ml += (gt_comp == inst).astype(int)

    heatmap_px_ml.resize((resolution[1],resolution[0]))
    heatmap_obj_ml.resize((resolution[1],resolution[0]))

    # ----------------------------------------------------------------------------------------
    # Bayes heatmaps
    # ----------------------------------------------------------------------------------------

    print("Creating Bayes heatmaps of class " + labels[k].name)
    heatmap_px_bay  = np.zeros(shape)
    heatmap_obj_bay = np.zeros(shape)

    for im in range(n):

        # load components
        bayes_comp = np.load("merged-comp-arrays/B-" + labels[k].name + "/" + bayes_list[im] + ".npy").flatten()
        gt_array = np.asarray(Image.open(ground_truth_dir + gt_list[im]).resize((resolution[0], resolution[1])))
        gt_comp = label(((gt_array == labels[k].color).all(axis=2)).astype(int)).flatten()

        # count non-detected pixels (false negatives)
        count = np.multiply((bayes_comp != 0).astype(int), (gt_comp != 0).astype(int))
        count = count + (gt_comp != 0).astype(int)
        count[count != 1] = 0
        heatmap_px_bay += count

        # count non-detected objects
        for inst in range(1, np.max(gt_comp)+1):
            check = np.multiply(bayes_comp, (gt_comp == inst).astype(int))
            if np.count_nonzero(check) == 0:
                heatmap_obj_bay += (gt_comp == inst).astype(int)

    heatmap_px_bay.resize((resolution[1],resolution[0]))
    heatmap_obj_bay.resize((resolution[1],resolution[0]))

    # ----------------------------------------------------------------------------------------
    # Save heatmaps
    # ----------------------------------------------------------------------------------------
    
    vmin = np.min(heatmap_px_bay); vmax = np.max(heatmap_px_bay)
    save_heat(heatmap_px_bay,heatmap_px_ml,"pixel-wise of " + labels[k].name,labels[k].name + "-px",vmin,vmax)
    save_heat(heatmap_obj_bay, heatmap_obj_ml,"object-wise of " + labels[k].name, labels[k].name + "-obj",vmin,vmax)

##############################################################################################

print("DONE!")


