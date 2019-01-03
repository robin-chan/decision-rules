# ==============================================================================
# Author:   Robin Chan, University of Wuppertal
# Contact:  rchan@uni-wuppertal.de
# GitHub:   https://github.com/robin-chan
# ==============================================================================

"""
Remove small connected components
Group remaining connected components with small distance in-between
Output: modified arrays of connected components
"""

import os
import math
import numpy as np
from PIL import Image
from labels import labels
from globals import *
from skimage.measure import label, regionprops
from skimage.segmentation import find_boundaries
from skimage.morphology import convex_hull_image

#####################################################################################
#
# function: merge connected components
# INPUT:
# im_path: path to image
# color: color of class whose connected components are merged
#
#####################################################################################

def merge_cc(im_path,color,min_size=10,max_dist=10):

    # ------------------------------------------------------------------------------------
    # Step 1: extract every single pixel
    # ------------------------------------------------------------------------------------

    im = np.asarray(Image.open(im_path))
    mask = ((im==color).all(axis=2)).astype(int)

    # ------------------------------------------------------------------------------------
    # Step 2: remove every component with less than min_size pixels
    # ------------------------------------------------------------------------------------

    labelled = label(mask)
    for i in range(np.max(labelled)+1):
        if np.count_nonzero(labelled == i) <= min_size:
            labelled[labelled == i] = 0

    labelled[labelled != 0] = 1
    labelled = label(labelled)
    regions = regionprops(labelled)

    # ------------------------------------------------------------------------------------
    # Step 3: group components with Euclidean distance less than max_dist
    # ------------------------------------------------------------------------------------

    for blob in regions:

        label_boundaries = find_boundaries(labelled == blob.label)
        bound = np.column_stack(np.where(label_boundaries))

        for j_blob in regions:
            if j_blob.label > blob.label:
                regrouped = False
                j_label_boundaries = find_boundaries(labelled == j_blob.label)
                j_bound = np.column_stack(np.where(j_label_boundaries))

                i = 0
                while not regrouped and i < len(bound):
                    j = 0
                    while not regrouped and j < len(j_bound):
                        if math.hypot(j_bound[j][1] - bound[i][1], j_bound[j][0] - bound[i][0]) < max_dist:
                            labelled[labelled == j_blob.label] = min(blob.label, j_blob.label)
                            j_blob.label = min(blob.label, j_blob.label)
                            regrouped = True
                        j += 1
                    i += 1

    comp = np.unique(labelled)

    # ------------------------------------------------------------------------------------
    # Step 4: create convex hulls (optional)
    # ------------------------------------------------------------------------------------

    '''
    for i in range(1,len(comp)):
        label_boundaries = find_boundaries(labelled == comp[i])
        if np.count_nonzero((label_boundaries).astype(int)) != 0:
            chull = convex_hull_image(label_boundaries)
            labelled[chull] = comp[i]
        else: labelled[labelled == comp[i]] = 0
    '''

    # ------------------------------------------------------------------------------------
    # Step 5: return array
    # ------------------------------------------------------------------------------------

    for i in range(len(comp)):
        labelled[labelled == comp[i]] = i

    return labelled


#####################################################################################
#
# Setup
#
#####################################################################################

# ------------------------------------------------------------------------------------
# Choose data folder
# ------------------------------------------------------------------------------------

folder = work_dir + "out/predictions/"

# ------------------------------------------------------------------------------------
# Initialization
# ------------------------------------------------------------------------------------

os.chdir(work_dir + "out/graphics/")
if not os.path.exists("1_merged-comp-arrays"):
    for k in class_indices:
        os.makedirs("1_merged-comp-arrays/B-" + labels[k].name)
        os.makedirs("1_merged-comp-arrays/ML-" + labels[k].name)

ml_list = sorted(os.listdir(folder+"/ML"))
bayes_list = sorted(os.listdir(folder+"/B"))
n = len(ml_list)


#####################################################################################
#
# Main
#
#####################################################################################

print("########################################################################")
print("Start merging connected components for class:")
for k in class_indices:
    print(str(k) + ":" + labels[k].name)
    for im in range(n):
        if im % 10 == 0: print("%d/%d Images processed" % (im, n))
        for k in class_indices:
            # save array with merged connected components in folder "merged/"
            np.save("1_merged-comp-arrays/ML-" + labels[k].name + "/" + ml_list[im],
                    merge_cc(folder + "/ML/" + ml_list[im], labels[k].color))
            np.save("1_merged-comp-arrays/B-" + labels[k].name + "/" + bayes_list[im],
                    merge_cc(folder + "/B/" + bayes_list[im], labels[k].color))
    print("%d/%d Images processed" % (n,n))

print("DONE")