# ==============================================================================
# Author:   Robin Chan, University of Wuppertal
# Contact:  rchan@uni-wuppertal.de
# GitHub:   https://github.com/robin-chan
# ==============================================================================

"""
Analyze Bayes and ML prediction segments
Output: txt files with relevant metrics
"""


import os
import numpy as np
from PIL import Image
from skimage.measure import label
from tabulate import tabulate
from itertools import zip_longest
from labels import labels
from globals import *

#####################################################################################
#
# function: save values to txt file
#
#####################################################################################

def save_to_file(id,c1,c2,c3,c4,c5,c6,head,filename="text"):
    table = [[i for i in element if i is not None] for element in
             list(zip_longest(*[id, c1, c2, c3,c4,c5,c6]))]
    f = open("tables/" + filename + ".txt", 'w')
    f.write(tabulate(table, head))
    f.close()

#####################################################################################
#
# function: save more values to txt file
#
#####################################################################################

def save_to_file2(id,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13,c14,c15,head,filename="text"):
    table = [[i for i in element if i is not None] for element in
             list(zip_longest(*[id,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13,c14,c15]))]
    f = open("tables/" + filename + ".txt", 'w')
    f.write(tabulate(table, head))
    f.close()

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
if not os.path.exists("tables"): os.makedirs("tables")
gt_list     = sorted(os.listdir(ground_truth_dir))
ml_list     = sorted(os.listdir(work_dir + "out/predictions/ML/"))
bayes_list  = sorted(os.listdir(work_dir + "out/predictions/B/"))
n = len(gt_list)

#####################################################################################
#
# Main
#
#####################################################################################

print("########################################################################")
for k in class_indices:
        
    # ----------------------------------------------------------------------------------------
    # Start analyzing components from prediction images
    # ----------------------------------------------------------------------------------------

    print("Looking for " + str(labels[k].name) +" compenents in ML & Bayes prediction")
    t1_id = []; t1_c1 = []; t1_c2 = []; t1_c3 = []; t1_c4 = []; t1_c5 = []; t1_c6 = []  # data for table-1
    t2_id = []; t2_c1 = []; t2_c2 = []; t2_c3 = []; t2_c4 = []; t2_c5 = []; t2_c6 = []  # data for table-2

    for im in range(n):
        if im % 10 == 0: print("%d/%d Images processed" % (im, n))

        # load (merged) components
        ml_comp     = np.load("merged-comp-arrays/ML-" + labels[k].name + "/" + ml_list[im] + ".npy").flatten()
        bayes_comp  = np.load("merged-comp-arrays/B-" + labels[k].name + "/" + bayes_list[im] + ".npy").flatten()
        gt_array    = np.asarray(Image.open(ground_truth_dir + gt_list[im]).resize((640, 480)))
        gt_comp     = label(((gt_array == labels[k].color).all(axis=2)).astype(int)).flatten()  # not merged

        # ------------------------------------------------------------------------------------
        # Maximum Likelihood
        # ------------------------------------------------------------------------------------

        for inst in range(1,np.max(ml_comp)+1): # check every instance in ML-prediction

            M = np.count_nonzero(ml_comp == inst)
            B = np.count_nonzero(bayes_comp[ml_comp == inst])
            G = np.count_nonzero(gt_comp[ml_comp == inst])
            G_inst = np.unique(gt_comp[ml_comp == inst])
            D = len(G_inst[G_inst != 0])

            t1_id.append((im,inst))     # image and instance index (useful for visualization)
            t1_c1.append(M)             # pixel size of ML-component
            t1_c2.append(float(B)/M)    # percentage of Bayes pixels in ML-component
            t1_c3.append(float(G)/M)    # percentage of GT pixels in ML-component (precision)
            t1_c4.append(G)             # number of pixels in ML-component that are correct (true positives)
            t1_c5.append(M-G)           # number of pixels in ML-component that are incorrect (false positives)
            t1_c6.append(D)             # number of instances that are detected by ML-component

        # ------------------------------------------------------------------------------------
        # Bayes
        # ------------------------------------------------------------------------------------

        for inst in range(1,np.max(bayes_comp)+1):  # check every instance in Bayes-prediction

            B = np.count_nonzero(bayes_comp == inst)
            M = np.count_nonzero(ml_comp[bayes_comp == inst])
            G = np.count_nonzero(gt_comp[bayes_comp == inst])
            G_inst = np.unique(gt_comp[bayes_comp == inst])
            D = len(G_inst[G_inst != 0])

            t2_id.append((im,inst))     # image and instance index (useful for visualization)
            t2_c1.append(B)             # pixel size of Bayes-component
            t2_c2.append(float(M)/B)    # percentage of ML pixels in Bayes-component
            t2_c3.append(float(G)/B)    # percentage of GT pixels in Bayes-component (precision)
            t2_c4.append(G)             # number of pixels in Bayes-component that are correct (true positives)
            t2_c5.append(B-G)           # number of pixels in Bayes-component that are incorrect (false positives)
            t2_c6.append(D)             # number of instances that are detected by Bayes-component

    print("%d/%d Images processed" % (n, n))

    # ----------------------------------------------------------------------------------------
    # Save data
    # ----------------------------------------------------------------------------------------

    save_to_file(t1_id,t1_c1,t1_c2,t1_c3,t1_c4,t1_c5,t1_c6, # data
                 head = ["img,inst","M","B/M", "G/M","TP","FP","D"],
                 filename = "comp-" + labels[k].name + "-ML")
    save_to_file(t2_id,t2_c1,t2_c2,t2_c3,t2_c4,t2_c5,t2_c6, # data
                 head = ["img,inst","B","M/B", "G/B","TP","FP","D"],
                 filename = "comp-" + labels[k].name + "-B")


    # ----------------------------------------------------------------------------------------
    # Start analyzing components in Ground Truth
    # ----------------------------------------------------------------------------------------

    print("Computing eval-measures for " + str(labels[k].name) + " compenents in GT")
    y_id=[];y1=[];y2=[];y3=[];y4=[];y5=[];y6=[];y7=[];y8=[];y9=[];y10=[];y11=[];y12=[];y13=[];y14=[];y15=[]

    for im in range(n):
        if im % 10 == 0: print("%d/%d Images processed" % (im, n))

        # load (merged) components
        ml_comp     = np.load("merged-comp-arrays/ML-" + labels[k].name + "/" + ml_list[im] + ".npy").flatten()
        bayes_comp  = np.load("merged-comp-arrays/B-" + labels[k].name + "/" + bayes_list[im] + ".npy").flatten()
        gt_array    = np.asarray(Image.open(ground_truth_dir + gt_list[im]).resize((640, 480)))
        gt_comp     = label(((gt_array == labels[k].color).all(axis=2)).astype(int)).flatten()  # not merged

        for inst in range(1, np.max(gt_comp) + 1):

            M_inst = np.trim_zeros(np.unique(ml_comp[gt_comp == inst]))
            B_inst = np.trim_zeros(np.unique(bayes_comp[gt_comp == inst]))
            M = 0; B = 0
            for j_inst in range(0,len(M_inst)): M += np.count_nonzero(ml_comp == M_inst[j_inst])
            for j_inst in range(0,len(B_inst)): B += np.count_nonzero(bayes_comp == B_inst[j_inst])
            G = np.count_nonzero(gt_comp == inst)

            # compute true-positives-, false-positives-, false-negatives-pixel for GT-components
            TP_ml   = np.count_nonzero(ml_comp[gt_comp == inst])
            TP_bay  = np.count_nonzero(bayes_comp[gt_comp == inst])
            FP_ml   = M - TP_ml
            FP_bay  = B - TP_bay
            FN_ml   = G - TP_ml
            FN_bay  = G - TP_bay

            # precision
            prc_ml  = float(TP_ml) / (TP_ml + FP_ml)    if M != 0 else 'nan'
            prc_bay = float(TP_bay) / (TP_bay + FP_bay) if B != 0 else 'nan'

            # recall
            rec_ml  = float(TP_ml) / (TP_ml + FN_ml)
            rec_bay = float(TP_bay) / (TP_bay + FN_bay)

            # intersection over union
            IoU_ml  = float(TP_ml) / (TP_ml + FP_ml + FN_ml)
            IoU_bay = float(TP_bay) / (TP_bay + FP_bay + FN_bay)

            y_id.append((im, inst)) # image and instance index (useful for visualization)
            y1.append(G)            # pixel size of GT-component
            y2.append(M)            # pixel size of ML-components which have an intersect with GT-component
            y3.append(B)            # pixel size of B-components which have an intersect with GT-component
            y4.append(TP_ml)        # number of pixels in ML-component that are correct
            y5.append(TP_bay)       # number of pixels in Bayes-component that are correct
            y6.append(FP_ml)        # number of pixels in ML-component that are incorrect
            y7.append(FP_bay)       # number of pixels in Bayes-component that are incorrect
            y8.append(FN_ml)        # number of pixels in GT-component which are not found by ML
            y9.append(FN_bay)       # number of pixels in GT-component which are not found by Bayes
            y10.append(prc_ml)      # percentage of ML-pixels in GT-component relative to ML-component size
            y11.append(prc_bay)     # percentage of B-pixels in GT-component relative to B-component size
            y12.append(rec_ml)      # percentage of ML-pixels in GT-component relative to GT-component size
            y13.append(rec_bay)     # percentage of B-pixels in GT-component relative to GT-component size
            y14.append(IoU_ml)      # percentage of joint ML-GT-pixels relative to union of ML-GT-pixels
            y15.append(IoU_bay)     # percentage of joint B-GT-pixels relative to union of B-GT-pixels

    print("%d/%d Images processed" % (n, n))

    # ----------------------------------------------------------------------------------------
    # Save data
    # ----------------------------------------------------------------------------------------

    save_to_file2(y_id, y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12, y13, y14, y15,
                  head = ["img,inst", "G", "M", "B", "TP_ml", "TP_bayes", "FP_ml", "FP_bayes", "FN_ml",
                          "FN_bayes", "PRC_ml", "PRC_bayes", "REC_ml", "REC_bayes", "IoU_ml", "IoU_bayes"],
                  filename = "eval-" + labels[k].name + "-GT")

##############################################################################################

print("DONE!")