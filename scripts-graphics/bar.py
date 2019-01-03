# ==============================================================================
# Author:   Robin Chan, University of Wuppertal
# Contact:  rchan@uni-wuppertal.de
# GitHub:   https://github.com/robin-chan
# ==============================================================================

"""
Perform semantic segmentation
Output: prediction images with class colors
"""


import os
import numpy as np
import matplotlib.pyplot as plt
from labels import labels
from globals import *

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
if not os.path.exists("bar"): os.makedirs("bar")
os.environ['PATH'] = os.environ['PATH'] + ':/Library/TeX/texbin'
plt.rc('font', size=10, family='serif')
plt.rc('text', usetex=True)

#####################################################################################
#
# Main
#
#####################################################################################

print("########################################################################")
for k in class_indices:

    # ----------------------------------------------------------------------------------------
    # Plot bar charts
    # ----------------------------------------------------------------------------------------

    print("Plot false & non detection bar chart for class " + labels[k].name)

    # ----------------------------------------------------------------------------------------
    # Load data
    # ----------------------------------------------------------------------------------------

    FILE_GT = 'tables/eval-' + labels[k].name + '-GT.txt'
    FILE_B  = 'tables/comp-' + labels[k].name + '-B.txt'
    FILE_ML = 'tables/comp-' + labels[k].name + '-ML.txt'

    G = np.loadtxt(FILE_GT, skiprows=2, usecols=2)
    M = np.loadtxt(FILE_GT, skiprows=2, usecols=3)
    B = np.loadtxt(FILE_GT, skiprows=2, usecols=4)

    P_ml    = np.loadtxt(FILE_ML, skiprows=2, usecols=2)
    D_ml    = np.loadtxt(FILE_ML, skiprows=2, usecols=7)
    P_bay   = np.loadtxt(FILE_B, skiprows=2, usecols=2)
    D_bay   = np.loadtxt(FILE_B, skiprows=2, usecols=7)

    # ----------------------------------------------------------------------------------------
    # Non-detection
    # ----------------------------------------------------------------------------------------

    nd_bay  = [np.count_nonzero(np.multiply(G[np.logical_and(G >= ub - np.max([ub / 2, 25]), G <= ub)],
                                            B[np.logical_and(G >= ub - np.max([ub / 2, 25]), G <= ub)]) == 0)
               for ub in [25, 50, 100, 200, 400, 800]]
    nd_ml   = [np.count_nonzero(np.multiply(G[np.logical_and(G >= ub - np.max([ub / 2, 25]), G <= ub)],
                                            M[np.logical_and(G >= ub - np.max([ub / 2, 25]), G <= ub)]) == 0)
               for ub in [25, 50, 100, 200, 400, 800]]
    ratio = [float(nd_ml[intv]) / nd_bay[intv] for intv in range(len(nd_ml))]

    # ----------------------------------------------------------------------------------------
    # Create and save figures
    # ----------------------------------------------------------------------------------------

    fig1, (_, _) = plt.subplots(1, 2, figsize=(6, 3))
    plt.subplot(121)
    plt.tick_params(axis='y', which='major', labelsize=14)
    plt.bar(np.arange(6), nd_bay, label="Bayes")
    plt.bar(np.arange(6), nd_ml, label="ML")
    plt.xticks(np.arange(6),("$[0,25]$", "$[25,50]$", "$[50,100]$", "$[100,200]$", "$[200,400]$", "$[400,800]$"),
               rotation=90,fontsize=14)
    plt.xlabel("Comp-size in GT",fontsize=16)
    plt.ylabel("Frequency",fontsize=16)
    plt.legend(prop={'size': 11})
    plt.tight_layout()

    plt.subplot(122)
    plt.tick_params(axis='y', which='major', labelsize=14)
    plt.bar(np.arange(6), np.ones(6), label="Bayes")
    plt.bar(np.arange(6), ratio, label="ML")
    plt.xticks(np.arange(6),("$[0,25]$", "$[25,50]$", "$[50,100]$", "$[100,200]$", "$[200,400]$", "$[400,800]$"),
               rotation=90,fontsize=14)
    plt.xlabel("Comp-size in GT",fontsize=16)
    plt.ylabel("Ratio",fontsize=16)
    plt.legend(prop={'size': 11})
    plt.tight_layout()

    #fig1.suptitle("Non-detection of " + labels[k].name)    ### optional: add title to graphic
    #fig1.subplots_adjust(top=0.85)
    fig1.savefig("bar/ND-" + labels[k].name + ".pdf", transparent=True)

    # ----------------------------------------------------------------------------------------
    # False detection
    # ----------------------------------------------------------------------------------------

    fd_bay  = [np.sum(D_bay[np.logical_and(P_bay >= ub - np.max([ub / 2, 25]), P_bay <= ub)] == 0)
               for ub in [25,50,100, 200, 400, 800]]
    fd_ml   = [np.sum(D_ml [np.logical_and(P_ml  >= ub - np.max([ub / 2, 25]), P_ml  <= ub)] == 0)
               for ub in [25,50,100, 200, 400, 800]]
    ratio   = [float(fd_bay[intv]) / fd_ml[intv] for intv in range(len(fd_ml))]

    # ----------------------------------------------------------------------------------------
    # Create and save figures
    # ----------------------------------------------------------------------------------------

    fig2, (_, _) = plt.subplots(1, 2, figsize=(6, 3))
    plt.subplot(121)
    plt.tick_params(axis='y', which='major', labelsize=14)
    plt.bar(np.arange(6), fd_ml, label = "ML", color = "C1")
    plt.bar(np.arange(6), fd_bay, label = "Bayes", color = "C0")
    plt.xticks(np.arange(6),("$[0,25]$", "$[25,50]$", "$[50,100]$", "$[100,200]$", "$[200,400]$", "$[400,800]$"),
               rotation=90,fontsize=14)
    plt.xlabel("Comp-size in pred",fontsize=16)
    plt.ylabel("Frequency",fontsize=16)
    plt.legend(prop={'size': 11})
    plt.tight_layout()

    plt.subplot(122)
    plt.tick_params(axis='y', which='major', labelsize=14)
    plt.bar(np.arange(6), np.ones(6), label = "ML", color = "C1")
    plt.bar(np.arange(6), ratio, label = "Bayes", color = "C0")
    plt.xticks(np.arange(6),("$[0,25]$", "$[25,50]$", "$[50,100]$", "$[100,200]$", "$[200,400]$", "$[400,800]$"),
               rotation=90,fontsize=14)
    plt.xlabel("Comp-size in pred",fontsize=16)
    plt.ylabel("Ratio",fontsize=16)
    plt.legend(prop={'size': 11})
    plt.tight_layout()

    #fig2.suptitle("False detection of " + labels[k].name)  ### optional: add title to graphic
    #fig2.subplots_adjust(top=0.85)
    fig2.savefig("bar/FD-" + labels[k].name + ".pdf", transparent=True)

##############################################################################################

print("DONE!")
