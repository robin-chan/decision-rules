# ==============================================================================
# Author:   Robin Chan, University of Wuppertal
# Contact:  rchan@uni-wuppertal.de
# GitHub:   https://github.com/robin-chan
# ==============================================================================

"""
Compare precision and recall performances between Bayes and ML
Output: cdf plots
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
# initialization
# ------------------------------------------------------------------------------------

os.chdir(work_dir + "out/graphics/")
if not os.path.exists("cdf"): os.makedirs("cdf")
os.environ['PATH'] = os.environ['PATH'] + ':/Library/TeX/texbin' # for tex in matplotlib
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
    # Plot histograms
    # ----------------------------------------------------------------------------------------

    print("Plot precision & recall for class " + labels[k].name)
    file    = 'tables/eval-' + labels[k].name + '-GT.txt'

    fig, (_,_) = plt.subplots(1, 2, figsize=(4, 2))

    for cols, measure in enumerate(["Precision","Recall"]):
        ml  = np.loadtxt(file, skiprows=2, usecols=2*cols+11)
        bay = np.loadtxt(file, skiprows=2, usecols=2*cols+12)

        data1 = bay[np.logical_not(np.isnan(bay))]
        data2 = ml[np.logical_not(np.isnan(ml))]

        counts1, bin_edges1 = np.histogram(data1, bins=15, density=False)
        counts2, bin_edges2 = np.histogram(data2, bins=15, density=False)

        counts1 = counts1.astype(float) / len(data1)
        counts2 = counts2.astype(float) / len(data2)

        cdf1 = np.cumsum(counts1)
        cdf2 = np.cumsum(counts2)

        x1 = bin_edges1[0:-1]
        x2 = bin_edges2[0:-1]

        y1 = cdf1
        y2 = cdf2

        plt.subplot("12"+str(cols+1))
        plt.plot(x1,y1, label="Bayes", color="C0")
        plt.plot(x2,y2, label="ML", color="C1")
        plt.legend(loc=2,prop={'size': 7})
        plt.xlabel(measure)
        plt.xlim(-0.05, 1.05)
        plt.ylim(-0.05, 1.05)
        if cols == 0: plt.ylabel('Cumulative percent')
        plt.tight_layout()

    #fig.suptitle("Performance measures for " + labels[k].name)
    #fig.subplots_adjust(top=0.80)
    fig.savefig("cdf/cdf-" + labels[k].name + ".pdf",transparent=True)

##############################################################################################

print("DONE!")

