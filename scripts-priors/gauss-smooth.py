# ==============================================================================
# Author:   Robin Chan, University of Wuppertal
# Contact:  rchan@uni-wuppertal.de
# GitHub:   https://github.com/robin-chan
# ==============================================================================

"""
Smooth priors in order to reduce data specific noise
Output: smoothed priors array
"""

import numpy as np
from scipy import ndimage
from labels import labels
from globals import *

#--------------------------------------------------------------------------------
# initialization
#--------------------------------------------------------------------------------

priors = np.load(work_dir + "out/priors-array/priors.npy")
sigma = 80
p = np.zeros(priors.shape)

#--------------------------------------------------------------------------------
# smoothening
#--------------------------------------------------------------------------------

print("########################################################################")
print("Smooth priors:")
print("Wait...")

for k in range(len(labels)):
    p[:,:,k]=ndimage.gaussian_filter(priors[:,:,k], sigma)

np.save(work_dir + "out/priors-array/priors_smooth",p)
print("SMOOTHENING PRIORS DONE!")