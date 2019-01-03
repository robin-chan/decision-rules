# ==============================================================================
# Author:   Robin Chan, University of Wuppertal
# Contact:  rchan@uni-wuppertal.de
# GitHub:   https://github.com/robin-chan
# ==============================================================================

"""
Count how often one class appears at every pixel position
for creating pixel-wise a-priori probabilities.
Output: priors array
"""


import numpy as np
import multiprocessing
from PIL import Image
from labels import labels
from globals import *

#--------------------------------------------------------------------------------
# convert label colors to numpy array
#--------------------------------------------------------------------------------

color = []
l = len(labels)
for i in range(l):
    color.append(np.asarray(labels[i].color))

#--------------------------------------------------------------------------------
# function for counting pixels and saving into array
#--------------------------------------------------------------------------------

def count_pixel(filename):

    ### import image, count pixel
    img = np.array(Image.open(filename).convert('RGB'))
    counter = np.zeros([HEIGHT,WIDTH,l])
    for i in range(HEIGHT):
        for j in range(WIDTH):
            for k in range(l):
                if (img[i,j] == color[k]).all():
                    counter[i,j,k] = 1
                    break
    return counter

#--------------------------------------------------------------------------------
# initialization
#--------------------------------------------------------------------------------

gt = gt_train_dir
gt_list  = os.listdir(gt)
N = len(gt_list)
HEIGHT,WIDTH,_ = np.array(Image.open(gt+gt_list[0])).shape
priors = np.zeros([HEIGHT,WIDTH,l])

core = 10   ### Number of cores to be used
pool = multiprocessing.Pool(processes = core)
n = int(N/core)

#--------------------------------------------------------------------------------
# create priors array and save to file
#--------------------------------------------------------------------------------

print("########################################################################")
print("Computing priors:")

for k in range(n):

    out = pool.map(count_pixel,((gt+gt_list[i]) for i in range(k*core,(k+1)*core)))
    for j in range(core): priors=priors+out[j]
    if ((k+1)*core)%10 == 0: print("{}/{} images processed" .format((k+1)*core,N))

np.save(work_dir + "out/priors-array/priors", priors/N)
print("DONE!")