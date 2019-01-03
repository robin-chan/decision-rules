#!/usr/bin/python
#
# Set global variables here
#

import os
from labels import *


#--------------------------------------------------------------------------------
# Set paths to directories
#--------------------------------------------------------------------------------

work_dir = os.getcwd() + "/"
gt_train_dir = "data/GT-train"
ground_truth_dir = work_dir + "data/GT/"
input_dir = work_dir + "data/INPUT/"

#--------------------------------------------------------------------------------
# Select model and classes to analyze
#--------------------------------------------------------------------------------

model_path = work_dir + "data/model/frozen-graph.pb"
graph_input_node = "input-node-name"
graph_output_node = "output-node-name"
resolution = (640,480)
class_indices = [name2label['PERSON'].Id,name2label['INFO'].Id]