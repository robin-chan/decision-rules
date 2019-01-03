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
import tensorflow as tf
from PIL import Image
from argparse import ArgumentParser
from labels import labels
from globals import *


#################################################################################
#
# function that load graph from pb-file
# INPUT:
# frozen_graph_filename: path to pb-file
#
#################################################################################

def load_graph(frozen_graph_filename):
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, input_map=None, return_elements=None,
                            name="prefix", op_dict=None, producer_op_list=None)

    return graph

#################################################################################
#
# function that runs the prediction scenario
# INPUT:
# batch: image file names in the batch to process
# sess: tensorflow session
#
#################################################################################

def prediction(batch, sess):
    input_batch = np.array(
        [np.array(Image.open(file).convert("LA").resize((XDIM, YDIM)))[:, :, :1]
         for file in batch])
    feed_dict = {}
    feed_dict[x] = input_batch
    y_out = sess.run(y, feed_dict = feed_dict)

    return y_out

#################################################################################
#
# function that computes softmax probabilities
# INPUT:
# logits: activation values obtained by prediction model
#
#################################################################################

def softmax(logits):

    sf = np.zeros(logits.shape)

    for i in range(logits.shape[0]):    ### image-wise softmax
        x = logits[i,:,:,:]
        xmax = np.amax(x, axis=-1)
        xmax = xmax.reshape(xmax.shape + (1,))
        xmax1 = xmax
        for k in range(1, x.shape[-1]):
            xmax = np.concatenate((xmax, xmax1), axis=-1)
        e_x = np.exp(x - xmax)
        e_xsum = np.sum(e_x, axis=-1)
        e_xsum = e_xsum.reshape(e_xsum.shape + (1,))
        e_xsum1 = e_xsum
        for k in range(1, x.shape[-1]):
            e_xsum = np.concatenate((e_xsum, e_xsum1), axis=-1)
        sf[i,:,:,:] = e_x / e_xsum

    return sf

#################################################################################
#
# function that shrinks size of array by averaging
# INPUT:
# a: array
# shape: new array shape
#
#################################################################################

def rebin(a, shape):
    sh = shape[0],a.shape[0]//shape[0],shape[1],a.shape[1]//shape[1]
    return a.reshape(sh).mean(-1).mean(1)

#################################################################################
#
# function that loads the priors form file
# INPUT:
# N: batch size
# eps: epsilon cut off value
#
#################################################################################

def load_priors(M,eps):
    epsilon_array = np.full((resolution[1], resolution[0], N), eps)
    tmp = np.zeros((resolution[1], resolution[0], N))
    for i in range(N):
        tmp[:, :, i] = rebin(np.load(work_dir + "out/priors-array/priors_smooth.npy")[:, :, i],
                             (resolution[1],resolution[0]))
    priors_1 = tmp
    priors_1 = priors_1 + epsilon_array
    priors_1[:,:,0] = np.ones((480,640))    ### exclude class BACKGROUND

    priors = np.zeros((M, 480, 640, N))
    priors[0:M,:,:,:] = priors_1            ### expand array shape

    return priors

#################################################################################
#
# function that creates prediction image from prediction array
# INPUT:
# pred: array with shape (XDIM,YDIM), each element with predicted class ID
# savepath: where to save image
#
#################################################################################

def recolor_and_save(pred, savepath):
    new_img = np.zeros((resolution[1], resolution[0], 3), dtype=np.uint8)
    for i in range(resolution[1]):
        for j in range(resolution[0]):
            for k in range(len(labels)):
                if (pred[i, j] == labels[k].Id):
                    new_img[i, j] = labels[k].color
                    break
    Image.fromarray(new_img).save(savepath)


#################################################################################
#
# MAIN
#
#################################################################################

def main(args,x,y):

    DATA = work_dir + args.input
    DR = args.decision
    GAL = args.gal

    tmp = os.listdir(DATA)
    files = [os.path.join(DATA, f) for f in tmp]
    files.sort()

    SAMPLING_BATCH_SIZE = int(x.shape[0])
    BATCHES = int(len(files) / SAMPLING_BATCH_SIZE)
    ### TODO: if images remain -> if int(len(files) % SAMPLING_BATCH_SIZE) != 0

    os.chdir(work_dir + "out/predictions")

    if DR == "ML":
        PRIORS = load_priors(SAMPLING_BATCH_SIZE, 1e-5)
        if not os.path.exists("ML"): os.makedirs("ML")

    else:
        PRIORS = np.ones((SAMPLING_BATCH_SIZE, YDIM, XDIM, N))
        if not os.path.exists("B"): os.makedirs("B")


    with tf.Session(graph=graph) as sess:

        print("########### PARSED ARGUMENTS ###########")
        print("Input folder  : " + DATA)
        print("Decision Rule : " + DR)
        if GAL is not None: print("MC Sampling   : %d" % GAL)


        print("########### PREDICITON START ###########")
        for k in range(BATCHES):

            ### Extract image batches
            batch = files[(k * SAMPLING_BATCH_SIZE):((k + 1) * SAMPLING_BATCH_SIZE)]
            print("Processing batch %d/%d " % (k + 1, BATCHES))

            if GAL is None: ### No Monte Carlo Sampling
                logits = prediction(batch, sess)
                out = softmax(logits)
                pred = np.argmax(out / PRIORS, axis=-1)


            else:   ### Monte Carlo Sampling
                SAMPLING_ROUNDS = GAL
                gal = np.zeros((SAMPLING_BATCH_SIZE, YDIM, XDIM, CDIM))
                for i in range(SAMPLING_ROUNDS):
                    if (i + 1) % 10 == 0: print("%d sampling rounds done ..." % (i+1))
                    logits = prediction(batch, sess)
                    out = softmax(logits)
                    gal = gal + out
                mean = gal / SAMPLING_ROUNDS
                pred = np.argmax(mean / PRIORS, axis=-1)

            ### Save predictions
            for id, file in enumerate(batch):
                recolor_and_save(pred[id, :, :], DR + "/" + str(file.split(os.sep)[-1]))


#################################################################################
#
# RUN MAIN
#
#################################################################################

if __name__ == "__main__":

    XDIM = resolution[0]
    YDIM = resolution[1]
    CDIM = 1
    N = len(labels)

    graph = load_graph(model_path)
    x = graph.get_tensor_by_name(graph_input_node)
    y = graph.get_tensor_by_name(graph_output_node)

    parser = ArgumentParser()
    parser.add_argument("-i", "--input", required=True, help="folder path")
    parser.add_argument("-d", "--decision", choices=["ML","B"],required=True, help="Decision Rule", default="B")
    parser.add_argument("-g", "--gal", type=int, help="MC sampling rounds")
    main(parser.parse_args(),x,y)
    print("########### PREDICITON DONE ############")
