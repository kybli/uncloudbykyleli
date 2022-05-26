from skimage import filters
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.metrics import mean_squared_error
import skimage.io as io
from skimage.util import img_as_ubyte

import argparse
import cv2
from glob import glob
import logging
import numpy as np
import os
import re

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--foggy-dir', type=str, default=None)
    parser.add_argument('--defogged-dir', type=str, default=None)
    parser.add_argument('--gt-dir', type=str, required=True)
    parser.add_argument('--metric-file', type=str, default=None)
    parser.add_argument('--edge-dir', type=str, default=None)
    parser.add_argument('--use-edge-metrics', action='store_true')

    return parser.parse_args()


def tryint(s):
    try:
        return int(s)
    except:
        return s


def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [ tryint(c) for c in re.split('([0-9]+)', s) ]


def edge_detection_metrics(img, gt, return_edges=False, use_edge=True):
    if return_edges and not use_edge:
        raise Exception("cannot set return edges to true, but not use edges")
    
    img_ = img
    gt_ = gt

    if use_edge:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gt_gray = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY)

        img_ = filters.sobel(img_gray)
        gt_ = filters.sobel(gt_gray)

        ssim = structural_similarity(img_, gt_)
    else:
        ssim = structural_similarity(img_, gt_, channel_axis=2)

    mse = mean_squared_error(img_, gt_)
    psnr = peak_signal_noise_ratio(img_, gt_)

    if return_edges:
        return mse, ssim, psnr, img_, gt_

    return mse, ssim, psnr


if __name__ == '__main__':
    # file_name = "extremely_foggy_highway"
    # file_path = "~/proj-x/fog-mask/img/input/{}.png".format(file_name)
    logging.getLogger().setLevel(logging.INFO)

    args = parse_args()


    foggy_dir = args.foggy_dir
    defogged_dir = args.defogged_dir
    gt_dir = args.gt_dir

    if not(foggy_dir or defogged_dir):
        raise Exception("need a foggy dir or a defogged dir")
    
    gt_paths = glob('{:s}/*.jpg'.format(gt_dir))
    gt_paths.sort(key=alphanum_key)

    foggy_paths = None
    defogged_paths = None

    if foggy_dir:
        foggy_paths = glob('{:s}/*.jpg'.format(foggy_dir))
        foggy_paths.sort(key=alphanum_key)

    if defogged_dir:
        defogged_paths = glob('{:s}/*.jpg'.format(defogged_dir))
        defogged_paths.sort(key=alphanum_key)
    
    metric_file_path = args.metric_file
    metric_file = open(metric_file_path, 'w')
    metric_file.write("-----------------------------------------------------------------------------------------------------------\n")
    
    edge_dir = args.edge_dir

    for i in range(len(gt_paths)):
        logging.info("{} | COMPUTING METRICS FOR:".format(str(i)))
        gt_file_name = gt_paths[i].split("/")[-1]
        logging.info(gt_file_name)

        gt = io.imread(gt_paths[i])
        
        if foggy_paths:
            foggy_file_name = foggy_paths[i].split("/")[-1]
            logging.info(foggy_file_name)

            fog_img = io.imread(foggy_paths[i])

            if args.use_edge_metrics:
                mse_fog, ssim_fog, psnr_fog, edge_fog, edge_gt = edge_detection_metrics(fog_img, gt, return_edges=True)
            else:
                mse_fog, ssim_fog, psnr_fog = edge_detection_metrics(fog_img, gt, return_edges=False, use_edge=False)

        if defogged_paths:
            defogged_file_name = defogged_paths[i].split("/")[-1]
            logging.info(defogged_file_name)

            defogged_img = io.imread(defogged_paths[i])

            if args.use_edge_metrics:
                mse_defogged, ssim_defogged, psnr_defogged, edge_defogged, _ = edge_detection_metrics(defogged_img, gt, return_edges=True)
            else:
                mse_defogged, ssim_defogged, psnr_defogged = edge_detection_metrics(defogged_img, gt, return_edges=False, use_edge=False)

        logging.info("\n")

        metric_file.write("[{}]\n\n".format(gt_file_name))
        if foggy_paths:
            metric_file.write("FOGGY IMAGE STATS:\n")
            metric_file.write("MSE: {}\n".format(mse_fog))
            metric_file.write("PSNR: {}\n".format(psnr_fog))
            metric_file.write("SSIM: {}\n\n".format(ssim_fog))
        
        if defogged_paths:
            metric_file.write("DEFOGGED IMAGE STATS:\n")
            metric_file.write("MSE: {}\n".format(mse_defogged))
            metric_file.write("PSNR: {}\n".format(psnr_defogged))
            metric_file.write("SSIM: {}\n\n".format(ssim_defogged))
        
        if edge_dir and args.use_edge_metrics:
            gt_edge_dir = os.path.join(edge_dir, "gt")
            input_edge_dir = os.path.join(edge_dir, "input")
            output_edge_dir = os.path.join(edge_dir, "output")

            if not os.path.isdir(gt_edge_dir):
                os.makedirs(gt_edge_dir)
            if not os.path.isdir(input_edge_dir):
                os.makedirs(input_edge_dir)
            if not os.path.isdir(output_edge_dir):
                os.makedirs(output_edge_dir)

            io.imsave(os.path.join(gt_edge_dir, "{}_edge.jpg".format(gt_file_name.split('.')[0])), img_as_ubyte(edge_gt))
            io.imsave(os.path.join(input_edge_dir, "{}_edge.jpg".format(foggy_file_name.split('.')[0])), (edge_fog * 255).astype(np.uint8))
            io.imsave(os.path.join(output_edge_dir, "{}_edge.jpg".format(defogged_file_name.split('.')[0])), (edge_defogged * 255).astype(np.uint8))

            logging.info("saved edges for {}".format(gt_file_name))

        metric_file.write("\n\n\n")
