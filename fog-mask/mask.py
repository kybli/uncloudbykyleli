import argparse
import cv2
from glob import glob
import logging
import numpy as np
import math
import skimage.io as io, skimage.morphology as morph
import os

# value of 0 = masked (foggy)
# value of 1 = unmasked (not foggy)

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--foggy-dir', type=str, required=True)
    parser.add_argument('--mask-dir', type=str, required=True)
    parser.add_argument('-r', '--resume', action='store_true', help="resume run")

    return parser.parse_args()

def dark_channel(img, size):
    dark_channel_img = np.amin(img, axis=2)
    footprint = np.ones((size, size))
    dark_channel_img = morph.erosion(dark_channel_img, footprint)
    return dark_channel_img

def atmospheric_light(img, dark_channel_img):
    h, w = img.shape[:2]
    img_size = h * w
    top_point_one_percent = int(max(math.floor(img_size / 1000), 1))

    flat_dark_channel_img = dark_channel_img.reshape(img_size)
    flat_img = img.reshape(img_size, 3)
    flat_img_intensities = np.mean(img, 2).reshape(img_size)

    brightest_dc_indexes = np.argpartition(flat_dark_channel_img, -top_point_one_percent)[-top_point_one_percent:]
    
    top_atm_pixel_count = int(max(top_point_one_percent * 0.5, 1))
    atm_light_pixel_indexes = np.argpartition(flat_img_intensities[brightest_dc_indexes], -top_atm_pixel_count)[-top_atm_pixel_count:]

    atm_light = np.mean(np.take(np.take(flat_img, brightest_dc_indexes, 0), atm_light_pixel_indexes, 0), 0)

    return atm_light

def transmission_estimate(img, atm_light, size):
    w = 0.95
    return (255 * (1 - (w * dark_channel((img / atm_light), size)))).astype(np.uint8)

def refine_transmission(img, transmission_estimate_img):
    r = 60      # radius of kernel
    e = 0.0001  # regularization

    gray_scale_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY);
    gray_scale_img = np.float64(gray_scale_img)/255;

    transmission_estimate_img = transmission_estimate_img / 255

    mean_I = cv2.boxFilter(gray_scale_img, cv2.CV_64F, (r,r))
    mean_p = cv2.boxFilter(transmission_estimate_img, cv2.CV_64F, (r,r))
    corr_I = cv2.boxFilter(gray_scale_img * gray_scale_img,cv2.CV_64F, (r,r))
    corr_Ip = cv2.boxFilter(gray_scale_img * transmission_estimate_img, cv2.CV_64F, (r,r))

    var_I   = corr_I - (mean_I * mean_I)
    cov_Ip = corr_Ip - mean_I * mean_p

    a = cov_Ip / (var_I + e)
    b = mean_p - (a * mean_I)

    mean_a = cv2.boxFilter(a, cv2.CV_64F, (r,r))
    mean_b = cv2.boxFilter(b, cv2.CV_64F, (r,r))

    q = (mean_a * gray_scale_img) + mean_b
    return (q * 255).astype(np.uint8)


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    logging.info("Beginning mask.py")

    args = parse_args()

    src_dir = args.foggy_dir
    dest_dir = args.mask_dir
    resume = args.resume

    src_file_paths = glob('{:s}/**/*.png'.format(src_dir), recursive=True)
    
    for src_file_path in src_file_paths:
        # set up file paths
        src_rel_path = os.path.relpath(src_file_path, start=src_dir)
        output_rel_path = src_rel_path[:-4] + "_mask.png"
        output_file_path = os.path.join(dest_dir, output_rel_path)

        _, src_file_name = os.path.split(src_file_path)
        output_dir, _ = os.path.split(output_file_path)

        # if resume and file exits, skip it
        if resume and os.path.isfile(output_file_path):
            logging.info("RESUME: SKIP {}. It already exitst".format(src_file_name))
            continue

        # create output file dir if it doesn't exist
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
            logging.info("Output directory, {}, does not exist. Successfully created output directory".format(output_dir))

        # read image and compute dark channel prior
        src_img = io.imread(src_file_path)

        dark_channel_img = dark_channel(src_img, 15)
        atm_light = atmospheric_light(src_img, dark_channel_img)
        transmission_estimate_img = transmission_estimate(src_img, atm_light, 15)
        refined_transmission_img = refine_transmission(src_img, transmission_estimate_img)

        # save everything
        io.imsave(output_file_path, refined_transmission_img)
        logging.info("Saved mask for {} as {}".format(src_file_name, output_file_path))
    