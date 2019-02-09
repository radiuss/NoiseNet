__author__ = "Mykhail L. Uss"
__copyright__ = "Copyright 2018, Mykhail L. Uss"
__credits__ = ["Mykhail L. Uss"]
__license__ = "GPLv3"
__version__ = "1.0.1"
__maintainer__ = "Mykhail L. Uss"
__email__ = "mykhail.uss@gmail.com", "uss@xai.edu.ua"
__status__ = "Prototype"

import rawpy
import exifread
import os
from math import sqrt
import cv2
import numpy as np
from tqdm import trange
import time
import re
import fnmatch


# Nikon D80 calibration nlf for ISO = 100, 200, 400, 800
nlf_d80_iso100 = np.array([[8.42733e-06, 0.142803, 8.0322],
                           [ 6.60859e-06,  0.136115,     4.9582],
                           [2.45252e-05,  0.17193,    12.2661]])
nlf_d80_iso200 = np.array([[8.44655e-06, 0.283699, 32.7126],
                           [ 5.30043e-06,  0.268418,    19.0212],
                           [1.99618e-05,  0.34668,    47.6598]])
nlf_d80_iso400 = np.array([[3.65459e-07, 0.557463, 143.1934],
                           [ 4.33092e-06,  0.513291,    82.0848],
                           [1.66535e-05,  0.67501,   211.2958]])
nlf_d80_iso800 = np.array([[-7.28926e-06, 1.1409160, 223.40250],
                           [-5.743534e-06, 1.0208800, 223.47100],
                           [1.721961e-05, 1.2756660, 293.2322]])

VALID_RAW_FORMATS = ('*.nef', '*.dng')

def calc_noise_nlf(iso):
    """ calculate noise nlf for fiven iso value by
        interpolation between ISO = 100, 200, 400, 800

    :param iso: iso value
    :return: noise nlf (second order polinomial): var = k0 + k1*I+k2*I^2
    """
    nlf = np.zeros((3, 3), dtype=float)
    for i in range(0, 3):
        for j in range(0, 3):
            x = np.array([100.0, 200.0, 400.0, 800.0])
            y = np.array([nlf_d80_iso100[i, j], nlf_d80_iso200[i, j], nlf_d80_iso400[i, j], nlf_d80_iso800[i, j]])
            idx = np.argsort(abs(x - iso))
            idx = idx[0:3]
            p = np.polyfit(x[idx], y[idx], 2)
            nlf[i, j] = np.polyval(p, iso)
    return nlf


def prepare_patch(patch):
    """ prepare normalized patch as input for NoiseNet model

    :param patch:
    :return: normalized patch
    """
    n = patch.shape[1]
    nh = round(n / 2)
    # local mean
    pt_mean = np.mean(patch[0, nh - 3:nh + 3, nh - 3:nh + 3])
    # pt_mean = np.mean(patch[0, nh - 4:nh + 4, nh - 4:nh + 4])
    patch_norm = patch - np.mean(patch)  # local mean subtraction
    k_norm = sqrt(np.var(patch_norm))
    if k_norm > 0:
        # normalization to unit variance
        patch_norm /= k_norm
    return patch_norm, k_norm, pt_mean


def collect_train_data(path_db, patch_size):
    """ collect patched for training

    :param path_db: path the the training database
    :param patch_size: size of the patch
    :return: numpy ndarray if patches of size (?, 1, patch_size, patch_size)
             and array of ground truth noise sd of size (?)
    """
    patches = []
    labels = []
    intensity_max = 3800

    all_img_paths = []
    for files in VALID_RAW_FORMATS:
        rule = re.compile(fnmatch.translate(files), re.IGNORECASE)
        all_img_paths += [os.path.join(path_db, name) for name in os.listdir(path_db) if rule.match(name)]
    n_images = len(all_img_paths)

    n_used_images = 0
    with trange(n_images) as t:
        t.set_description('Processing %d images of test dataset' % n_images)
        for idx in t:
            img_path = all_img_paths[idx]
            with rawpy.imread(img_path) as raw:
                t.set_postfix(name=img_path)
                f = open(img_path, 'rb')
                tags = exifread.process_file(f)
                for tag in tags.keys():
                    if tag == 'EXIF ISOSpeedRatings':
                        iso = tags[tag].values[0]

                if iso > 400:  # for iso > 400 ground truth nlf is not available
                    continue

                n_used_images += 1

                nlf_gt = calc_noise_nlf(iso)
                rgb = np.array(raw.raw_image)
                for scale in (1, 2, 4):  # loop over image scales
                    for idx_channel in range(3):  # loop over image channels
                        if idx_channel == 0:  # red channel
                            img = np.array(rgb[1::2, 0::2]).astype(dtype='float32')
                        elif idx_channel == 1:  # green channel
                            img = np.array(rgb[0::2, 0::2]).astype(dtype='float32')
                        elif idx_channel == 2:  # blue channel
                            img = np.array(rgb[0::2, 1::2]).astype(dtype='float32')
                        elif idx_channel == 3:  # green channel
                            img = np.array(rgb[1::2, 1::2]).astype(dtype='float32')

                        # get nlf for the given channel (channels 1 and 3 are both green and have the same nlf)
                        if idx_channel < 3:
                            nlf = np.array(nlf_gt[idx_channel, 0:3])
                        else:
                            nlf = np.array(nlf_gt[1, 0:3])

                        if scale > 1:  # downscale image
                            kernel = np.ones(scale) / scale
                            img = cv2.sepFilter2D(img, -1, kernel, kernel)
                            img = np.array(img[scale:-scale:scale, scale:-scale:scale])
                            nlf /= scale**2

                        # add patches from regular grid
                        stp = int(1.5 * patch_size)
                        for i in range(idx_channel * 16, img.shape[0] - 2 * patch_size, stp):
                            for j in range(idx_channel * 16, img.shape[1] - 2 * patch_size, stp):
                                # prepare patch
                                patch = np.array(img[i:i + patch_size, j:j + patch_size]).astype(dtype='float32')
                                patch = patch.reshape(1, patch_size, patch_size)
                                patch_norm, k_norm, patch_mean = prepare_patch(patch)
                                sd_gt = sqrt(np.polyval(nlf[0:3], patch_mean))

                                #  skip patches affected by intensity clipping
                                #  and abnormally flat patches
                                if (patch_mean - 5. * sd_gt * scale > 0 and
                                        patch_mean + 5. * sd_gt * scale < intensity_max and k_norm > 0.2 * sd_gt):

                                    if scale > 1:
                                        for k in range(1):
                                            k_norm_to_255 = (np.max(patch) - np.min(patch)) / np.random.uniform(10, 255)
                                            patch_tmp = (patch - np.min(patch)) / k_norm_to_255
                                            sd0 = np.random.uniform(2.5, 15.)
                                            patch_tmp = patch_tmp + np.random.normal(0., sd0, (patch_size, patch_size))
                                            patch_tmp = np.round(patch_tmp)
                                            patch_norm, k_norm_tmp, patch_mean = prepare_patch(patch_tmp)
                                            sd_gt_tmp = sqrt((sd_gt / k_norm_to_255) ** 2 + sd0 ** 2)
                                            patches.append(patch_norm)
                                            labels.append(sd_gt_tmp / k_norm_tmp)
                                    else:
                                        patches.append(patch_norm)
                                        labels.append(sd_gt / k_norm)
    time.sleep(0.5)
    print("Adding pure noise patches")

    # add 25% of pure noise patches
    for i in range(int(len(patches)*0.25)):
        fr = np.random.normal(0., 1., (1, patch_size, patch_size))
        fr_norm, k_norm, patch_mean = prepare_patch(fr)
        sd0_norm = 1.0 / k_norm
        patches.append(fr_norm)
        labels.append(sd0_norm)

    time.sleep(0.5)
    n_samples = len(labels)
    print("Train data collected. Number of samples = %d. Samples per image is %f" %
          (n_samples, n_samples / n_used_images))

    patches = np.asarray(patches, dtype='float32')
    labels = np.asarray(labels, dtype='float32')

    # randomize patches order
    idx_perm = np.random.permutation(np.arange(n_samples))
    patches = patches[idx_perm, :, :]
    labels = labels[idx_perm]

    return patches, labels
