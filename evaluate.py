__author__ = "Mykhail L. Uss"
__copyright__ = "Copyright 2018, Mykhail L. Uss"
__credits__ = ["Mykhail L. Uss"]
__license__ = "GPLv3"
__version__ = "1.0.1"
__maintainer__ = "Mykhail L. Uss"
__email__ = "mykhail.uss@gmail.com", "uss@xai.edu.ua"
__status__ = "Prototype"

import os
import scipy.io
from scipy.stats import norm
import rawpy
import exifread
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from nlf_estimation import NoiseEstimator, NoiseModel
from utils import calc_noise_nlf, prepare_patch, VALID_RAW_FORMATS
import xlwt
import time
from tqdm import trange
import re
import fnmatch


class Evaluator:
    """ NoiseNet model evaluation on pure noise and on NED2012 database.

    """

    def __init__(self, model, sigma_rel_th, verbose=0, out_dir=None):
        self.model = model
        # self.patch_size = int(self.model.get_layer("input_1").get_output_at(0).get_shape()[2])  # get input patch size
        self.patch_size = 32
        self.sigma_rel_th = sigma_rel_th
        self.verbose = verbose
        self.out_dir = out_dir
        self.sd_nlf_error_noisenet = None
        self.sd_nlf_error_nidct = None

    def get_res(self):
        """ get NoiseNet results obtained for NED2012 database

        :return: Relative SD (in percents) of SD and SI noise components estimation
        """
        if self.sd_nlf_error_noisenet is None:
            print("Results for NED2012 database are not available")
            return 0., 0.
        else:
            return 100 * self.sd_nlf_error_noisenet[0], 100 * self.sd_nlf_error_noisenet[1]

    def print_res(self):
        """ display NoiseNet results obtained for NED2012 database

        """
        if self.sd_nlf_error_noisenet is not None:
            print("NI+NoiseNet. Relative estimation error SD:      SD component %f%%, SI component %f%%"
                  % (100 * self.sd_nlf_error_noisenet[0], 100 * self.sd_nlf_error_noisenet[1]))
        else:
            print("Results for NI+NoiseNet are not available")
        if self.sd_nlf_error_nidct is not None:
            print("NI+DCT.      Relative estimation error SD:      SD component %f%%, SI component %f%%"
                  % (100 * self.sd_nlf_error_nidct[0], 100 * self.sd_nlf_error_nidct[1]))
        else:
            print("Results for NI+DCT are not available")

    def process_NED2012(self, path_db):
        """ Collect stats for NED2012 database
            Noise NLF is stored as (2, 3, ?) array
            axis=0 stores NLF coefficients
            NLF[0, :, :] is signal-dependent (SD) noise variance component
            NLF[1, :, :] is signal-independent (SI) noise variance
            axis=1 is image channels (red, green, blue)
            axis=2 is database images

        :param path_db: path to the NED2012 database
        """

        # scan dataset for .nef or .dng images
        all_img_paths = []
        for files in VALID_RAW_FORMATS:
            rule = re.compile(fnmatch.translate(files), re.IGNORECASE)
            all_img_paths += [os.path.join(path_db, name) for name in os.listdir(path_db) if rule.match(name)]
        # read ISO value for each image
        n_images = len(all_img_paths)
        if n_images == 0:
            return
        path_aux_data = os.path.join(path_db, 'NED2012_Auxiliary.mat')
        if not os.path.exists(path_aux_data):
            print("No auxiliary data found in the path provided for NED2012 database. Aborting evaluation.")
            return

        iso = self.__read_iso(all_img_paths)

        # process NED2012 dataset and normalize nlf to ISO=100
        start = time.time()
        nlf_hat_noisenet = self.__process_ds(all_img_paths, iso)
        end = time.time()
        nlf_hat__noise_net_iso100 = self.__normalize_to_iso100(nlf_hat_noisenet, iso)
        if self.verbose == 1:
            print("NED2012 processed in %fs. Time per image is %fs.\n" % (end - start, (end - start) / n_images))

        # calculate ground truth nlf using calibration data for D80 camera and normalize nlf to ISO=100
        nlf_gt = self.__calc_gt_nlf(iso)
        nlf_gt_iso100 = self.__normalize_to_iso100(nlf_gt, iso)

        #  get NI+DCT measurement (originally saved for ISO=100)
        mat = scipy.io.loadmat(path_aux_data)
        nlf_gt_nidct_iso100 = mat['SD2_NI_DCT_JEI']
        nlf_gt_nidct_iso100 = nlf_gt_nidct_iso100[:, :, ::-1]
        nlf_gt_nidct_iso100 = np.transpose(nlf_gt_nidct_iso100, (2, 1, 0))

        # calculate bias and SD of noise components estimation error w.r.t. calibration data
        mean_nlf = np.mean(nlf_gt_iso100, axis=(1, 2))
        err_NINoiseNet = np.zeros((2, 3))
        err_NIDCT = np.zeros((2, 3))
        for i in np.arange(0, 2, 1):
            for j in np.arange(0, 3, 1):
                tmp = 100*(nlf_hat__noise_net_iso100[i, j, :]-nlf_gt_iso100[i, j, :])/mean_nlf[i]
                err_NINoiseNet[i, j] = sqrt(np.mean(tmp**2))
                tmp = 100*(nlf_gt_nidct_iso100[i, j, :]-nlf_gt_iso100[i, j, :])/mean_nlf[i]
                err_NIDCT[i, j] = sqrt(np.mean(tmp**2))

        if self.verbose == 1:
            print('SD signal dependent. NI+NoiseNet  : ', end="")
            print(err_NINoiseNet[0, :])
            print('SD signal dependent. NI+DCT  : ', end="")
            print(err_NIDCT[0, :])
            print('SD signal independent. NI+NoiseNet: ', end="")
            print(err_NINoiseNet[1, :])
            print('SD signal independent. NI+DCT: ', end="")
            print(err_NIDCT[1, :])
            print(' ')

        err_norm = (nlf_hat__noise_net_iso100 - nlf_gt_iso100) / nlf_gt_iso100
        sd_nlf_error = np.sqrt(np.mean(err_norm ** 2, axis=(1, 2)))
        self.sd_nlf_error_noisenet = sd_nlf_error
        if self.verbose == 1:
            print("NI+NoiseNet. Relative estimation error SD:      SD component %f%%, SI component %f%%"
                  % (100. * sd_nlf_error[0], 100. * sd_nlf_error[1]))
            sd_nlf_error = 1.48 * np.median(np.abs(err_norm), axis=(1, 2))
            print("NI+NoiseNet. Relative estimation error 1.48MAD: SD component %f%%, SI component %f%%"
                  % (100. * sd_nlf_error[0], 100. * sd_nlf_error[1]))

        err_norm = (nlf_gt_nidct_iso100 - nlf_gt_iso100) / nlf_gt_iso100
        sd_nlf_error = np.sqrt(np.mean(err_norm ** 2, axis=(1, 2)))
        self.sd_nlf_error_nidct = sd_nlf_error
        if self.verbose == 1:
            print("NI+DCT.    Relative estimation error SD:      SD component %f%%, SI component %f%%"
                  % (100. * sd_nlf_error[0], 100. * sd_nlf_error[1]))
            sd_nlf_error = 1.48 * np.median(np.abs(err_norm), axis=(1, 2))
            print("NI+DCT.    Relative estimation error 1.48MAD: SD component %f%%, SI component %f%%\n"
                  % (100. * sd_nlf_error[0], 100. * sd_nlf_error[1]))

        # save results
        if self.out_dir is not None:
            wb = xlwt.Workbook()
            ws = wb.add_sheet(self.model.name)

            ws.write(0, 0, 'Noise component')
            ws.write(0, 1, 'Reg')
            ws.write(0, 2, 'Green')
            ws.write(0, 3, 'Blue')
            ws.write(1, 0, 'SD signal dependent. NoiseNet, %')
            ws.write(2, 0, 'SD signal dependent. NI+DCT, %')
            ws.write(3, 0, 'SD signal independent. NoiseNet, %')
            ws.write(4, 0, 'SD signal independent. NI+DCT, %')

            for i in range(3):
                ws.write(1, i + 1, err_NINoiseNet[0, i])
                ws.write(2, i + 1, err_NIDCT[0, i])
                ws.write(3, i + 1, err_NINoiseNet[1, i])
                ws.write(4, i + 1, err_NIDCT[1, i])

            stats_path = os.path.join(self.out_dir, self.model.name+"_stats.xls")
            wb.save(stats_path)

            if self.verbose == 1:
                print("Evaluation results saved to %s" % stats_path)

    @staticmethod
    def __normalize_to_iso100(nlf, iso):
        """ Convert Signal-dependent NLF (Noise Level Function) from arbitrary ISO value to ISO=100
            Signal-dependent noise variance component is transformed as 100/ISO
            signal-independent noise variance is transformed as (100/ISO)^2

        :param nlf: vector of NLF's to convert. ndarray of size (2, n_ch, n_nlf)
                    where nlf[0, :, :] is signal-dependent noise variance component,
                    nlf[1, :, :] is signal-independent noise variance
                    n_ch is number of channels, n_nlf is number of nlf's
        :param iso: vector of ISO values for each nlf entry. Array of size (n_nlf)
        :return: vector of normalized nlf's of the same size as nlf
        """

        if nlf.ndim != 3 or nlf.shape[0] != 2 or iso.size != nlf.shape[2]:
            print("NLF normalization failed due to sizes mismatch.")
            return nlf

        n = iso.size
        n_ch = nlf.shape[1]
        k_norm = np.zeros((2, 1, n))
        k_norm[0, 0, :] = 100. / iso
        k_norm[1, 0, :] = (100. / iso) ** 2
        nlf_norm = nlf * np.repeat(k_norm, n_ch, axis=1)
        return nlf_norm

    @staticmethod
    def __read_iso(paths):
        """ read ISO values from nef or dng image files

        :param paths: list of images paths
        :return:
        """
        n_im = len(paths)
        iso = np.zeros(n_im)
        for idx, img_path in enumerate(paths):
            f = open(img_path, 'rb')
            tags = exifread.process_file(f)
            for tag in tags.keys():
                if tag == 'EXIF ISOSpeedRatings':
                    iso[idx] = tags[tag].values[0]
        return iso

    @staticmethod
    def __calc_gt_nlf(iso):
        n_im = iso.size
        res = np.zeros((2, 3, n_im))
        for idx, iso_cur in enumerate(iso):
            nlf = calc_noise_nlf(iso_cur)
            nlf = np.transpose(nlf, (1, 0))
            res[0:2, :, idx] = nlf[1:3, :]

        return res

    def __process_ds(self, img_paths, iso=None):
        """ Apply NI+NoiseNet estimator to a set of nef or dng images in channel-wise manner
            is iso value is provided, Nikon D80 NLF non-linearity correction will be applied
            For details, see NED2012 database description or refer to
            M. L. Uss, B. Vozel, V. V. Lukin, and K. Chehdi, "Image informative maps for component-wise
            estimating parameters of signal-dependent noise," J. Electron. Imaging, vol. 22, pp. 013019-013019, 2013."

        :param img_paths: list of images to process
        :param iso: list of iso values for each image
        :return: ndarray of nlf-s for each image channel and for each image in the set
        """
        n_images = len(img_paths)
        res = np.zeros((2, 3, n_images))

        # initialize estimator
        n_samples = 2000
        sd_init = 100
        intensity_max = 3700
        estimator = NoiseEstimator(self.model, NoiseModel.ADDITIVE_POISSON, n_samples, sd_init, intensity_max)

        time.sleep(0.5)
        with trange(n_images) as t:
            t.set_description('Processing %d images of NED2012 dataset' % n_images)
            for idx in t:
                img_path = img_paths[idx]

                with rawpy.imread(img_path) as raw:

                    t.set_postfix(name=img_path)

                    if iso is None:
                        nlf_gt = None
                    else:
                        nlf_gt = calc_noise_nlf(iso[idx])

                    rgb = np.array(raw.raw_image)
                    for idx_channel in range(3):
                        # read one image spectral component
                        # Valid for bayer pattern of Nikon D80 camera
                        if idx_channel == 0:  # red channel
                            img = np.array(rgb[1::2, 0::2]).astype(dtype='float32')
                        elif idx_channel == 1:  # green channel
                            img = np.array(rgb[0::2, 0::2]).astype(dtype='float32')
                        elif idx_channel == 2:   # blue channel
                            img = np.array(rgb[0::2, 1::2]).astype(dtype='float32')

                        # estimate nlf by NI+NoiseNet method
                        if nlf_gt is None:
                            nlf_hat = estimator.calculate_noise_level_function(img)
                        else:
                            # Nikon D80 NLF linearization
                            # Note that while true NLF is provided to the estimator, it is
                            # only used to agree image NLF with var_0 + var_sd * I model
                            # No knowledge of true NLF is used during noise nlf estimation process
                            nlf_hat = estimator.calculate_noise_level_function(img, nlf_gt[idx_channel, :])

                        res[:, idx_channel, idx] = nlf_hat

        return res

    def pure_noise_test(self):
        """ Applies NoiseNet to realizations of pure gaussian noise
            in order to study estimates bias and accuracy of SD estimation SD prediction

        """
        n_test = 10000  # set number of test samples
        sd0 = 1  # noise standard deviation
        sd_hat = np.zeros(n_test)
        sd_sd_hat = np.zeros(n_test)
        for idx in range(n_test):
            patch_norm, k_norm, pt_mean = prepare_patch(np.random.normal(0, sd0, (1, self.patch_size, self.patch_size)))
            sd_init = np.array(sd0 / k_norm, ndmin=2)
            score = self.model.predict([patch_norm.reshape(1, 1, self.patch_size, self.patch_size), sd_init])
            sd_hat[idx] = score[0, 0] * k_norm
            sd_sd_hat[idx] = score[0, 1] * sd0

        # calculate weighted mean sd estimate
        sd_mean = np.mean(sd_hat / sd_sd_hat ** 2) / np.mean(1.0 / sd_sd_hat ** 2)
        norm_sd = sqrt(np.mean(((sd_hat - sd0) / sd_sd_hat) ** 2))  # Relative estimation error sd
        sd_sd_min = sd0 / (self.patch_size * sqrt(2))  # Potential sd estimation error
        print("Pure noise test")
        print("  -- True/estimated noise SD = %f/%f. Estimation bias = %f%%" % (sd0, sd_mean, 100.*(sd_mean - sd0) / sd0))
        print("  -- Potential SD estimation error (sample SD) = %f, NoiseNet estimation error SD = %f, mean error SD predicted by NoiseNet = %f" %
              (sd_sd_min, np.std(sd_hat), np.mean(sd_sd_hat)))
        print("  -- Relative estimation error SD (equals unity if uncertainty quantifier ideally predicts regressor error SD) = %f" % norm_sd)

        plt.figure(figsize=(10, 7))
        plt.subplot(211)
        plt.hist(sd_hat, bins=100, density=True, label='SD estimates by NoiseNet')
        x = np.linspace(norm.ppf(0.001, sd0, sd_sd_min), norm.ppf(0.999, sd0, sd_sd_min), 100)
        plt.plot(x, norm.pdf(x, sd0, sd_sd_min), 'r-', lw=2, alpha=0.6, label='Potential SD estimates')
        plt.xlabel("Noise SD estimates by NoiseNet")
        plt.ylabel("Probability density")
        plt.legend()
        plt.subplot(212)
        sd_sd_range = (0., 3.0*sd_sd_min)
        plt.hist(sd_sd_hat, bins=100, density=True, range=sd_sd_range, label='Noise SD estimates SD prediction by NoiseNet')
        tmp = 1.48 * np.median(np.abs(sd_sd_hat - np.median(sd_sd_hat)))
        plt.plot((sd_sd_min, sd_sd_min), (0, norm.pdf(0., 0., tmp)), 'r-', lw=2, alpha=0.6, label='Sample SD')
        plt.xlim(sd_sd_range)
        plt.xlabel("Noise SD estimates SD prediction by NoiseNet")
        plt.ylabel("Probability density")
        plt.legend()
        stats_path = os.path.join(self.out_dir, self.model.name + "_stats_pure_noise.png")
        plt.savefig(stats_path, dpi=300)
        if self.verbose == 1:
            print("Evaluation results for pure noise saved to %s" % stats_path)
        plt.close()