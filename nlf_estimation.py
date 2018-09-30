__author__ = "Mykhail L. Uss"
__copyright__ = "Copyright 2018, Mykhail L. Uss"
__credits__ = ["Mykhail L. Uss"]
__license__ = "GPLv3"
__version__ = "1.0.1"
__maintainer__ = "Mykhail L. Uss"
__email__ = "mykhail.uss@gmail.com", "uss@xai.edu.ua"
__status__ = "Prototype"

import cv2
import numpy as np
from math import sqrt
from enum import Enum
import statsmodels.api as sm
from utils import prepare_patch


class NoiseModel(Enum):

    """Noise NFL type.

    """
    ADDITIVE = 1          # var = var_0
    ADDITIVE_POISSON = 2  # var = var_0 + var_sd * I
                          # where var is noise variance,
                          # var_0 is signal-independent noise variance
                          # var_sd is signal-dependent noise component


class NoiseEstimator:

    """ Implementation of NI+NoiseNet noise NFL parameters estimation.

    Only signal-dependent model is supported
    """

    def __init__(self, model, type_nlf, n_samples, sd_init, intensity_max, sigma_rel_th=0.2):
        """

        :param model: NoiseNet model
        :param type_nlf: noise NLF type
        :param n_samples: number of samples per image
        :param sd_init: initial value of noise variance
        :param intensity_max: maximum expected image intensity
        :param sigma_rel_th: threshold on relative noise SD estimation error
        """
        self.model = model
        self.type_nlf = type_nlf
        self.sigma_rel_th = sigma_rel_th
        self.n_samples = n_samples
        self.patch_size = int(self.model.get_layer("input_1").get_output_at(0).get_shape()[2])  # get input patch size
        self.sd_init = sd_init
        self.intensity_max = intensity_max
        self.iter_max = 4  # number if iterations for noise nlf refinement
        self.sd_min = 0.1  # min allowed noise SD value

    def calculate_noise_level_function(self, img, nlf_gt=None):
        """ NI+NoiseNet implementation
            supports only signal-dependent noise model

        :param img: image to process
        :param nlf_gt: ground truth noise nlf. If provided used to correct nlf non-linearity
                       (compensates for the quadratic nlf term)
        :return: noise nlf estimate
        """
        s_half = round(self.patch_size / 2)  # path half size

        if self.type_nlf == NoiseModel.ADDITIVE_POISSON:
            pos_raw, pos_col = self.draw_uniform_patches(img, self.n_samples, self.patch_size, 50, self.intensity_max)
        else:
            print("Noise NLF is not supported")
            return np.zeros((1,))
        n_samples = len(pos_raw)

        patches = np.zeros((n_samples, 1, self.patch_size, self.patch_size), dtype='float32')
        coeff_norm = np.zeros((n_samples, 1), dtype='float32')  # proportionality coeff between original
                                                                # and normalized patch noise SD
        sd_pred = np.zeros((n_samples, 1), dtype='float32')  # prediction of noise SD for each patch
        patch_mean_intensity = np.zeros((n_samples, 1), dtype='float32')  # patch mean intensity
        for k in range(n_samples):
            i = pos_raw[k]
            j = pos_col[k]
            # prepare patch
            patch = np.array(img[(i - s_half):(i + s_half), (j - s_half):(j + s_half)]).astype(dtype='float32')
            patch = patch.reshape(1, self.patch_size, self.patch_size)
            patch_norm, k_norm, pt_mean = prepare_patch(patch)
            k_corr = 1
            if nlf_gt is not None:  # optional nlf quadratic term compensation
                if nlf_gt.size == 3:
                    sd0nl = sqrt(np.polyval(nlf_gt[0:3], pt_mean))
                    sd0 = sqrt(np.polyval(nlf_gt[1:3], pt_mean))
                    k_corr = sd0 / sd0nl

            patches[k, :, :, :] = patch_norm
            coeff_norm[k] = k_norm * k_corr
            sd_pred[k] = self.sd_init
            patch_mean_intensity[k] = pt_mean

        for i in range(self.iter_max):
            sd_guess = 1.0 * sd_pred / coeff_norm
            sd_guess = np.minimum(sd_guess, 0.95)
            # apply NoiseNet model
            score = self.model.predict([patches, sd_guess])
            sd_hat = score[:, 0] * coeff_norm[:, 0]  # noise SD estimates
            sigma_sd = score[:, 1]  # prediction of relative SD noise SD estimates
            if np.sum(sigma_sd < self.sigma_rel_th) < 25:  # not enough patches with certain noise SD estimates
                # is_textured = True
                if self.type_nlf == NoiseModel.ADDITIVE_POISSON:
                    nlf_hat = np.zeros((2,))
                elif self.type_nlf == NoiseModel.ADDITIVE:
                    nlf_hat = np.zeros((1,))
                break

            if self.type_nlf == NoiseModel.ADDITIVE_POISSON:
                # refine noise nlf estimate
                var_hat = sd_hat ** 2
                var_sd = 2. * sigma_sd
                nlf_hat = self.__estimate_add_poiss(patch_mean_intensity[:, 0], var_hat, var_sd, self.intensity_max)
                if not nlf_hat.any():
                    break
            else:
                print("Noise NLF is not supported")
                return np.zeros((1,))

            # update predicted value of noise SD for each patch
            sd_pred_new = np.sqrt(np.maximum(np.polyval(nlf_hat, patch_mean_intensity), self.sd_min))
            sd_pred = 0.2 * sd_pred + 0.8 * sd_pred_new

        return nlf_hat

    def __estimate_add_poiss(self, x, y, sd_rel_y, intensity_level_max):
        """ Estimate signal-independent and signal-dependent noise variance components

        :param x: patch intensity
        :param y: patch variance
        :param sd_rel_y: relative SD of patch variance estimates
        :param intensity_level_max: maximum intensity level
        :return: nlf estimate
        """
        n_certain_required = 10  # number of required certain estimates
        var_y = sd_rel_y ** 2  # relative SD of patch variance estimates
        # select certain estimates
        idx_to_use = np.where(sd_rel_y < 2. * self.sigma_rel_th)
        if idx_to_use[0].size < n_certain_required:  # not enough certain estimates
            return np.zeros((2, ))
        for i in range(5):  # iterations for robust line fitting

            # perform weighted least square nlf estimation
            X = x[idx_to_use]
            X = sm.add_constant(X)
            mod_wls = sm.WLS(y[idx_to_use], X, weights=1. / var_y[idx_to_use])
            res_wls = mod_wls.fit()
            nlf_hat = np.array(res_wls.params[::-1])

            # predict noise variance for each patch using current nlf estimate
            y_pred = np.polyval(nlf_hat, x)
            y_pred = np.maximum(y_pred, self.sd_min)
            sd_hat = np.sqrt(y_pred)
            # update var_y
            var_y = (sd_rel_y * y_pred) ** 2

            # detect patches affected by clipping effect
            to_use = np.logical_and(x - 5. * sd_hat > 0., x + 5. * sd_hat < intensity_level_max)
            # detect outliers
            r = abs(y / y_pred - 1.0) / sd_rel_y
            to_use = np.logical_and(to_use, r < 3.0)
            # detect uncertain estimates
            to_use = np.logical_and(to_use, sd_rel_y < 2. * self.sigma_rel_th)
            idx_to_use = np.where(to_use)
            if idx_to_use[0].size < n_certain_required:  # not enough certain estimates
                return np.zeros((2, ))

        return nlf_hat

    @staticmethod
    def draw_uniform_patches(img, n_samples, patch_size, n_bin, intensity_level_max):
        """ find n_samples patches of size patch_size by patch_size pixels
            with minimum variance and unifomely covering image intensity range

        :param img: image to process
        :param n_samples: required number of patches
        :param patch_size: size of patch
        :param n_bin: number of intensity bins
        :return: patches row and column indexes
        """
        step = 9
        # calculate ing block mean and variance
        kernel = np.ones((patch_size, 1), np.float32) / patch_size
        mean_img = cv2.sepFilter2D(img, -1, kernel, kernel)
        mean2_img = cv2.sepFilter2D(img**2, -1, kernel, kernel)
        var_img = mean2_img - mean_img**2
        kernel = np.ones((step, 1), np.float32) / step
        mean_img = cv2.sepFilter2D(img, -1, kernel, kernel)  # local mean in the patch center
        # prepare patches grid
        raw_pos = np.arange(patch_size, img.shape[0] - patch_size, step)
        col_pos = np.arange(patch_size, img.shape[1] - patch_size, step)
        x, y = np.meshgrid(raw_pos, col_pos)
        # calculate mean and variance of each patch
        var = np.ravel(var_img[x, y])
        mean = np.ravel(mean_img[x, y])
        # calculate patches intensity range
        intensity_range_min = np.percentile(mean, 1)
        intensity_range_max = np.percentile(mean, 99)
        intensity_range_max = min(intensity_range_max, intensity_level_max)
        intensity_range = (intensity_range_max - intensity_range_min) / n_bin
        # select patches
        n_samples_per_bin_available = np.zeros(n_bin)
        for i in range(n_bin):
            # in each intensity bin select fixed number of patches with minimum variance
            intensity = intensity_range_min + intensity_range * i
            idx = np.where(np.logical_and(mean > intensity - intensity_range / 2.,
                                          mean < intensity + intensity_range / 2.))[0]
            n_samples_per_bin_available[i] = idx.size
        n_samples_available = np.sum(n_samples_per_bin_available)

        pos_raw = []
        pos_col = []
        # t = np.arange(n_bin)
        # n_samples_per_bin = 0.1 * (2. * t / (n_bin - 1.0) - 1.0) ** 2 + 0.9
        # n_samples_per_bin = 0.0 * (1.0 - t / (n_bin - 1.0)) + 1.0
        n_samples_per_bin = np.ones((n_bin,))
        n_samples_per_bin *= n_samples / np.sum(n_samples_per_bin)
        n_samples_per_bin = np.minimum(n_samples_per_bin, n_samples_per_bin_available)
        while np.sum(n_samples_per_bin) < min(n_samples_available, n_samples) - 1.:
            n_samples_per_bin *= 1.1
            n_samples_per_bin = np.minimum(n_samples_per_bin, n_samples_per_bin_available)
        n_samples_per_bin = n_samples_per_bin.astype("int")

        for i in range(n_bin):
            # in each intensity bin select fixed number of patches with minimum variance
            intensity = intensity_range_min + intensity_range * i
            idx = np.where(np.logical_and(mean > intensity - intensity_range / 2.,
                                          mean < intensity + intensity_range / 2.))[0]
            idx_min_var = np.argsort(var[idx])
            idx = idx[idx_min_var[0:min(n_samples_per_bin[i], idx_min_var.size)]]
            pos_raw.extend(np.ravel(x)[idx])
            pos_col.extend(np.ravel(y)[idx])

        return pos_raw, pos_col
