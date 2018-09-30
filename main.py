__author__ = "Mykhail L. Uss"
__copyright__ = "Copyright 2018, Mykhail L. Uss"
__credits__ = ["Mykhail L. Uss"]
__license__ = "GPLv3"
__version__ = "1.0.1"
__maintainer__ = "Mykhail L. Uss"
__email__ = "mykhail.uss@gmail.com", "uss@xai.edu.ua"
__status__ = "Prototype"

import argparse
import os
import sys
import shutil
from evaluate import Evaluator
from keras.optimizers import Adam
from model import noise_net_model, u_l2, m_uncert, err_rel, err_norm_sd, bias, SIGMA_EPSILON
from utils import collect_train_data
from keras.callbacks import TensorBoard, ModelCheckpoint, Callback
import tensorflow as tf
from keras import backend as K
from keras.models import load_model
import numpy as np

# DEFAULTS
SIGMA_REL_TH = 0.20  # threshold on relative noise SD estimation error
BATCH_SIZE = 32  # default batch size value
EPOCHS_MAX = 250  # maximum number of epochs


class LrCallback(Callback):

    """ Callback class for logging Learning rate

    """

    def on_epoch_end(self, epoch, logs={}):
        lr = K.eval(self.model.optimizer.lr)
        decay = K.eval(self.model.optimizer.decay)
        iterations = K.eval(self.model.optimizer.iterations)
        lr_with_decay = lr / (1. + decay * iterations)
        logs["Learning rate"] = np.array(lr_with_decay, dtype="float64")


class NED2012Callback(Callback):

    """ Callback class for (optional) logging NED2012 evaluation results

    """

    def __init__(self, NED2012_path):
        self.NED2012_path = NED2012_path
        self.loss = []

    def on_epoch_end(self, epoch, logs={}):
        self.loss.append(logs["val_loss"])
        if len(self.loss) == 1 or min(self.loss[:-1]) > self.loss[-1]:
            ev = Evaluator(self.model, SIGMA_REL_TH)
            ev.process_NED2012(self.NED2012_path)
            ev.print_res()
            sd_SD, sd_SI = ev.get_res()
            logs["NED2012 SD component error"] = np.array(sd_SD, dtype="float64")
            logs["NED2012 SI component error"] = np.array(sd_SI, dtype="float64")

class TrainValTensorBoard(TensorBoard):
    def __init__(self, log_dir='./logs', **kwargs):
        # Make the original `TensorBoard` log to a subdirectory 'training'
        training_log_dir = os.path.join(log_dir, 'training')
        super(TrainValTensorBoard, self).__init__(training_log_dir, **kwargs)

        # Log the validation metrics to a separate subdirectory
        self.val_log_dir = os.path.join(log_dir, 'validation')

    def set_model(self, model):
        # Setup writer for validation metrics
        self.val_writer = tf.summary.FileWriter(self.val_log_dir)
        super(TrainValTensorBoard, self).set_model(model)

    def on_epoch_end(self, epoch, logs=None):
        # Pop the validation logs and handle them separately with
        # `self.val_writer`. Also rename the keys so that they can
        # be plotted on the same figure with the training metrics
        logs = logs or {}
        val_logs = {k.replace('val_', ''): v for k, v in logs.items() if k.startswith('val_')}
        for name, value in val_logs.items():
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.val_writer.add_summary(summary, epoch)

        self.val_writer.flush()

        # Pass the remaining logs to `TensorBoard.on_epoch_end`
        logs = {k: v for k, v in logs.items() if not k.startswith('val_')}
        super(TrainValTensorBoard, self).on_epoch_end(epoch, logs)

    def on_train_end(self, logs=None):
        super(TrainValTensorBoard, self).on_train_end(logs)
        self.val_writer.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Train/evaluate NoiseNet model.')
    parser.add_argument('mode', type=str, help='train/eval modes.')
    parser.add_argument('model_path', type=str, help='path to NoiseNet model.')
    parser.add_argument('--NED2012_path', type=str, default=None, help='path to NED2012 database.')
    parser.add_argument('--train_db', type=str, default=None, help='path to train database.')
    parser.add_argument('--lr', type=float, default=0.000005, help='learning rate.')
    args = parser.parse_args()
    model_path = args.model_path

    if args.mode == 'train':
        print("NoiseNet training mode")

        if args.train_db is None:
            print("Train database path not specified, aborting training process.")
            sys.exit(0)

        # initialize model
        model = noise_net_model()
        img_size = int(model.get_layer("input_1").get_output_at(0).get_shape()[2])  # get input patch size
        model.name = os.path.basename(model_path)

        optim = Adam(lr=args.lr, beta_1=0.9, beta_2=0.999, decay=1e-6)
        model.compile(loss=u_l2,
                      optimizer=optim,
                      metrics=[m_uncert, err_rel, err_norm_sd, bias])

        X, Y = collect_train_data(args.train_db, img_size)

        if not os.path.exists('./Graph'):
            os.makedirs('./Graph')
        else:
            for root, dirs, files in os.walk('./Graph'):
                for f in files:
                    os.unlink(os.path.join(root, f))
                for d in dirs:
                    shutil.rmtree(os.path.join(root, d))

        tb_call_back = TrainValTensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)

        if args.NED2012_path is None:
            callbacks = [LrCallback(), tb_call_back,
                         ModelCheckpoint(model_path, verbose=1, save_best_only=True)]
        else:
            callbacks = [LrCallback(), NED2012Callback(args.NED2012_path), tb_call_back,
                         ModelCheckpoint(model_path, verbose=1, save_best_only=True)]

        model.fit([X, Y], Y,
                  batch_size=BATCH_SIZE,
                  epochs=EPOCHS_MAX,
                  validation_split=0.2,
                  verbose=2,
                  callbacks=callbacks,
                  )
        print("NoiseNet training completed. Path to model: %s" % model_path)

    elif args.mode == 'eval':
        print("NoiseNet evaluation mode.")
        print("Evaluating model %s" % model_path)
        try:
            netBNPE = load_model(model_path, custom_objects={"SIGMA_EPSILON": SIGMA_EPSILON, "test_loss": u_l2,
                                                             "u_l2": u_l2, "m_uncert": m_uncert,
                                                             "bias": bias, "err_rel": err_rel,
                                                             "err_norm_sd": err_norm_sd})
            print("Model %s successfully loaded" % model_path)
        except:
            print("Failed to load model %s" % model_path)
            sys.exit(0)

        stats_path = model_path + '_stats.xls'
        ev = Evaluator(netBNPE, SIGMA_REL_TH, 1, stats_path)
        # perform test on pure gaussian noise
        ev.pure_noise_test()
        if args.NED2012_path is None:
            print("NED2012_path not specified. Skipping NED2012 evaluation stage")
        else:
            # perform test on NED2012 database
            ev.process_NED2012(args.NED2012_path)
    else:
        Warning("Unsupported mode")