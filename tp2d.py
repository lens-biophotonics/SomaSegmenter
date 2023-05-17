# =============================================================================
# TILED PREDICTOR
#
# last major rev. 2020/04
#
# Filippo Maria Castelli
# LENS Biophotonics Group
# castelli@lens.unifi.it
# =============================================================================

import pickle
import functools
import logging
from pathlib import Path
from itertools import product

import scipy.signal as signal
import numpy as np
import tensorflow as tf

# Enabling memory growth for all GPUs
gpu_list = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpu_list:
    tf.config.experimental.set_memory_growth(gpu, True)

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


class TiledPredictor:
    def __init__(
        self,
        input_volume,
        batch_size,
        window_size=(128, 128),
        num_rotations=3,
        tmp_path="temp",
        model=None,
    ):
        """
        TiledPredictor
        
        Patch-based reconstruction of large prediction frames.
        The volume 
        

        Parameters
        ----------
        input_volume : numpy array
            Input data.
        batch_size : int
            Batch size.
        window_size : tuple, optional
            (width, height) shape of the predictions. The default is (128, 128).
        num_rotations : TYPE, optional
            Number average rotations. The default is 3.
        tmp_path : TYPE, optional
            DESCRIPTION. The default is "temp".
        model : TYPE, optional
            DESCRIPTION. The default is None.
         : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        # Sanity checks
        assert len(input_volume.shape) == 4, "Data must be in z, y, x, channels format"

        self.tmp_path = Path(tmp_path)
        self.tmp_path.mkdir(exist_ok=True, parents=True)

        # Inputs
        self.input_volume = input_volume
        self.window_size = window_size
        self.batch_size = batch_size
        self.num_rotations = num_rotations

        # Other attributes
        self.rotated_views = []
        self.cached_windows = dict()
        
        self.model = model

        # Init sequence

        # Padding volume
        self.padded_original, self.padding = self._pad_volume(
            input_volume=self.input_volume
        )
        self.padded_output_shape = list(self.padded_original.shape[:-1]) + [
            1
        ]  # we're limiting to only 1 channel
        self.output_shape = list(self.input_volume.shape[:-1]) + [1]

        # Generate rotations
        self.rotations = self._gen_rotations(self.padded_original)

        # Generating output
        # > generate a prediction list
        # > maybe unpad and un-rotate inside _predict_view
        self.prediction_list = []
        self.out_volume = np.zeros(shape=self.output_shape, dtype="float")
        self._average_predicted_views()

    def _pad_volume(self, input_volume):
        """applies padding to the input volume"""
        assert self.window_size[0] % 2 == 0, "window size must be divisible by 2"
        aug_unit = self.window_size[0] // 2  # half window
        dims = input_volume.shape[:-1]  # z, y and x dimensions
        pads = np.array([[0, 0], [0, 0], [0, 0], [0, 0]])
        for i, dim in enumerate(dims):  # che cazzo ho scritto
            # padding aug_unit before and aug_unit after
            pads[i] = pads[i] + aug_unit  # [aug_unit, aug_unit]
            dim = (
                dim + pads[i].sum()
            )  # di fatto sto aumentando la dimensione di una quantità window_size
            # If we still have pixels to pad divide them between before and after
            r = dim % aug_unit
            pads[i][0] = pads[i][0] + r // 2
            pads[i][1] = pads[i][1] + r // 2 + r % 2
        # we set pads[0] = [0,0] as we don't need to pad z dimension (we're making predictions on slices)
        pads[0] = [0, 0]
        padded_volume = np.pad(input_volume, pad_width=pads, mode="reflect")
        return padded_volume, pads

    def _gen_rotations(self, padded_volume):
        rotation_list = []
        for i, rotation_number in enumerate(range(self.num_rotations + 1)):
            rotated_volume = np.rot90(
                m=np.array(padded_volume), k=rotation_number, axes=(1, 2)
            )
            rotation_id = i
            rotation_list.append((rotation_id, rotated_volume))
        return rotation_list  # is a list of (k_rotation, rotated_volume)

    def _predict_view(self, rotation):
        """For a given flip/rotation of the 3D input volume istantiates a SingleViewPredictor for patch-based
        prediction of that view """
        predictor = SingleViewPredictor(
            rotation=rotation,  # this is a (k_rotation, rotated_volume) tuple
            padding=self.padding,  # [ [pad_before, pad_after] ...] list
            batch_size=self.batch_size,
            window_size=self.window_size,
            tmp_path=self.tmp_path,
            model=self.model
        )
        predicted_view = predictor.predict_from_patches()

        self.prediction_list.append(predicted_view)
        return predicted_view

    def _average_predicted_views(self):
        """performs an average over all all views to obtain the final predictions"""
        for i, rotation in enumerate(self.rotations):
            logging.info(
                "Predicting on rotation {} out of {}".format(i, len(self.rotations))
            )
            current_view = self._predict_view(rotation)
            self.out_volume = self.out_volume + current_view
        self.out_volume = self.out_volume / len(self.rotations)

        return self.out_volume


class SingleViewPredictor:
    """Class for computing predictions on a single view"""

    def __init__(
        self,
        rotation,
        window_size,
        padding,
        batch_size,
        tmp_path="tmp",
        model=None,
    ):
        # Inputs
        self.rotated_volume = rotation[1]
        self.rotation_id = rotation[0]
        self.window_size = window_size
        self.padding = padding
        self.tmp_path = tmp_path
        self.batch_size = batch_size
        self.model = model

        # Other attributes
        self.window = self.weighting_window(window_size=self.window_size)
        self.batch_queue = []

        # Init sequence
        # Loading padded volume
        self.prediction_volume = np.zeros_like(self.rotated_volume).astype("float32")

    def weighting_window(self, window_size, power=2):
        """generates a 2D weighting window"""
        wind = self.spline_window(window_size, power)
        wind = np.expand_dims(wind, axis=-1)
        wind = wind * wind.transpose()
        wind = wind / wind.max()
        wind = np.expand_dims(wind, axis=-1)
        return wind.astype("float32")

    @staticmethod
    def spline_window(window_size, power=2):
        """ generates 1D spline window profile"""
        intersection = int(window_size[0] / 4)
        wind_outer = (abs(2 * (signal.triang(window_size[0]))) ** power) / 2
        wind_outer[intersection:-intersection] = 0

        wind_inner = 1 - (abs(2 * (signal.triang(window_size[0]) - 1)) ** power) / 2
        wind_inner[:intersection] = 0
        wind_inner[-intersection:] = 0

        wind = wind_inner + wind_outer
        wind = wind / np.average(wind)
        return wind

    def _get_pivot_points(self):
        """Generates a list of all possible pivot points for the 3D windowss"""
        padz_len, pady_len, padx_len = self.rotated_volume.shape[:-1]
        step = self.window_size[0] // 2
        x_points = range(0, padx_len - self.window_size[1] + 1, step)
        y_points = range(0, pady_len - self.window_size[0] + 1, step)
        z_points = range(0, padz_len)

        return list(
            product(z_points, y_points, x_points)
        )  # this can probably be done with a meshgrid

    def predict_from_patches(self):
        """Prediction + Reconstruction step"""
        # Define pivot points for the patches
        pivot_points = self._get_pivot_points()

        # Organize pivot points in chunks
        chunk_size = 100 * self.batch_size
        pivot_chunk_list = list(self._get_chunks(pivot_points, chunk_size))

        # Batch generator
        callable_patch_batch_generator = functools.partial(
            self._patch_batch_generator,
            padded_volume=self.rotated_volume,
            pivot_points=pivot_points,
            window_size=self.window_size,
            batch_size=self.batch_size,
        )
        second_generator = callable_patch_batch_generator()
        # List of paths for the saved prediction tensors
        prediction_pathlist = []
        predicted = 0
        # Predicting in batches and saving chunks of predicted data
        logging.info("Running predictions on patches")
        for i, chunk in enumerate(pivot_chunk_list):
            logging.debug(
                "chunk {} / {}, chunk length: {}".format(
                    i, len(pivot_chunk_list), len(chunk)
                )
            )
            logging.debug("already processed {}".format(predicted))
            n_steps = len(chunk) // self.batch_size
            if len(chunk) % self.batch_size != 0:
                n_steps = n_steps + 1  # Extra steps if there are spare chunks
            batch_list = []
            for i in range(n_steps):
                batch_list.append(next(second_generator))
            
            batch_data = np.vstack(batch_list)

            predictions = self.model.predict(batch_data)
            predicted += len(predictions)
            # debug_flag = False
            # if debug_flag: # Debug mode: I use constant output to see if reconstruction yields correct values
            #     predictions = np.zeros_like(predictions) + 0.5

            path = self.tmp_path.joinpath("predictions_{}.npy".format(i))
            np.save(path, predictions)
            prediction_pathlist.append(path)

        # Saving a reference to all predictions (debug-only)
        prediction_pathlist_pickle_path = self.tmp_path.joinpath("pred_pathlist.pickle")
        with prediction_pathlist_pickle_path.open(mode="wb") as wfile:
            pickle.dump(prediction_pathlist, wfile)

        # Reconstruction
        logging.info("Reconstructing frame")
        for i, chunk in enumerate(pivot_chunk_list):
            logging.info(
                "Reconstructing from chunk {} / {}".format(i, len(pivot_chunk_list))
            )
            path_to_chunk = prediction_pathlist[i]
            prediction_chunk = np.load(path_to_chunk)

            # Weighting the entire chunk
            # tiled_windows = np.repeat(np.expand_dims(self.window, axis=0),
            #                           axis=0,
            #                           repeats=prediction_chunk.shape[0])
            weighted_chunk = (
                prediction_chunk * self.window
            )  # broadcasting window weighting

            for j, pivot in enumerate(chunk):
                prediction = weighted_chunk[j]
                z, y, x = pivot
                self.prediction_volume[
                    z, y : y + self.window_size[0], x : x + self.window_size[1]
                ] += prediction

        logging.info("Normalization and back-transform")
        self._normalize()
        self._back_transform()

        return self.prediction_volume

    @staticmethod
    def _patch_batch_generator(padded_volume, pivot_points, window_size, batch_size=2):
        """Generator for keras predictions"""
        i = 0
        patch_list = []
        for pivot in pivot_points:
            z, y, x = pivot
            patch = padded_volume[z, y : y + window_size[0], x : x + window_size[1], :]
            patch_list.append(patch)
            i += 1
            if i == batch_size:
                i = 0
                batch = np.array(
                    patch_list
                )  # Yielding a batch in the form of a numpy array
                patch_list = []
                yield batch
            # Last batch should contain the remaining patches
        batch = np.array(patch_list, dtype="float32")
        patch_list = []
        yield batch

    @staticmethod
    def _get_chunks(chunklist, chunk_size):
        """Yield successive n-size chunks from chunklist"""

        for i in range(0, len(chunklist), chunk_size):
            yield chunklist[i : i + chunk_size]

    def _normalize(self, subdivisions=None):
        self.prediction_volume[
            self.prediction_volume > 1
        ] = 1  # Todo: find out why tf did I wrote this

    def _back_transform(self):
        """perform unrotation, unflipping and unpadding"""

        k = self.rotation_id
        # rot first, flip last
        self._unrot(k)
        self._unpad()

    def _unflip(self, flip_ax):
        """ flips image to original format"""
        self.prediction_volume = np.flip(self.prediction_volume, axis=flip_ax)

    def _unrot(self, k):
        """rotate image to original format"""
        self.prediction_volume = np.rot90(self.prediction_volume, k=-k, axes=(1, 2))

    def _unpad(self):
        """removes padding"""
        y_min, y_max = self.padding[1]
        x_min, x_max = self.padding[2]

        self.prediction_volume = self.prediction_volume[
            :, y_min:-y_max, x_min:-x_max, :
        ]
