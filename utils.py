# =============================================================================
# UTILS
#
# last major rev. 2020/04
#
# Filippo Maria Castelli
# LENS Biophotonics Group
# castelli@lens.unifi.it
# =============================================================================

import argparse
import pickle

import numpy as np
from skimage import io as skio
import tifffile

# Arg load from file
class LoadArgsFromFile(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        with values as f:
            parser.parse_args(f.read().split(), namespace)


# Load a volume from file
def load_volume(img_path,
                normalize=True,
                expand_dims=False):
    """
    load a volume image with scikit-image io imread
    opt normalize and expand last dim
    """
    in_volume = skio.imread(img_path, plugin="pil")
    if normalize:
        in_volume = in_volume / 255
    if expand_dims:
        in_volume = np.expand_dims(in_volume, axis=-1)

    return in_volume

def save_volume(
    volume,
    out_path,
    filename="predictions",
    save_tiff=True,
    save_pickle=True,
):
    if save_pickle:
        final_result_pickle_path = out_path.joinpath(filename + ".pickle")

        # As a pickle
        with final_result_pickle_path.open(mode="wb") as out_file:
            pickle.dump(volume, out_file)

    if save_tiff:
        # As a compressed multipage tiff
        tiff_path = out_path.joinpath(filename + ".tiff")
        # save_tiff(predictions.out_volume, tiff_path)
        tifffile.imwrite(tiff_path, volume.astype(np.float32))