# =============================================================================
# PREDICTION SCRIPT
#
# last major rev. 2020/04
#
# Filippo Maria Castelli
# LENS Biophotonics Group
# castelli@lens.unifi.it
# =============================================================================
from argparse import ArgumentParser
from pathlib import Path
import logging

import git
from tensorflow.python.keras.models import load_model

from tp2d import TiledPredictor
from metrics import jaccard_index
from metrics import dice_coefficient
from utils import LoadArgsFromFile
from utils import load_volume
from utils import save_volume
from RunDescriptor import RunDescriptor
from performance_evaluation import PerformanceMetrics
# Setting logger level
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


def main():
    parser = ArgumentParser()

    parser.add_argument(
        "-i",
        "--img",
        action="store",
        type=str,
        dest="img_path_str",
        default="test_frames.tif",
        help="Input images path",
    )

    parser.add_argument(
        "-o",
        "--output",
        action="store",
        type=str,
        dest="out_path_str",
        default="out",
        help="Output path",
    )

    parser.add_argument(
        "-m",
        "--model",
        action="store",
        type=str,
        dest="model_path_str",
        default="out/model.hdf5",
        help="Model location",
    )

    parser.add_argument(
        "-g",
        "--groundtruth",
        action="store",
        type=str,
        dest="gt_path_str",
        default=None,
        help="Ground truth path, optional",
    )

    parser.add_argument(
        "--thr",
        action="store",
        type=float,
        dest="threshold",
        default=0.5,
        help="Threshold for crisp performance metrics evaluation, optional",
    )

    parser.add_argument(
        "-b",
        "--batch_size",
        action="store",
        type=int,
        dest="batch_size",
        default=60,
        help="Batch size",
    )

    parser.add_argument(
        "-t",
        "--temp",
        action="store",
        type=str,
        dest="tmp_path_str",
        default="tmp",
        help="temp directory path, optional",
    )

    parser.add_argument(
        "--descriptorpath",
        action="store",
        type=str,
        dest="descriptor_path",
        help="RunDescriptor file path",
    )

    parser.add_argument(
        "--notes",
        action="store",
        type=str,
        dest="notes",
        help="Notes for RunDescriptor"
    )

    # Loading args from file
    parser.add_argument("--file", type=open, action=LoadArgsFromFile)

    # Option parsing
    args = parser.parse_args()

    # Path definitions
    model_path = Path(args.model_path_str)
    img_path = Path(args.img_path_str)
    out_path = Path(args.out_path_str)
    tmp_path = Path(args.tmp_path_str)
    gt_path = Path(args.gt_path_str) if args.gt_path_str is not None else None
    
    # Output directories
    out_path.mkdir(exist_ok=True, parents=True)

    if args.descriptor_path is not None:
        descriptor_path = Path(args.descriptor_path)
    else:
        descriptor_path = out_path

    # Logs directory
    logs_path = out_path.joinpath("logs")
    logs_path.mkdir(exist_ok=True, parents=True)
    logfile_path = logs_path.joinpath("logging_log.log")
    fh = logging.FileHandler(str(logfile_path))
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)

    # Custom metrics
    custom_objects = {
        "jaccard_index": jaccard_index,
        "dice_coefficient": dice_coefficient
        }
    
    keras_model = load_model(str(model_path),
                             custom_objects=custom_objects)
    
    # Data loading
    in_volume = load_volume(img_path, expand_dims=False)

    # Prediction and Reconstruction
    predictions = TiledPredictor(
        input_volume=in_volume,
        batch_size=args.batch_size,
        tmp_path=tmp_path,
        num_rotations=0,
        model=keras_model
    )

    # Saving results
    save_volume(
        volume=predictions.out_volume[:,:,:,0],
        out_path=out_path,
        filename=img_path.stem,
    )

    # Calculating performances on test dataset
    performance_metrics_dict = calculate_metrics(
        predictions.out_volume, gt_path=gt_path, classification_threshold=args.threshold
    )
    
    # Run descriptor output
    RunDescriptor(
        descriptor_dir_path=descriptor_path,
        entry_type="predict",
        model_type="2d",
        model_path=model_path,
        log_dir_path=logs_path,
        input_data_path=img_path,
        script_options=vars(args),
        git_repo=git.Repo(".."),
        notes=args.notes,
        predictions_path=out_path,
        ground_truth_path=gt_path,
        performance_metrics_dict=performance_metrics_dict,
    )

def calculate_metrics(prediction_volume,
                      gt_path,
                      classification_threshold=0.5):
    """helper function for metric calculation"""
    if gt_path is not None:
        logging.info(
            "calculating performance metrics against test dataset {}".format(
                str(gt_path)
            )
        )
        ground_truth = load_volume(gt_path)
        metrics = PerformanceMetrics(
            y_true=ground_truth,
            y_pred=prediction_volume[:, :, :, 0],
            thr=classification_threshold,
        )
        performance_metrics_dict = metrics.measure_dict
    else:
        performance_metrics_dict = None

    return performance_metrics_dict


if __name__ == "__main__":
    main()
