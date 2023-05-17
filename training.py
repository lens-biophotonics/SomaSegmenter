# =============================================================================
# TRAINING SCRIPT
#
# last major rev. 2020/04
#
# Filippo Maria Castelli
# LENS Biophotonics Group
# castelli@lens.unifi.it
# =============================================================================

import pickle
import logging
from pathlib import Path
from argparse import ArgumentParser

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau
import git

from data_generators import build_dataset
from utils import LoadArgsFromFile
from models import neuroSegUNET

# Import appropriate metrics for for keras custom objects load
from metrics import jaccard_index
from metrics import dice_coefficient
from RunDescriptor import RunDescriptor
from tp2d import TiledPredictor
from performance_evaluation import PerformanceMetrics
from utils import load_volume, save_volume

# Enabling memory growth for all GPUs
gpu_list = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpu_list:
    tf.config.experimental.set_memory_growth(gpu, True)

# Setting logger level
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


def main():
    # parsing options
    parser = ArgumentParser()

    parser.add_argument(
        "-d",
        "--dataset",
        action="store",
        type=str,
        dest="dataset_path_str",
        default="/mnt/ssd1/em_dataset",
        help="Training dataset path",
    )

    parser.add_argument(
        "-o",
        "--out",
        action="store",
        type=str,
        dest="out_path_str",
        default="/mnt/ssd1/out",
        help="Output path",
    )

    parser.add_argument(
        "-e",
        "--epochs",
        action="store",
        dest="epochs",
        type=int,
        default=10,
        help="Number of training epochs",
    )

    parser.add_argument(
        "-b",
        "--batch",
        action="store",
        dest="batch_size",
        type=int,
        default=50,
        help="Batch size",
    )

    parser.add_argument(
        "--ilr",
        action="store",
        dest="initial_learning_rate",
        type=float,
        default=0.00067,
        help="Initial learning rate",
    )

    parser.add_argument(
        "--cropshape",
        action="store",
        dest="crop_shape",
        type=tuple,
        default=(128, 128),
        help="Crop window shape",
    )

    parser.add_argument(
        "--descriptorpath",
        action="store",
        type=str,
        dest="descriptor_path",
        help="Path for descriptor file",
    )

    parser.add_argument(
        "--notes",
        action="store",
        default="",
        type=str,
        dest="notes",
        help="Notes for descriptor"
    )

    parser.add_argument(
        "--basefilters",
        action="store",
        type=int,
        default=16,
        dest="base_filters",
        help="Base filters",
    )

    parser.add_argument(
        "--depth",
        action="store",
        dest="depth",
        type=int,
        default=3,
        help="Model depth",
    )
    parser.add_argument(
        "--transpconv",
        action="store_true",
        dest="transposed_convolution",
        help="Use transposed_conv instead of resize-conv",
    )
    
    parser.add_argument(
        "--noisestd",
        action="store",
        dest="noise_std",
        type=float,
        default=0.01,
        help="Max standard deviation for data augmentation noise",
    )

    parser.add_argument(
        "--brightrange",
        action="store",
        dest="brightness_range",
        type=float,
        default=0.01,
        help="Range around 1.0 for data augmentation brigtness scaling.",
    )

    parser.add_argument(
        "--gammarange",
        action="store",
        dest="gamma_range",
        type=float,
        default=0.09,
        help="Range around 1.0 for data augmentation gamma transform.",
    )
    
    parser.add_argument(
        "--prediction_batch_size",
        action="store",
        type=int,
        default=60,
        dest="prediction_batch_size",
        help="batchsize in performance evaluation test step"
    )

    # Loading arguments from file
    parser.add_argument("--file", type=open, action=LoadArgsFromFile)
    args = parser.parse_args()

    # Defining paths
    #
    # the dataset directory must contain the following files:
    #
    # train_frames.tif
    # train_masks.tif
    # val_frames.tif
    # val_masks.tif
    # test_frames.tif
    # test_masks.tif
    
    dataset_path = Path(args.dataset_path_str)

    train_frames_path = dataset_path.joinpath("train_frames.tif")
    train_masks_path = dataset_path.joinpath("train_masks.tif")

    val_frames_path = dataset_path.joinpath("val_frames.tif")
    val_masks_path = dataset_path.joinpath("val_masks.tif")

    test_frames_path = dataset_path.joinpath("test_frames.tif")
    test_masks_path = dataset_path.joinpath("test_masks.tif")

    out_path = Path(args.out_path_str)
    tmp_path = out_path.joinpath("tmp")
    
    descriptor_path = (
        Path(args.descriptor_path) if args.descriptor_path is not None else out_path
    )

    # generating output directories
    out_path.mkdir(exist_ok=True, parents=True)
    tmp_path.mkdir(exist_ok=True, parents=True)

    # output files paths
    model_history_path = out_path.joinpath("model_history.pickle")
    final_model_path = out_path.joinpath("final_model.hdf5")
    csv_summary_path = out_path.joinpath("run_summary.csv")

    # logs paths
    logs_path = out_path.joinpath("logs")
    logs_path.mkdir(exist_ok=True, parents=True)
    logfile_path = logs_path.joinpath("run_log.log")
    
    fh = logging.FileHandler(str(logfile_path))
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)  


    # data flows
    data_augmentation_settings = {
        "p_rotate": 0.5,
        "p_flip": 0.5,
        "p_brightness_scale": 0.3,
        "p_gaussian_noise": 0.5,
        "p_gamma_transform": 0.1,
        "brightness_scale_range": args.brightness_range,
        "gaussian_noise_std_max": args.noise_std,
        "gamma_range": args.gamma_range,
    }

    training_dataset = build_dataset(
        frame_path=train_frames_path,
        labels_path=train_masks_path,
        crop_shape=args.crop_shape,
        batch_size=args.batch_size,
        data_augmentation=True,
        augment_settings=data_augmentation_settings,
    )

    validation_dataset = build_dataset(
        frame_path=val_frames_path,
        labels_path=val_masks_path,
        crop_shape=args.crop_shape,
        batch_size=args.batch_size,
        data_augmentation=False,
    )
    
    # Keras callbacks definition
    checkpoint_callback = ModelCheckpoint(
        filepath=str(out_path) + "/weights.{epoch:02d}.hdf5",
        monitor="val_loss",
        verbose=1,
        save_best_only=False,
        mode="min",
    )
    
    csv_logger_callback = CSVLogger(
        filename=str(csv_summary_path),
        append=True, separator=";"
    )
    
    reduce_lr_callback = ReduceLROnPlateau(
        factor=0.1,
        patience=5,
        min_lr=0.00000000067,  # TODO check learning rate values
        verbose=1,
    )

    callback_list = [
        checkpoint_callback,
        csv_logger_callback,
        reduce_lr_callback
    ]

    # model definition
    throwaway_iterator = validation_dataset.__iter__()
    throwaway_batch = next(throwaway_iterator)
    
    input_shape = throwaway_batch[0].shape.as_list()[1:]
    
    #input_shape = (*args.crop_shape, 1)
    model = neuroSegUNET(
        input_shape=input_shape,
        base_filters=args.base_filters,
        batch_normalization=True,
        pre_activation=True,
        depth=args.depth,
        transposed_convolution=args.transposed_convolution,
    )

    # Metrics definition
    metrics = ["accuracy", jaccard_index, dice_coefficient]

    # optimizer and loss definition
    model.compile(
        optimizer=Adam(lr=args.initial_learning_rate),
        loss="binary_crossentropy",
        metrics=metrics,
    )

    # Model fitting
    model_history = model.fit(
        x=training_dataset,
        validation_data=validation_dataset,
        epochs=args.epochs,
        steps_per_epoch=training_dataset.steps_per_epoch,
        validation_steps=validation_dataset.steps_per_epoch,
        callbacks=callback_list,
        max_queue_size=100,
        shuffle=False,
    )

    # pickling model history
    with model_history_path.open(mode="wb") as hist_file:
        pickle.dump(model_history.history, hist_file)

    # saving model
    model.save(str(final_model_path))
    
    
    test_frames = load_volume(test_frames_path, expand_dims=False)
    test_masks = load_volume(test_masks_path, expand_dims=False)
    
    test_predictor = TiledPredictor(input_volume=test_frames,
                                       batch_size=args.prediction_batch_size,
                                       window_size=args.crop_shape,
                                       num_rotations=0,
                                       tmp_path=tmp_path,
                                       model=model)
    
    test_prediction = test_predictor.out_volume[:,:,:,0]
    
    save_volume(test_prediction,
                out_path=out_path,
                filename="test_predictions")
    
    metrics = PerformanceMetrics(y_true=test_masks,
                                     y_pred=test_prediction,
                                     thr=0.5)
    
    metrics_dict = metrics.measure_dict

    # RunDescriptor
    descr = RunDescriptor(
        descriptor_dir_path=descriptor_path,
        entry_type="training",
        model_type="2d",
        model_path=final_model_path,
        dataset_path=dataset_path,
        epochs=args.epochs,
        model_history_dict=model_history.history,
        git_repo=git.Repo(".."),
        callback_list=callback_list,
        script_options=vars(args),
        log_dir_path=logs_path,
        notes=args.notes,
        performance_metrics_dict=metrics_dict
    )


if __name__ == "__main__":
    main()
