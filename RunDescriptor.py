# =============================================================================
# RUN DESCRIPTOR
#
# last major rev. 2020/04
#
# Filippo Maria Castelli
# LENS Biophotonics Group
# castelli@lens.unifi.it
# =============================================================================
import yaml
import datetime
import logging
import pickle
import shutil
import os
import zipfile
from pathlib import Path

import pandas as pd

class RunDescriptor:
    """
    Descriptor class for TF-Keras training / prediction runs.
    
    A descriptor can be specified by initializing the details of the train or
    prediction run. It automatically generates a .zip archive which can be
    loaded to initialize another RunDescriptor.

    Parameters
    ----------
    descriptor_dir_path : pathlib Path or str
        Path to save descriptors to.
    entry_type : str
        entry type descriptor ("train"/"predict").
    model_type : str
        model type descriptor ("2D"/"3D").
    model_path : pathlib Path or str
        Path to keras hdf5 model
    model_history_dict : dict
        Keras model history dictionary:
            
        >>> model = model.fit(...)
        >>> history = model.history
    input_data_path : pathlib Path or str
        Path to input data.
    log_dir_path : pathlib Path or str
        Path to logs directory.
    dataset_path : pathlib Path or str
        Path to training dataset.
    epochs : int
        Number of training epochs.
    git_repo : git.Repo
        Requires gitpython: repo descriptor.
        
        >>> import git
        >>> repo = git.Repo()
    callback_list : list
        list of Keras callbacks as passed to model.fit()
    script_options : dict
        ArgumentParser args
        
        >>> parser = ArgumentParser()
        >>> args = parser.parse_args()
    load_from_archive : pathlib Path or str
        Specify .zip archive path to load descriptor from path.
    notes : str
        Optional notes.
     : TYPE
        DESCRIPTION.
        
    Methods
    -------
    compile_descriptor()
        compile descriptor fields and generate a .zip descriptor file in
        descriptor_dir_path
    set_path_mode(mode)
        change every local path (inside descriptor directory) to either a
        "relative" or an "absolute" path
    create_archive()
        generates an archive assuming every descriptor is already specified.
        compile_descriptor() calls create_archive() internally, it's not 
        recommended to call this function alone.
    load_from_archive(path)
        loads an archive from a .zip path
    
    Attributes
    ----------
    callbacks : dict
        dictionary of serializable callback options if callback is supported
    callbacks_config_csv_path : path
        csv file of serialized callback options
    callbacks_config_pickle_path : path
        pickle file of serialized callback options
    callbacks_list : list
        list of Keras callback objects
    dataset_path : pathlib Path
        path to original dataset
    datetime : str
        datetime string
    descriptor_dir_path : pathlib Path
        path to descriptor output directory
    descriptor_name : str
        name of descriptor
    descriptor_series : pd.Series
        a pd.Series resume of the descriptor
    entry_type : str
        entry type, "train" or "predict"
    epochs : int
        number of epochs
    git_repo : git.Repo
        gitPython Repo object
    git_repo_dict : dict
        repository descriptor dict
    input_data_path : pathlib Path
        path to input data
    log_dir_local_path : pathlib Path
        path to local copy of log directory, stored in descriptor archive
    log_dir_path : pathlib Path
        path to original log directory
    model_type : str
        model type, "2D" or "3D"
    model_history_dict : dict
        Keras model history dictionary
    model_path : pathlib Path
        path to original model
    model_local_path : pathlib Path
        path to local copy of model, stored in descriptor archive
    model_history_csv_path : pathlib Path
        path to a csv of model history, stored in descriptor archive
    model_history_pickle_path : pathlib Path
        path to pickle of model history, stored in descriptor archive
    notes : str
        notes
    path_mode : str
        path format mode, either "relative" or "absolute"
    yaml_path : pathlib Path
        path to local .yml descriptor file, stored in descriptor archive
    script_options : dict
        script option dictionary
    zip_path : pathlib Path
        path to .zip archive

    """

    def __init__(
        self,
        descriptor_dir_path=None,
        entry_type=None,
        model_type=None,
        model_path=None,
        model_history_dict=None,
        input_data_path=None,
        log_dir_path=None,
        dataset_path=None,
        epochs=None,
        git_repo=None,
        callback_list=None,
        script_options=None,
        load_from_archive=None,
        notes=None,
        predictions_path=None,
        ground_truth_path=None,
        performance_metrics_dict=None,
    ):
        self.FIELD_DATETIME = "datetime"
        self.FIELD_ENTRY_TYPE = "entry_type"
        self.FIELD_MODEL_TYPE = "model_type"
        self.FIELD_MODEL_PATH = "model_path"
        self.FIELD_MODEL_LOCAL_PATH = "model_local_path"
        self.FIELD_MODEL_TYPE = "model_type"
        self.FIELD_MODEL_HISTORY_PICKLE_PATH = "model_history_pickle_path"
        self.FIELD_MODEL_HISTORY_CSV_PATH = "model_history_csv_path"
        self.FIELD_INPUT_DATA_PATH = "input_data_path"
        self.FIELD_LOG_DIR_PATH = "log_dir_path"
        self.FIELD_LOG_DIR_LOCAL_PATH = "log_dir_local_path"
        self.FIELD_DATASET_PATH = "dataset_path"
        self.FIELD_EPOCHS = "epochs"
        self.FIELD_GIT_REPO_DICT = "git_repo_dict"  # CHANGED REPO DICT
        self.FIELD_CALLBACKS = "callbacks"
        self.FIELD_CALLBACK_CONFIG_PICKLE_PATH = "callback_config_pickle_path"
        self.FIELD_CALLBACK_CONFIG_CSV_PATH = "callback_config_csv_path"
        self.FIELD_SCRIPT_OPTIONS = "script_options"
        self.FIELD_DESCRIPTOR_PATH = "descriptor_path"
        self.FIELD_DESCRIPTOR_DIR_PATH = "descriptor_dir_path"
        self.FIELD_NOTES = "notes"
        self.FIELD_PATH_MODE = "path_mode"
        self.FIELD_PREDICTIONS_PATH = "predictions_path"
        self.FIELD_GROUND_TRUTH_PATH = "ground_truth_path"
        self.FIELD_MODEL_PERFORMANCE_METRICS_PICKLE_PATH = (
            "model_performance_metrics_pickle_path"
        )

        self.CALLBACKS_NON_SERIALIZABLE_KEYS = {
            "ReduceLROnPlateau": [
                "validation_data",
                "model",
                "_chief_worker_only",
                "monitor_op",
            ],
            "ModelCheckpoint": [
                "validation_data",
                "model",
                "_chief_worker_only",
                "monitor_op",
            ],
            "CSVLogger": ["model", "csv_file"],
        }
        self.SUPPORTED_CALLBACKS = list(self.CALLBACKS_NON_SERIALIZABLE_KEYS.keys())

        if load_from_archive is None:
            self.entry_type = entry_type
            self.model_type = model_type
            self.model_path = self._string_to_path(model_path)
            self.model_history_dict = model_history_dict
            self.input_data_path = self._string_to_path(input_data_path)
            self.log_dir_path = self._string_to_path(log_dir_path)
            self.dataset_path = self._string_to_path(dataset_path)
            self.epochs = epochs
            self.git_repo = git_repo
            self.git_repo_dict = self._repo_dict()
            self.callback_list = callback_list
            self.callbacks = {}
            self.script_options = script_options
            self.descriptor_dir_path = (
                self._string_to_path(descriptor_dir_path)
                if descriptor_dir_path is not None
                else Path.cwd()
            )
            self.notes = notes
            self.path_mode = "absolute"
            self.predictions_path = self._string_to_path(predictions_path)
            self.ground_truth_path = self._string_to_path(ground_truth_path)
            self.model_performance_metrics_dict = performance_metrics_dict

            # Setting datetime
            now = datetime.datetime.now()
            self.datetime = now.strftime("%d/%m/%Y %H:%M:%S")

            # Setting output path
            now_str = now.strftime("_%d_%m_%Y-%H_%M")
            descriptor_name = str(self.entry_type) + str(self.model_type) + now_str
            self.descriptor_path = self.descriptor_dir_path.joinpath(descriptor_name)

            self.descriptor_name = self.descriptor_path.name
            self.descriptor_path.mkdir(parents=True, exist_ok=True)

            # Path for local model
            self.model_local_path = self.descriptor_path.joinpath(
                self.descriptor_name + "_local_model.hdf5"
            )

            # Paths for model history
            self.model_history_pickle_path = self.descriptor_path.joinpath(
                self.descriptor_name + "_model_history.pickle"
            )
            self.model_history_csv_path = self.descriptor_path.joinpath(
                self.descriptor_name + "_model_history.csv"
            )

            # Paths for callback configs
            self.callback_config_pickle_path = self.descriptor_path.joinpath(
                self.descriptor_name + "_callback_config.pickle"
            )
            self.callback_config_csv_path = self.descriptor_path.joinpath(
                self.descriptor_name + "_callback_config.csv"
            )

            # Paths for local log dir
            self.log_dir_local_path = self.descriptor_path.joinpath("logs")

            # Paths for model performance metrics
            self.model_performance_metrics_pickle_path = self.descriptor_path.joinpath(
                self.descriptor_name + "_model_performance_metrics.pickle"
            )

            # Configuration file path and zip path
            self.yaml_path = self.descriptor_path.joinpath(
                self.descriptor_name + ".yml"
            )
            self.zip_path = self.descriptor_path.parent  # Archive everything at the end

            self._compile_local_path_list()
            self.compile_descriptor()

        else:
            self.path_mode = "absolute"
            self.load_from_archive(load_from_archive)
            self._compile_local_path_list()
            self._create_dict()
            if self.entry_type != "predict":
                self.model_history_dict = self._load_from_pickle(
                    self.model_history_pickle_path
                )
            # Support for older predict descriptors with no performance_metrics_dict
            if self.model_performance_metrics_pickle_path is None:
                self.model_performance_metrics_pickle_path = self.descriptor_path.joinpath("_model_performance_metrics.pickle")
                
            self.model_performance_metrics_dict = self._load_from_pickle(
                self.model_performance_metrics_pickle_path
            )

        # Every local path should be an ABSOLUTE

        # Converting all paths to absolute format
        self.set_path_mode("absolute")
        self.descriptor_series = self._gen_series()

    def _compile_local_path_list(self):
        """compile dict of paths to local files in descriptor archive"""
        self._local_paths = {
            "model_local_path": self.model_local_path,
            "log_dir_local_path": self.log_dir_local_path,
            "model_history_pickle_path": self.model_history_pickle_path,
            "model_history_csv_path": self.model_history_csv_path,
            "callback_config_pickle_path": self.callback_config_pickle_path,
            "callback_config_csv_path": self.callback_config_csv_path,
            "yaml_path": self.yaml_path,
        }

    def compile_descriptor(self):
        """
        Compile descriptor field,
        create a local copy of the model, write a yaml descriptor file and 
        create a .zip archive
        """
        self._copy_model()
        self._copy_log_dir()
        self._serialize_callbacks()
        self._serialize_history()
        self._serialize_model_performance_metrics()
        self._create_dict()
        self._write_yaml()
        self.create_archive()

    def _convert_to_absolute_path(self, path):
        """Convert a relative path to an absolute one"""
        if not path.is_absolute():
            return self.descriptor_path.joinpath(path)
        else:
            return path

    def _convert_to_relative_path(self, path):
        """Convert an absolute path to a relative one"""
        if path.is_absolute():
            return path.relative_to(self.descriptor_path)
        else:
            return path

    def set_path_mode(self, mode="relative"):
        """Change path mode from "relative" to "absolute" and vice-versa."""
        local_paths = self._local_paths

        if mode == "relative" and self.path_mode == "absolute":
            self.path_mode = "relative"
            for attr_name, path in local_paths.items():
                path = self._convert_to_relative_path(path)
                self.__setattr__(attr_name, path)
        elif mode == "absolute" and self.path_mode == "relative":
            self.path_mode = "absolute"
            for attr_name, path in local_paths.items():
                path = self._convert_to_absolute_path(path)
                self.__setattr__(attr_name, path)

    @staticmethod
    def _path_to_string(path, relative_to_path=None):
        """Convert a pathlib Path to string"""
        if path is not None:
            if relative_to_path is not None:
                path = path.relative_to(relative_to_path)
            return str(path)
        else:
            return None

    @staticmethod
    def _string_to_path(in_string):
        """Convert a string to a pathlib Path"""
        if in_string is not None:
            return Path(in_string)
        else:
            return None

    @staticmethod
    def _list_to_string(str_list):
        """Convert a list of strings to a string"""
        return " ".join(map(str, str_list))

    @staticmethod
    def _copy_dict(dictionary):
        """Returns a deep copy of a dict"""
        return {key: value for key, value in dictionary.items()}

    def _create_dict(self):
        """Create a dict of properties to be exported in YAML"""
        current_path_mode = self.path_mode
        self.set_path_mode("relative")
        self.descriptor_dict = {
            self.FIELD_DATETIME: self.datetime,
            self.FIELD_ENTRY_TYPE: self.entry_type,
            self.FIELD_MODEL_TYPE: self.model_type,
            self.FIELD_CALLBACKS: self.callbacks,
            self.FIELD_EPOCHS: self.epochs,
            self.FIELD_SCRIPT_OPTIONS: self.script_options,
            self.FIELD_GIT_REPO_DICT: self.git_repo_dict,
            self.FIELD_DESCRIPTOR_PATH: self._path_to_string(self.descriptor_path),
            self.FIELD_DESCRIPTOR_DIR_PATH: self._path_to_string(
                self.descriptor_dir_path
            ),
            self.FIELD_MODEL_PATH: self._path_to_string(self.model_path),
            self.FIELD_MODEL_LOCAL_PATH: self._path_to_string(self.model_local_path),
            self.FIELD_DATASET_PATH: self._path_to_string(self.dataset_path),
            self.FIELD_INPUT_DATA_PATH: self._path_to_string(self.input_data_path),
            self.FIELD_CALLBACK_CONFIG_CSV_PATH: self._path_to_string(
                self.callback_config_csv_path
            ),
            self.FIELD_CALLBACK_CONFIG_PICKLE_PATH: self._path_to_string(
                self.callback_config_pickle_path
            ),
            self.FIELD_MODEL_HISTORY_CSV_PATH: self._path_to_string(
                self.model_history_csv_path
            ),
            self.FIELD_MODEL_HISTORY_PICKLE_PATH: self._path_to_string(
                self.model_history_pickle_path
            ),
            self.FIELD_LOG_DIR_PATH: self._path_to_string(self.log_dir_path),
            self.FIELD_LOG_DIR_LOCAL_PATH: self._path_to_string(
                self.log_dir_local_path
            ),
            self.FIELD_NOTES: self.notes,
            self.FIELD_PREDICTIONS_PATH: self._path_to_string(self.predictions_path),
            self.FIELD_MODEL_PERFORMANCE_METRICS_PICKLE_PATH: self._path_to_string(
                self.model_performance_metrics_pickle_path
            ),
        }
        self.set_path_mode(current_path_mode)

        # TEMP FIX: key naming error
        # in some older archives I compiled a "utracked_files" dict key which I
        # dont' want to support, changing it in "untracked_files" at read-time
        try:
            untracked_files = self.descriptor_dict[self.FIELD_GIT_REPO_DICT][
                "utracked_files"
            ]
            self.descriptor_dict[self.FIELD_GIT_REPO_DICT][
                "untracked_files"
            ] = untracked_files
            self.descriptor_dict[self.FIELD_GIT_REPO_DICT].pop("utracked_files")
        except KeyError:
            pass

        return self.descriptor_dict

    def _gen_series(self):
        descriptor_dict = self.descriptor_dict.copy()
        subdict = self.descriptor_dict.copy()
        subdict["callbacks"] = (
            tuple(subdict["callbacks"]) if subdict["callbacks"] is not None else []
        )
        subdict.pop("git_repo_dict")
        subdict.pop("script_options")
        series = pd.Series(subdict)

        git_repo_dict = descriptor_dict["git_repo_dict"]
        # git_repo_dict["untracked_files"] = self._list_to_string(git_repo_dict["untracked_files"])
        git_repo_dict["untracked_files"] = (
            tuple(git_repo_dict["untracked_files"])
            if git_repo_dict["untracked_files"] is not None
            else []
        )

        series = series.append(pd.Series(git_repo_dict))
        series = series.append(pd.Series(descriptor_dict["script_options"]))
        return series

    def _write_yaml(self, sort_keys=False):
        """Write the descriptor's yaml from descriptor_dict"""
        with self.yaml_path.open(mode="w") as out_file:
            yaml.dump(self.descriptor_dict, out_file, sort_keys=sort_keys)

    def _repo_dict(self):
        """Create a repo-descriptive dictionary"""
        repo = self.git_repo
        if repo is not None:
            repo_dict = {
                "working_dir": str(repo.working_dir),
                "active_branch": str(repo.active_branch),
                "commit": str(repo.commit()),
                "is_dirty": str(repo.is_dirty()),
                "untracked_files": repo.untracked_files,
            }
        else:
            repo_dict = None
        self.repo_dict = repo_dict
        return self.repo_dict

    @staticmethod
    def _dict_to_csv(dictionary, fpath, separator=";"):
        """Generate a CSV output file from a dictionary"""
        with fpath.open(mode="w") as csv_out:
            pd.DataFrame(dictionary).to_csv(csv_out, sep=separator)

    @staticmethod
    def _to_pickle(obj, fpath):
        """Save obj to a picklable file in fpath"""
        with fpath.open(mode="wb") as pickle_out:
            pickle.dump(obj, pickle_out)

    @staticmethod
    def _load_from_pickle(fpath):
        "Loads a pickled object"
        if fpath.is_file():
            with fpath.open(mode="rb") as input_file:
                obj = pickle.load(input_file)
            return obj
        else:
            logging.warning(str(fpath) + " is not a file")
            return None

    def _serialize_history(self):
        """Serialize model history both in CSV and in pickle"""
        if self.model_history_dict is not None:
            self._dict_to_csv(
                dictionary=self.model_history_dict, fpath=self.model_history_csv_path
            )
            self._to_pickle(
                obj=self.model_history_dict, fpath=self.model_history_pickle_path
            )
        else:
            self.model_history_pickle_path = None
            self.model_history_csv_path = None

    def _serialize_model_performance_metrics(self):
        """Serialize model performance metrics as pickle"""
        # Model Performance Metrics
        if self.model_performance_metrics_dict is not None:
            self._to_pickle(
                obj=self.model_performance_metrics_dict,
                fpath=self.model_performance_metrics_pickle_path,
            )
        else:
            self.model_performance_metrics_pickle_path = None

    def _get_serializable_callback_params_dict(self, callback):
        """Remove from the callback property dictionary only the keys that are
        known to be non-serializable."""
        callback_name = callback.__class__.__name__

        if callback_name not in self.SUPPORTED_CALLBACKS:
            logging.warning("callback {} is not supported".format(callback_name))
            return {}
        non_serializable_keys = self.CALLBACKS_NON_SERIALIZABLE_KEYS[callback_name]

        callback_dict = vars(callback)

        for key in non_serializable_keys:
            callback_dict.pop(key)

        return callback_dict

    def _serialize_callbacks(self):
        """Create a config dict using serializable parameters of callbacks"""
        if self.callback_list is not None:
            callback_nested_dict = {}
            for callback in self.callback_list:
                callback_name = callback.__class__.__name__
                callback_nested_dict[
                    callback_name
                ] = self._get_serializable_callback_params_dict(callback)

            self._to_pickle(
                obj=callback_nested_dict, fpath=self.callback_config_pickle_path
            )
            self._dict_to_csv(
                dictionary=callback_nested_dict, fpath=self.callback_config_csv_path
            )

            self.callbacks = [
                callback.__class__.__name__ for callback in self.callback_list
            ]
        else:
            self.callbacks = None
            self.callback_config_csv_path = None
            self.callback_config_pickle_path = None

    def _copy_model(self):
        """Create a local copy of the model"""
        if self.model_path is not None:
            shutil.copy(str(self.model_path), str(self.model_local_path))

    def _copy_log_dir(self):
        """Create a local copy of log directory"""
        if self.log_dir_path is not None:
            shutil.copytree(str(self.log_dir_path), str(self.log_dir_local_path))

    def create_archive(self):
        """Generate a zip archive"""
        zipfile_path = self.zip_path.joinpath(self.descriptor_name + ".zip")
        prev_path = Path.cwd()
        os.chdir(str(self.descriptor_path))
        with zipfile.ZipFile(
            str(zipfile_path), "w", zipfile.ZIP_DEFLATED, allowZip64=True
        ) as zf:

            for fpath in self.descriptor_path.rglob("*"):
                relative_path = fpath.relative_to(self.descriptor_path)
                zf.write(str(relative_path))

        zf.close()
        os.chdir(str(prev_path))

    def load_from_archive(self, archive_path=None):
        """Load descriptor attributes from a pre-existing .zip archive"""
        archive_path = self._string_to_path(archive_path)

        archive = zipfile.ZipFile(str(archive_path))
        if archive.testzip() is not None:
            logging.warning(
                "Archive may be corrupted, found invalid files {}".format(
                    archive.testzip()
                )
            )

        archive_namelist = archive.namelist()
        yaml_filenames = [fname for fname in archive_namelist if "yml" in fname]
        assert len(yaml_filenames) > 0, "There's no .yml file in {}".format(
            str(archive_path)
        )
        assert len(yaml_filenames) == 1, "Too many .yml files in {}".format(
            str(archive_path)
        )

        yaml_file = archive.read(yaml_filenames[0])
        yml_dict = yaml.load(yaml_file, Loader=yaml.FullLoader)

        missing_keys = []
        self.datetime = self._get_key(yml_dict, self.FIELD_DATETIME, missing_keys)
        self.entry_type = self._get_key(yml_dict, self.FIELD_ENTRY_TYPE, missing_keys)
        self.model_type = self._get_key(yml_dict, self.FIELD_MODEL_TYPE, missing_keys)
        self.callbacks = self._get_key(yml_dict, self.FIELD_CALLBACKS, missing_keys)
        self.epochs = self._get_key(yml_dict, self.FIELD_EPOCHS, missing_keys)
        self.script_options = self._get_key(
            yml_dict, self.FIELD_SCRIPT_OPTIONS, missing_keys
        )
        self.git_repo_dict = self._get_key(
            yml_dict, self.FIELD_GIT_REPO_DICT, missing_keys
        )
        self.descriptor_path = self._get_key(
            yml_dict, self.FIELD_DESCRIPTOR_PATH, missing_keys, to_path=True
        )
        self.descriptor_dir_path = self._get_key(
            yml_dict, self.FIELD_DESCRIPTOR_DIR_PATH, missing_keys, to_path=True
        )
        self.model_path = self._get_key(
            yml_dict, self.FIELD_MODEL_PATH, missing_keys, to_path=True
        )
        self.model_local_path = self._get_key(
            yml_dict, self.FIELD_MODEL_LOCAL_PATH, missing_keys, to_path=True
        )
        self.dataset_path = self._get_key(
            yml_dict, self.FIELD_DATASET_PATH, missing_keys, to_path=True
        )
        self.input_data_path = self._get_key(
            yml_dict, self.FIELD_INPUT_DATA_PATH, missing_keys, to_path=True
        )
        self.callback_config_csv_path = self._get_key(
            yml_dict, self.FIELD_CALLBACK_CONFIG_CSV_PATH, missing_keys, to_path=True
        )
        self.callback_config_pickle_path = self._get_key(
            yml_dict, self.FIELD_CALLBACK_CONFIG_PICKLE_PATH, missing_keys, to_path=True
        )
        self.model_history_csv_path = self._get_key(
            yml_dict, self.FIELD_MODEL_HISTORY_CSV_PATH, missing_keys, to_path=True
        )
        self.model_history_pickle_path = self._get_key(
            yml_dict, self.FIELD_MODEL_HISTORY_PICKLE_PATH, missing_keys, to_path=True
        )
        self.log_dir_path = self._get_key(
            yml_dict, self.FIELD_LOG_DIR_PATH, missing_keys, to_path=True
        )
        self.log_dir_local_path = self._get_key(
            yml_dict, self.FIELD_LOG_DIR_LOCAL_PATH, missing_keys, to_path=True
        )
        self.notes = self._get_key(yml_dict, self.FIELD_NOTES, missing_keys)
        self.predictions_path = self._get_key(
            yml_dict, self.FIELD_NOTES, missing_keys, to_path=True
        )
        self.model_performance_metrics_pickle_path = self._get_key(
            yml_dict,
            self.FIELD_MODEL_PERFORMANCE_METRICS_PICKLE_PATH,
            missing_keys,
            to_path=True,
        )

        if len(missing_keys) > 0:
            logging.warning(
                "Could not retrieve these keys form yaml file: {}".format(missing_keys)
            )

        self.yaml_path = self._string_to_path(yaml_filenames[0])
        self.zip_path = self._string_to_path(archive_path)
        self.descriptor_name = self.zip_path.stem
        self.descriptor_path = self.zip_path.parent.joinpath(self.descriptor_name)

        if self.entry_type != "predict":
            with zipfile.ZipFile(str(self.zip_path)) as file:
                file.extract(
                    self.model_history_pickle_path.name, path=self.descriptor_path
                )

    def _get_key(self, source_dict, key, notfoundlist, to_path=False):
        """Retrieve key from source_dict, append key to notfoundlist if key is missing,
        convert string to pathlib Path if to_path is True"""
        try:
            content = source_dict[key]
        except:
            notfoundlist.append(key)
            content = None
            pass
        if to_path:
            content = self._string_to_path(content)

        return content
