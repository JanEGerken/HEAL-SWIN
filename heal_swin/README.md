# Running HEAL-SWIN

HEAL-SWIN runs are managed using config files. Examples are given in the `run_configs` directory. These files can either be used as arguments to `train.py`, `resume.py` and `evaluate.py`, as detailed below, or run as python scripts directly, starting a run with the corresponding configuration, e.g.

    python3 default_train_run_config.py

The config files furthermore double as slurm jobscripts, so the corresponding run can also be launched by calling `sbatch` directly on a config file, e.g. 

    sbatch default_train_run_config.py
	
Note, however, that a slurm environment is not necessary to execute the code.

## Training

Training is performed by the `train.py` script. It has one mandatory argument, `--config_path`, which specifies the path to the training config file relative to the root directory. For example, to start training with default parameters, run

    python3 train.py --config_path=heal_swin/run_configs/default_train_run_config.py
	
Note that this requires the root directory `HEAL-SWIN` to be in the python path. So you could e.g. run this inside an interactive bash session in a container launched by `python3 run.py --env singularity bash` (which takes care of setting the python path) or you could run it from the root directory by

    python3 heal_swin/train.py --config_path=heal_swin/run_configs/default_train_run_config.py
	
or you could use `run.py` to start the training inside the container directly without an interactive session by running

    python3 run.py --env singularity train --config_path=heal_swin/run_configs/default_train_run_config.py
	
in the root directory.
	
### Training config files

Training config files are python modules which need to contain two functions which are called without any arguments by the training script to retrieve the configuration:

- `get_train_run_config` should return an instance of the dataclass `SingleModelTrainRun` defined in `training/train_config.py`
- `get_pl_config` should return an instance of the dataclass `PLConfig` also defined in `training/train_config.py`

The `SingleModelTrainRun` dataclass contains options for the training, data and model to be used. The `PLConfig` dataclass encapsulates the PyTorch Lightning options which are used for training. For more details, please refer to `training/train_config.py` directly.

The `run_configs` directory contains the following training config files

- `default_train_run_config.py`: Traines with default parameters, i.e. on healpix-projected data with a HEAL-SWIN model
- `segmentation/swin_XX_train_run_config.py`: Traines a SWIN transformer on the flat segmentation masks in the dataset XX
- `segmentation/swin_hp_XX_train_run_config.py`: Traines a HEAL-SWIN transformer on the spherical segmentation masks in the dataset XX
- `depth_estimation/depth_swin_synwoodscap_train_run_config.py`: Traines a SWIN transformer on flat SynWoodScapes fisheye depth maps
- `depth_estimation/depth_swin_hp_synwoodscape_train_run_config.py`: Traines a HEAL-SWIN transformer on spherical SynWoodScape fisheye depth maps

For further details on how to reproduce the runs in the paper, see the section below.

## Resuming

A training run can be resumed from checkpoint using the `resume.py` script. As `train.py`, it also has one mandatory argument, `--config_path`, which specifies the path to the resume config file relative to the root directory. For example, to resume the run specified in `run_configs/default_resume_config.py`, run

    python3 resume.py --config_path=heal_swin/run_configs/default_resume_config.py
	
as with `train.py` this can be run e.g. inside an interactive container session or using

    python3 run.py --env singularity resume --config_path=heal_swin/run_configs/default_resume_config.py

### Resume config files

As training config files, resume config files are also python modules which need to contain two functions which are called without any arguments by the resume script to retrieve the configuration:

- `get_resume_run_config` should return an instance of the dataclass `ResumeConfig` defined in `training/train_config.py`
- `get_pl_config` should return an instance of the dataclass `PLConfig` also defined in `training/train_config.py`

The `ResumeConfig` dataclass contains options for the resumed run like the path to the checkpoint or an MLflow run id and epoch specification as well as a training configuration in the form of a `SingleModelTrainRun` instance. The `PLConfig` dataclass encapsulates the PyTorch Lightning options which are used for training. Only the `resume_from_checkpoint` option of Lightning is overwritten with the path to the checkpoint specified in `ResumeConfig`. For more details, please refer to `training/train_config.py` directly.

The configuration in `run_configs/default_resume_config.py` loads the `SingleModelTrainRun` and `PLConfig` instances from the saved checkpoint and is therefore suited to resume crashed runs without specifying any training parameters again. Additionally, it reads the MLflow run id from the `RUN_ID` environment variable, so a run can be resumed conveniently via

    export RUN_ID=<MLFLow run id>; python3 run.py --env singularity resume --config_path=heal_swin/run_configs/default_resume_config.py
	
from the root directory or as a slurm job via

    export RUN_ID=<MLFLow run id>; sbatch default_resume_config.py

from inside `heal_swin/run_configs`. Note however that this also loads the maximum number of epochs from the checkpoint, so if you want to extend the training of a finished run, you need to modify the config file.

## Evaluation

Evaluation of a trained model is performed as a separate run, although a simple evaluation is performed at the end of training if the `eval_after_train` flag in the `TrainConfig` of `SingleModelTrainRun` is set to `True`.

The `evaluate.py` starts an evaluation run from a config file whose path relative to the root directory is specified by the one mandatory argument, `--config_path`. For example, to evaluate the run specified in `run_configs/default_evaluate_config.py`, run

    python3 evaluate.py --config_path=heal_swin/run_configs/default_evaluate_run_config.py
	
as above this can be run e.g. inside an interactive container session or using

    python3 run.py --env singularity evaluate --config_path=heal_swin/run_configs/default_evaluate_run_config.py

### Evaluation config files

Evaluation config files are python modules which need to contain two functions which are called without any arguments by the evaluation script to retrieve the configuration:

- `get_eval_run_config` should return an instance of the dataclass `EvaluateConfig` defined in `evaluation/evaluate_config.py`
- `get_pl_config` should return an instance of the dataclass `PLConfig` also defined in `training/train_config.py`

The `EvaluateConfig` dataclass contains options for the checkpoint to be evaluated, the specific evaluations to run, the training configuration used and the data on which to evaluate. In particular, it allows for the specification of a prediction writer which are used for more elaborate evaluations, as e.g. on masked and projected versions of the data. A list of available segmentation prediction writers can be found in the `get_pred_writer` functions of the flat data module in `data/segmentation/flat_datamodule.py` and the HEALPix data module in `data/segmentation/hp_datasets.py` and correspondingly for prediction writers for depth estimation. For more details, please refer to `evaluation/evaluate_config.py` and the data modules directly. The `PLConfig` dataclass encapsulates the PyTorch Lightning options which are used for the evaluation run.

The evaluation config files in `run_configs` all load the training, data and model configuration as well as the `PLConfig` instance from the saved checkpoint. They perform the evaluation on the best logged epoch and read the MLflow run id from the `RUN_ID` environment varibale. The evaluation configuration files available are

- `default_evaluate_run_config.py`: Performes an evaluation with default parameters, similar to the one performed after training when `eval_after_train` is set to `True`
- `segmentation/evaluate_all_config.py`: Performes a number of different evaluations for segmentation depending on the `SLURM_ARRAY_TASK_ID` environment variable. This file is desigend to be run as a slurm array job to run all implemented evaluations for semantic segmentation on a praticular checkpoint in parallel. For an evaluation of a flat model, task ids 0-4 are admissible, for a HEALPix model, task ids 0-5 are admissible.
- `depth_estimation/evaluate_all_depth_config.py`: Similar to `evaluate_all_config.py`, but evaluates depth estimation checkpoints. For an evaluation of a flat model, task ids 0-4 are admissible, for a HEALPix model, task ids 0-5 are admissible.

For instance, in order to launch an array job to perform all evaluations on a HEAL-SWIN checkpoint with a segmentation model, run

    export RUN_ID=<MLFLow run id>; sbatch -a 0-5 evaluate_all_config.py

inside `run_config`.

## Reproducing runs from the paper
In order to reproduce the numbers reported in the paper (XXX add arXiv link XXX), use the following commands.

### Segmentation
The config files for semantic segmentation on various datasets are available in `run_configs/segmentation`. As described above, SWIN runs are configured in `swin_XX_train_run_config` and HEAL-SWIN runs are configured in `swin_hp_XX_train_run_config.py`. The dataset `XX` can be one of

- `woodscape` for the WoodScape dataset with the ten classes `void`, `road`, `lanemarks`, `curb`, `person`, `rider`, `vehicles`, `bicycle`, `motorcycle` and `traffic_sign`
- `synwoodscape_large` for a selection of the dominant classes `void`, `building`, `road line`, `road`, `sidewalk`, `four-wheeler vehicle`, `sky` and `ego-vehicle` from the SynWoodScape dataset
- `synwoodscape_large_plus_AD` for a selection of the domaninant plus autonomous-driving related classes `void`, `building`, `pedestrian`, `road line`, `road`, `sidewalk`, `four-wheeler vehicle`, `traffic sign`, `sky`, `traffic light`, `two-wheeler vehicle` and `ego-vehicle` from the SynWoodScape dataset

After the training has completed, evaluate the segmentation runs with

    export RUN_ID=<MLFLow run id of SWIN run>; export SLURM_ARRAY_TASK_ID=3; python3 run.py --env singularity evaluate --config_path=heal_swin/run_configs/segmentation/evaluate_all_config.py
    export RUN_ID=<MLFLow run id of SWIN run>; export SLURM_ARRAY_TASK_ID=4; python3 run.py --env singularity evaluate --config_path=heal_swin/run_configs/segmentation/evaluate_all_config.py

and

    export RUN_ID=<MLFLow run id of HEAL-SWIN run>; export SLURM_ARRAY_TASK_ID=0; python3 run.py --env singularity evaluate --config_path=heal_swin/run_configs/segmentation/evaluate_all_config.py
    export RUN_ID=<MLFLow run id of HEAL-SWIN run>; export SLURM_ARRAY_TASK_ID=5; python3 run.py --env singularity evaluate --config_path=heal_swin/run_configs/segmentation/evaluate_all_config.py

or launch the evaluations as a slurm array job as described above.

The segmentation metrics which are reported in the paper are
- `val_hp_masked_iou` for the SWIN model evaluated on flat images
- `val_iou_projected_to_hp` for the SWIN model evaluated on the sphere
- `val_back_projected_hp_masked_iou_res_640_768` for the HEAL-SWIN model evaluated on flat images
- `val_iou_global` for the HEAL-SWIN model evaluated on the sphere

as evaluated on the best epoch.

### Depth Estimation

For depth estimation use 

    python3 run.py --env singularity train --config_path=heal_swin/run_configs/depth_swin_train_run_config.py

for the baseline using the SWIN-UNet model and

    python3 run.py --env singularity train --config_path=heal_swin/run_configs/depth_swin_hp_train_run_config.py

for depth estimation using the HEAL-SWIN model.

The depth estimation the metric reported in the paper is `best_chamfer_distance_full_res_hp_masked` which compares the point cloud created from the model's depth estimation to the point cloud created from the ground truth, with both masked to the area available to the HEAL-SWIN model. To evaluate the trained models with this metric, use
    
    export RUN_ID=<MLFLow run id>; export SLURM_ARRAY_TASK_ID=1; python3 run.py --env singularity evaluate --config_path=heal_swin/run_configs/depth_estimation/evaluate_all_depth_config.py

or launch the evaluations as a slurm array job as described above.
