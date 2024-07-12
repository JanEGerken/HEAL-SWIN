# HEAL-SWIN: A Vision Transformer On The Sphere

<img align="right" height="270ex" src="./assets/heal_swin_vs_swin.svg">

This repository contains the reference implementation of the HEAL-SWIN model[^1] for semantic segmentation of high-dimensional spherical images. HEAL-SWIN extends a UNet-variant[^2] of the Hierarchical Shifted-Window (SWIN) transformer[^3] to the spherical Hierarchical Equal Area iso-Latitude Pixelation (HEALPix) grid[^4]. For handling the HEALPix grid efficiently in Python, we use the healpy package[^5].

We provide a PyTorch model for HEAL-SWIN and the baseline SWIN-UNet as well as a training and evaluation environment based on PyTorch Lightning. The dataloaders, models and training environments provided are suitable for semantic segmentation and depth estimation on the WoodScape[^6] and SynWoodScape[^7] datasets. For logging, we use MLflow[^8] with either a filesystem backend or a database backend.

## Compute Environment

The code assumes that the root directory `HEAL-SWIN` is in the Python path. This can be achieved e.g. by running Python from the root directory instead of a subdirectory.

A number of directories outside the `heal_swin` directory can be moved to other places. These are

- `datasets`: The directory for the training data
- `mlruns`: The directory which is used to save logging information during training and evaluation results with MLflow
- `containers`: The directory in which the singularity container is saved
- `slurm`: The directory where MLflow looks for the slurm output file to be logged with the run
- `matplotlib_cache`: A cache directory for matplotlib

The paths to these directories are read from the file `compute_environment/current_environment.py`. Here, the name of the singularity container file can also be set. In the absence of this file the defaults from `compute_environment/local_environment.py` will be used which uses subdirectories of `HEAL-SWIN`. It is convenient to make `current_environment.py` a symlink to the existing environments.

## Build container

The Python packages required to run the code are specified in `containers/requirements.txt`. To build the singularity/apptainer container with which the code was tested, run `python3 run.py build-singularity`. The container will be saved according to the specifications in the compute environment. To build the docker container `heal_swin` from the requirements file, run `python3 run.py build-docker`.

## Install dependencies using pip

Although it is recommended to run the code inside the singularity/apptainer container which was used for development as described above, we also provide a `setup.py` to install the necessary dependencies using pip. To do this, execute

    pip3 install --upgrade pip
    pip3 install -e .[test,formatting,dev]
	
in the root directory. This will require Python 3.8 which can e.g. be installed using Conda. By installing in editable mode (`-e`), this will also add the root directory to the Python path as required. In order to compute the Chamfer distance for the depth estimation task, PyTorch3D in version 0.7.2 is required and has to be installed separately via

    FORCE_CUDA=1 pip3 install git+https://github.com/facebookresearch/pytorch3d.git@3145dd4d16edaceb394838364b8e87a440f83c10
	
Further details on the installation of PyTorch3D can be found [here](https://github.com/facebookresearch/pytorch3d/blob/3145dd4d16edaceb394838364b8e87a440f83c10/INSTALL.md). 

After installation, you can then use the `local` environment of the `run.py` script as detailed below. Note that in order to use the database backend of MLflow for logging, you additionally need to install SQLite.

## Data

The training and evaluation scripts in this repository require the WoodScapes dataset[^6] for semantic segmentation and/or the SynWoodScapes dataset[^7] for semantic segmentation and depth estimation. As of 2023-06-20, the WoodScapes dataset is available [here](https://drive.google.com/drive/folders/1X5JOMEfVlaXfdNy24P8VA-jMs0yzf_HR), while the SynWoodScapes dataset is available [here](https://drive.google.com/drive/folders/1N5rrySiw1uh9kLeBuOblMbXJ09YsqO7I). Both datasets should be saved in subdirectories `woodscape` and `synwoodscape`, respectively, of the `datasets` directory specified in the compute environment.

Using the `synwoodscape_merge_classes.py` script in `heal_swin/data/segmentation`, it is possible to generate copies of the semantic segmentation part of the SynWoodScapes dataset with merged classes. The calibration data and ground truth images are added to the new dataset via symlinks to the original `synwoodscape` directory. Together, all subdirectories of the `datasets` directory form "woodscape versions" and the data to be used can be selected by setting the corresponding option to the name of the desired subdirectory.

For the flat evaluation masked according to the utilized base pixels of the HEALPix grid, lists with samples corresponding to different camera calibrations are needed. This metadata as well as class color legends can be generated using the script `generate_metadata.py` in `heal_swin/data/segmentation.py`. The script `data_stats.py` in `heal_swin/data/segmentation` can be used to compute class prevalences useful for determining class weights in the cross-entropy loss.

To train a HEAL-SWIN model, the flat data needs to be projected onto the HEALPix grid. If the projected data is not available, the projection is initiated automatically at the beginning of training.


## Run code

The central entry point of the code is the `run.py` Python script. It is called as

    run.py [--env {local, singularity, docker}] <task> [task-specific arguments]

The optional `--env` argument specifies the environment in which the task is executed:
- `local` (default): no container, useful e.g. if you work inside a virtual Python environment
- `singularity`: the singularity container specified in the compute environment
- `docker`: the docker container `heal_swin`
  
If a task is run inside a container, then the necessary directories specified in the compute environment are bound and the root directory is added to the Python path.

Available tasks are:
- `build-singularity` to build the singularity/apptainer container (see above)
- `build-docker` to build the docker container (see above)
- `bash` to run an interactive bash session inside a container
- `train` to train a network
- `resume` to resume a training run
- `evaluate` to evaluate a trained network
- `start-mlflow-server` to start an HTTP MLflow server for logging
- `test-repo` to run integration tests (trains multiple small networks and evaluates them)
- `format-code` to run the code formatter

The `train`, `resume` and `evaluate` tasks just run the respective scripts in `heal_swin` inside the specified environment, they are documented there. The task-specific arguments are in these cases the arguments of the respective scripts.

The `start-mlflow-server` task accepts the following arguments:
- `--backend`, default `sqlite`: can be either `sqlite` or `filesystem` and specifies whether an SQLite server should be used for logging or the filesystem (much slower)
- `--port`, default `5000`: port at which the web interface is accessible
- `--workers`, default `1`: number of workers which access the database
- `--timeout`, default `30`: database access timeout in seconds before the server crashes

The `test-repo` task starts pytest and runs the tests inside `heal_swin/testing`. It accepts all pytest arguments.

## Reference

HEAL-SWIN was introduced in

O. Carlsson, J. E. Gerken, H. Linander, H. Spieß, F. Ohlsson, C. Petersson, and D. Persson, “HEAL-SWIN: A Vision Transformer On The Sphere,” 2023. [arXiv:2307.07313](http://arxiv.org/abs/2307.07313)

If you use this code, please cite
```
@inproceedings{carlsson2023,
  title = {{{HEAL-SWIN}}: {{A Vision Transformer On The Sphere}}},
  shorttitle = {{{HEAL-SWIN}}},
  booktitle = {Proceedings of the {{IEEE}}/{{CVF Conference}} on {{Computer Vision}} and {{Pattern Recognition}}},
  author = {Carlsson, Oscar and Gerken, Jan E. and Linander, Hampus and Spie{\ss}, Heiner and Ohlsson, Fredrik and Petersson, Christoffer and Persson, Daniel},
  year = {2023},
  month = jul,
  eprint = {2307.07313},
  primaryclass = {cs},
  pages = {6067--6077},
  urldate = {2024-07-12},
  archiveprefix = {arXiv},
  langid = {english},
  keywords = {Computer Science - Computer Vision and Pattern Recognition,Computer Science - Machine Learning}
}

```


[^1]: O. Carlsson, J. E. Gerken, H. Linander, H. Spieß, F. Ohlsson, C. Petersson, and D. Persson, “HEAL-SWIN: A Vision Transformer On The Sphere,” 2023. [arXiv:2307.07313](http://arxiv.org/abs/2307.07313).

[^2]: H. Cao, Y. Wang, J. Chen, D. Jiang, X. Zhang, Q. Tian, and M. Wang, “Swin-Unet: Unet-like pure transformer for medical image segmentation,” in Computer Vision – ECCV 2022 Workshops. ECCV 2022, L. Karlinsky, T. Michaeli, and K. Nishino, eds., vol. 13803 of Lecture Notes in Computer Science,
pp. 205–218. Springer International Publishing, 2022. [arXiv:2105.05537](https://arxiv.org/abs/2105.05537).

[^3]: Z. Liu, Y. Lin, Y. Cao, H. Hu, Y. Wei, Z. Zhang, S. Lin, and B. Guo, “Swin transformer: Hierarchical vision transformer using shifted windows,” in Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pp. 9992–10002. IEEE, 2021. [arXiv:2103.14030](https://arxiv.org/abs/2103.14030).

[^4]: K. M. Gorski, E. Hivon, and B. D. Wandelt, “Analysis issues for large CMB data sets,” [arXiv:astro-ph/9812350](https://arxiv.org/abs/astro-ph/9812350).

[^5]: A. Zonca, L. Singer, D. Lenz, M. Reinecke, C. Rosset, E. Hivon, and K. Gorski, “healpy: equal area pixelization and spherical harmonics transforms for data on the sphere in Python,” Journal of Open Source Software 4 (2019) 1298. [github:healpy/healpy](https://github.com/healpy/healpy)

[^6]: S. Yogamani, C. Hughes, J. Horgan, G. Sistu, S. Chennupati, M. Uricar, S. Milz, M. Simon, K. Amende, C. Witt, H. Rashed, S. Nayak, S. Mansoor, P. Varley, X. Perrotton, D. Odea, and P. Perez, “Woodscape: A multi-task, multi-camera fisheye dataset for autonomous driving,” in Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pp. 9307–9317. IEEE, 2019. [arXiv:1905.01489](https://arxiv.org/abs/1905.01489).

[^7]: A. R. Sekkat, Y. Dupuis, V. R. Kumar, H. Rashed, S. Yogamani, P. Vasseur, and P. Honeine, “SynWoodScape: Synthetic surround-view fisheye camera dataset for autonomous driving,” IEEE Robotics and Automation Letters 7 (2022) 8502–8509, [arXiv:2203.05056](https://arxiv.org/abs/2203.05056)

[^8]: [MLflow](https://mlflow.org)
