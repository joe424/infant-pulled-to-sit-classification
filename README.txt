## build environment
    conda create --name <env_name> python=3.6
    conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2     cudatoolkit=10.2 -c pytorch # CUDA 10.2
    pip install opencv-python
    conda install -c conda-forge tqdm
    conda install -c anaconda numpy
    conda install -c anaconda scipy
    pip install imgaug
    pip install pycocotools
    conda install -c conda-forge tabulate
    conda install -c conda-forge termcolor
    conda install -c anaconda click
    conda install -c conda-forge motmetrics
    conda install -c conda-forge yacs
    cd 2D_pose_estimation/lighttrack/graph/torchlight/; python setup.py install
    conda install -c anaconda h5py
    conda install -c anaconda scikit-learn

## download data
weight data and dataset can be get at 
https://drive.google.com/drive/folders/1e3ySedgC0dx4d6g9DAYxSouz3g1H4SHY?usp=sharing
and then
    sh move_data.sh

## train classification model
    cd classification; python train.py

## classify pulled-to-sit level
put videos under 2D_pose_estimation/videos/pull_to_sit/
and then
    python run.py