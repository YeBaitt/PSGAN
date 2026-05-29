# PSGAN

## Installation
### Requirements
* Python >= 3.6
* cuda == 10.2
* [Pytorch==1.5.1](https://pytorch.org/)
* [PyG: torch-geometric==1.5.0](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)

### Install
```
git clone https://github.com/YeBaitt/PSGAN.git
cd ./PSGAN
conda create -n psgan python=3.7
conda activate psgan
conda install pytorch==1.5.1 torchvision==0.6.1 cudatoolkit=10.2 -c pytorch
bash install.sh
```

# Usage

## Data Preparation
```
put your dataset under data/ directory with the same structure shown in the data/msl/

data
 |-msl
 | |-list.txt    # the feature names, one feature per line
 | |-train.csv   # training data
 | |-test.csv    # test data
 |-your_dataset
 | |-list.txt
 | |-train.csv
 | |-test.csv
 | ...

```

### Notices:
* You can request the original SWaT and WADI dataset from [iTrust](https://itrust.sutd.edu.sg/) or get my preprocessed datasets with FFT from [datasets](https://pan.baidu.com/s/1ssGF9FOlVfWWWjLTts4-9g?pwd=1037) 
* The first column in .csv will be regarded as index column. 
* The column sequence in .csv don't need to match the sequence in list.txt, we will rearrange the data columns according to the sequence in list.txt.
* test.csv should have a column named "attack" which contains ground truth label(0/1) of being attacked or not(0: normal, 1: attacked)


## Pretrained Checkpoints
* The pretrained checkpionts can be found in: [ckpt](https://pan.baidu.com/s/169iMlP_xnsmk7Bq90AGOlA?pwd=1037)
## Quick Start
```
bash run.sh [gpu_id] [dataset_name]
```

## Train
```
python main.py -comment "your comment"\
 -dataset [dataset_name]\
 -save_path_pattern  "[saved_path]"\
 -slide_stride 1\
 -slide_win 5\
 -batch 256\ 
 -epoch 50\ 
 -random_seed 5\ 
 -decay 0\ 
 -dim 128\ 
 -out_layer_num 10\ 
 -out_layer_inter_dim 64\ 
 -val_ratio 0.2\ 
 -report best\ 
 -topk 30\ 
 -slide_avg_win 3\ 
 -device cuda\ 
 -fusion_layer_num 10\
 -mode "train"

```

## Test

```
python main.py -comment "your comment"\
 -dataset [dataset_name]\
 -save_path_pattern  "[saved_path]"\
 -slide_stride 1\
 -slide_win 5\
 -batch 256\ 
 -epoch 50\ 
 -random_seed 5\ 
 -decay 0\ 
 -dim 128\ 
 -out_layer_num 10\ 
 -out_layer_inter_dim 64\ 
 -val_ratio 0.2\ 
 -report best\ 
 -topk 30\ 
 -slide_avg_win 3\ 
 -device cuda\ 
 -fusion_layer_num 10\
 -mode "test"\
 -load_model_path [path to ckpt]
```


