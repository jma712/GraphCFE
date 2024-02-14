# NeurIPS22:  CLEAR: Generative Counterfactual Explanations on Graphs

Code for the NeurIPS 2022 paper [*CLEAR: Generative Counterfactual Explanations
on Graphs*.](https://openreview.net/pdf?id=YR-s5leIvh).

## Environment
```
Python 3.6
Pytorch 1.2.0
Scipy 1.3.1
Numpy 1.17.2
Torch-geometric 1.7.2
```

## Dataset
Datasets can be found in [link](https://drive.google.com/drive/folders/1krE4mow5FZnQr0Re0EKoUG6k0H35luQM?usp=drive_link).

## Run Experiment
### Step 1:  Training a graph prediction model
Train a graph prediction model (i.e., the model which needs explanation). The trained prediction models used in this paper can be directly loaded from ```./model_save/```.

If you want to train them from scratch, run the following command (here we use the dataset *imdb_m* as an example):
```
python train_pred.py  --dataset imdb_m --epochs 600 --lr 0.001
```
Or you can also use any other graph prediction models instead.

### Step 2: Generating counterfactual explanations
```
python main.py --dataset imdb_m --experiment_type train
```
Here, when ```experiment_type``` is set to *train* or *test*, the model CLEAR will be trained or loaded from a saved file. When it is set to *baseline*, you can run the random perturbation based baselines (INSERT, REMOVE, RANDOM) by setting ```baseline_type```.

### Refenrences
The code is the implementation of this paper:


[1] Jing Ma, Ruocheng Guo, Saumitra Mishra, Aidong Zhang, Jundong Li. CLEAR: Generative Counterfactual Explanations on Graphs. Neural Information Processing Systems (NeurIPS), 2022. 


