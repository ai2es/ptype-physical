# Ptype-Physical

This package contains code for generating machine learning
estimates of winter precipitation type from vertical profiles of the atmosphere.

## Installation
1. Clone the repository from github `git clone git@github.com:ai2es/ptype-physical.git`
2. Within the terminal, go to the top-level directory with `cd ptype-physical`.
3. Install miniconda, then create a ptype environment with the following command: `conda env create -f environment.yml`.
4. Alternatively, add the dependencies to an existing environment with `conda env update -f environment.yml`.
5. Activate the environment by running `conda activate ptype` or `source activate ptype`.
6. Install the package directly: `pip install .` or link the package if you are debugging `pip install -e .`


## Train a classifier model to predict ptypes

Train a multi-layer perceptron model via
```bash
python applications/train_mlp.py -c config/ptype.yml
```

Upon completion, the model will save predictions on training, validation and test splits to file. 
Next compute metrics on the splits and case studies via

```bash
python applications/evaluate_mlp.py -c config/ptype.yml
```

Two types of approaches are supported: (1) A standard MLP trained with cross-entropy, and (2) evidential approach trained with Dirichlet loss.

* Option (1) may be selected by setting the loss to "categorical_crossentropy" and the output_activation to "softmax" in ptype.yml.

* Option (2) may be selected by setting the loss to "dirichlet" and the output_activation to "linear" in ptype.yml.

For more details, 

```bash
python applications/train_mlp.py --help
python applications/evaluate_mlp.py --help
```


## Active training 

Active training can be called with Option (1) via
```bash
python applications/active_training.py -c config/ptype.yml -i 20 -p "random"
```
which will perform 20 iterations with a random policy. 
The number of cross-validation steps at each iteration is controlled using "n_splits" in the config file.

One may also select "mc-dropout" for the policy, and use the parser option -s to set the number of Monte Carlo iterations:
```bash
python applications/active_training.py -c config/ptype.yml -i 20 -p "mc-dropout" -s 100
```

For option (2), the supported policies are 
```bash
"evidential", "aleatoric", "epistemic", "random", "mc-dropout"
```


The script can also be run on multiple GPU nodes for either option via
```bash
python applications/active_training.py -c config/ptype.yml -i 20 -p "random" -l 1 -n 20
```
which will launch 20 PBS jobs to GPU nodes. 

Once all iterations have completed, the results are currently viewed using notebooks/compare_active_training.ipynb

For more details, 

```bash
python applications/active_training.py --help
```

## Config file