# Uncertainty Quantification for Regression using Proper Scoring Rules

Alexander Fishkov<sup>1*</sup>, Kajetan Schweighofer<sup>2*</sup>, Mykyta Ielanskyi<sup>2</sup>, Nikita Kotelevskii<sup>1</sup>, Mohsen Guizani<sup>1</sup>, Maxim Panov<sup>1</sup>

$^1$ Department of Machine Learning, MBZUAI, UAE
$^2$ ELLIS Unit, LIT AI Lab, Institute for Machine Learning, JKU Linz, Austria   
$^*$ Equal contribution  

## Summary

In this work, we extend recent advances on uncertainty estimation in classification to the regression setting.
Under parameteric assumptions, we derive closed-form expressions for regression uncertainty measures and MC estimators thereof.
Our broad evaluation on synthetic and real-world regression datasets provides guidance for selecting reliable uncertainty measures for the task at hand.
Implementations of all considered uncertainty measures can be found here: [uncertainty_measures.py](./source/utils/uncertainty_measures.py).

## Installation

```
conda env create -f environment.yml
```

Important are python=3.12, pytorch=2.4 and torchvision=0.19 for reproducability.

## Setting up the datasets

For the toy tasks dots and arrow, execute the notebooks [notebooks/check_dots.ipynb](./notebooks/check_dots.ipynb) and [notebooks/check_arrows.ipynb](./notebooks/check_arrows.ipynb).
They both contain calls to the ```generate_and_save_data``` function of the specific task, which creates the datasets necessary for training and testing.
For cityscapes, follow the instructions in [datasets/README.md](datasets/README.md) first and then execute [datasets/generate_cityscapes.ipynb](datasets/generate_cityscapes.ipynb) to preprocess the dataset for training.

## Training and Evaluation

To train models, call [train.py](train.py).
Example configs can be seen in [run_experiment.sh](run_experiment.sh).

After these runs are finished, one needs to obtain probs on the test dataset for individual ensemble models and save the test targets for convinience.
This is done by calling [evaluate.py](evaluate.py).
All configs can be seen in [run_evaluation.sh](run_evaluation.sh).
Similarly for active learning, which is run by calling [active_learning.py](active_learning.py).
The final results for individual experiments are obtained throught the ```analyze_*``` and ```generate_*``` notebooks.

To play around with the difference between the usual parameterization of the Gaussian likelihood and the natural parametrization, there is the toy sine example in [toy_regression.ipynb](notebooks/toy_regression.ipynb).

