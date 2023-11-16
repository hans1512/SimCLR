# Setting up

The original setup used pipenv which seemed to work very poorly.
Instead we've opted to go with conda

To setup run

```
conda create -n SimCLR python=3.8
conda activate SimCLR
python -m pip install -r requirements.txt
# Change to a version matching your CUDA install
python -m pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
```

Once everything is installed you can run the project in the following order

```
python 0_train.py # Trains encoder with SimCLR 
python embedd_all.py # Embeds the dataset with the encoder and saves it in numpy format
python train_head.py # Trains a prediction head on the embedded representations
python predict.py # Runs predictions on the embedded representations, and prints the accuracy
```

# Using different datasets
To use any dataset other than CIFAR10 you must run the python files with the -d argument
The two other datasets supported are
```
-d MNIST
-d STL10    
```
Note: MNIST can only be run with the contrastive encoder model_mode parameter (set in runtime.yaml).
Additionally, the 1st channel must be changed from 3 to 1 since MNIST isn't RGB like the others

# Results
| Dataset | Model | Accuracy |
|---|---|---|
| CIFAR10 | Resnet50 | 56.3%  |
| MNIST | custom CNN | 95.4% |

We believe that the subpar CIFAR10 results are a result of not enough training and a too simple classification head.
The same classification head architecture was used on both datasets, and it is a simple network with one hidden layer