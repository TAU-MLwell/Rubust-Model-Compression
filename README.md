# Robust Model Compression Using Deep Hypotheses 

This repository contains the code for running the algorithms and experiments presented in the "Robust Model Compression Using Deep Hypotheses" paper.

## Requirements
It is required to install Pytorch.<br/> 
To install requirements:

```setup
pip install -r requirements.txt
```

# Running compression algorithm (CREMBO)
Running the CREMBO algorithm is done by using the CREMBO class in ```robust_compression.py```. The CREMBO class is initialized with the following: <br/>
  * create_model: Function that returns the small model (m). ```create_model(args) -> torch.nn.Module/sklearn model``` 
  * train_hypothesis: Training function for training small model (m). ```train_hypothesis(args, model, M, train_dataloader, test_dataloader, device) -> torch.nn.Module/sklearn         model```. Training function for pytorch models must use ```allowed_labels_loss``` as the loss function (see ```examples.py```). 
  * eval_model: Function for evaluating model on a validation set. ```eval_model(model, val_dataloader, device) -> float```
  * args: Optional arguments for create_model and train_hypothesis methods. 
  * delta: Step size in CREMBO

<br/>
After the initialization of the CREMBO class, running the CREMBO algorithm is simply done by calling it with the large model M, training, test and validation loaders (DataLoader for PyTorch models or (x,y) tuples for sklearn models) and device to run on (cpu/cuda).

```
 # initiate CREMBO class
 crembo = CREMBO(create_model_func, train_hypothesis_func, eval_model_func, args)

 # run crembo
 f = crembo(M, train_loader, test_loader, valid_loader, device)
```
In ```examples.py``` there are detailed examples on how to run CREMBO on DNN to DNN compression using Pytorch models (including training functions) and how to run CREMBO on sklearn models as well.

# Running experiments
## Compressing to Interpretable Models
To run the experiments for compressing to interpretable models presented in the paper, use the 'run.py' file and provide the required parameters:<br/>
-model rf/nn      describing which model you wish to compress<br/>
-experiment_name  generalization/robustness describing which experiment to run<br/>
-dataset_name heart/dermatology/arrhythmia/breast_cancer/iris describing which dataset to use<br/>

There are also optional parameters<br/>
-n_experiments (default 20) the number of experiments to run<br/>
-tree_depth (default 4) the depth of the compressed trees<br/>
-delta (default 1) the size of the interval in CREMBO algorithm<br/>
-rf_depth (default 12) the depth of trees in random forest<br/>
-rf_trees (default 100) number of trees in random forest<br/>

To run the experiments, use the 'run.py', for example::

```run
python run.py -model rf -experiment_name generalization -dataset_name heart
```

After the experiment is completed, the results will be printed to the console.

Datasets:<br/>
Datasets are provided here. They are taken from the UCI repository  
Heart (cleveland: https://archive.ics.uci.edu/ml/datasets/Heart+Disease  
Dermatology: https://archive.ics.uci.edu/ml/datasets/Dermatology  
Arrythmia: https://archive.ics.uci.edu/ml/datasets/Arrhythmia  
Breast_cancer: https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29  
Iris: https://archive.ics.uci.edu/ml/datasets/Iris  

## DNN to DNN Compression
To run the experiments for DNN to DNN compression presented in the paper, use the 'dnn2dnn_compression.py' file and provide the required parameters:<br/>
-dataset_name - Only cifar10 is currently supported.<br/>
-experiment - Type of experiment: train/compress (CREMBO)/kd<br/>
-experiment_name -  Name of experiment (used for saving models)<br/> 
-model_name - The type of model to train for 'train' experiments or the type of model to compress to for 'compress' and 'kd' experiments. (lenet/mobilenetv2/resnet18/VGG16) <br/>
<br/>
For 'compress' and 'kd' experiments you also need to provide
-large_model, the type of large model used for compression. Either resnet18 or VGG16 our available. The pretrained large models are provided in 'outputs/models' <br/>
<br/>
Other optional parameters are:<br/>
-temperature(int) - Temperature for kd experiments<br/>
-pin_memory(bool)<br/>
-save_dir - Where to save models<br/>
-device - Which device to use (cuda/cpu)<br/>
-epochs - Default 90<br/>
-lr - Learning rate (default 0.01)<br/>
-batch_size - Default 128<br/>
-step-size - After every step_size epochs reduce the lr by xgamma (default 60)<br/>
-gamma - By how much to reduce lr every step_size (default 0.1)<br/>
-num_workers - Number of workers (default 0)
<br/>
<br/>
Example of running CREMBO compression:

```
python dnn2dnn_compression.py -dataset_name cifar10 -model_name lenet -large_model resnet18 -experiment compress -device 'cuda' -pin_memory -num_workers 4 -experiment_name crembo_resnet18_Lenet5
```

Example of running KD compression:

```
python dnn2dnn_compression.py -dataset_name cifar10 -model_name lenet -large_model resnet18 -experiment kd -device 'cuda' -pin_memory -num_workers 4 -temperature 5 -experiment_name kd_resnet18_Lenet5

```

Example of training a model (a large model)

```
python dnn2dnn_compression.py -dataset_name cifar10 -model_name resnet18 -experiment train -device 'cuda' -pin_memory -num_workers 4 -experiment_name M_resnet18
```

<br/>
Datasets:<br/>
The CIFAR10 dataset will be automatically downloaded via Pytorch
