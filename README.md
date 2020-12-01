# Robust Model Compression Using Deep Hypotheses 

This repository includes the code for running the experiments presented in the paper 

## Requirements
It is required to install Pytorch. 
To install requirements:

```setup
pip install -r requirements.txt
```

# Running experiments
## Compressing to Interpretable Models
To run the experiments presented in the paper, you need to use the 'run.py' file and provide the required parameters:<br/>
-model rf/nn      describing which model you wish to compress<br/>
-experiment_name  generalization/robustness describing which experiment to run<br/>
-dataset_name heart/dermatology/arrhythmia/breast_cancer/iris describing which dataset to use<br/>

There are also optional parameters<br/>
-n_experiments (default 20) the number of experiments to run<br/>
-tree_depth (default 4) the depth of the compressed trees<br/>
-delta (default 1) the size of the interval in CREMBO algorithm<br/>
-rf_depth (default 12) the depth of trees in random forest<br/>
-rf_trees (default 100) number of trees in random forest<br/>

To train the experiments, use the 'run.py', for example::

```train
python run.py -model rf -experiment_name generalization -dataset_name heart
```

After the experiment is finished the results will be printed to the console


Datasets:<br/>
Datasets are provided here but they originate from the UCI repository  
Heart (cleveland: https://archive.ics.uci.edu/ml/datasets/Heart+Disease  
Dermatology: https://archive.ics.uci.edu/ml/datasets/Dermatology  
Arrythmia: https://archive.ics.uci.edu/ml/datasets/Arrhythmia  
Breast_cancer: https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29  
Iris: https://archive.ics.uci.edu/ml/datasets/Iris  

## DNN to DNN Compression
To run the experiments presented in the paper you need to use the 'dnn2dnn_compression.py' file and provide the required parameters:<br/>
-dataset_name - Only cifar10 is currently supported.<br/>
-experiment - Type of experiment: train/compress (CREMBO)/kd<br/>
-experiment_name -  Name of experiment (used for saving models)<br/> 
-model_name - The type of model to train for 'train' experiments or the type of model to compress to for 'compress' and 'kd' experiments. (lenet/mobilenetv2/resnet18/VGG16) <br/>
<br/>
For 'compress' and 'kd' experiments you also need to provide
-large_model - The type of large model used for compression. Either resnet18 or VGG16 our avilable. The pretrained large models are provided in 'outputs/models' <br/>
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
