# BMST
# BMST: Boosted Model Speedup and Memory Reduction Training

BMST is an innovative training method designed to significantly reduce training time and memory consumption for deep learning models. By leveraging data compression techniques, BMST enhances the efficiency of the training process without compromising the quality of the trained models.

## Key Features

- **Training Acceleration**: Achieves faster training times compared to traditional methods without sacrificing accuracy. This ensures that while the training process is made more efficient, the quality and predictive performance of the model remain high.
- **Memory Efficiency**: Reduces memory usage during training, allowing for larger models or datasets to be used within limited hardware resources.
- **Open Source**: The base model implementation references open-source code available online, ensuring transparency and ease of adaptation.
## Supported Datasets

BMST has been rigorously tested and validated across four public datasets to ensure its effectiveness and robustness:

- CIFAR-10
- CIFAR-100
- Oxford Flowers 102
- miniImageNet

## Installation

To install BMST, follow these simple steps:

1. Clone the repository:
2. git clone https://github.com/yourusername/bmst.git
3. 2. Navigate to the BMST directory:
cd bmst

3. Install the required dependencies using pip:
pip install -r requirements.txt


# BMST Performance Results on CIFAR-10

| Mode | Model | Training Time/epoch (s)↑ | Speed-up↑ | Memory Usage↓ (MiB) |
|------|-------|---------------------------|------------|---------------------|
| base | resnet18 | 19.31 | 1 | 3670 |
|      | resnet50 | 52.01 | 1 | 6764 |
|      | resnet101 | 86.41 | 1 | 8562 |
|      | resnet152 | 125.08 | 1 | 9820 |
|      | InceptionV3 | 113.54 | 1 | 8176 |
| fp16 | resnet18 | 10.39 | 1.86 | 2546 |
|      | resnet50 | 31.38 | 1.66 | 4048 |
|      | resnet101 | 48.72 | 1.77 | 5068 |
|      | resnet152 | 68.44 | 1.83 | 6298 |
|      | InceptionV3 | 63.04 | 1.80 | 5512 |
| BMST c2 2to1 | resnet18 | 11.85 | 1.63 | 2904 |
|             | resnet50 | 28.91 | 1.80 | 4462 |
|             | resnet101 | 47.25 | 1.83 | 5450 |
|             | resnet152 | 67.52 | 1.85 | 6602 |
| BMST-fp16 | resnet18 | 9.12 | 2.12 | 2432 |
|           | resnet50 | 19.20 | 2.71 | 3474 |
|           | resnet101 | 34.21 | 2.53 | 4150 |
|           | resnet152 | 47.56 | 2.63 | 4876 |
|           | InceptionV3 | 38.50 | 2.95 | 4162 |

# CIFAR-100 Dataset Results

| Mode | Model | Training Time/epoch (s)↑ | Speed-up↑ | Memory Usage↓ (MiB) |
|------|-------|---------------------------|------------|---------------------|
| base | resnet18 | 19.58 | 1 | 3670 |
|      | resnet50 | 52.26 | 1 | 6764 |
|      | resnet101 | 86.62 | 1 | 8562 |
|      | resnet152 | 125.13 | 1 | 9820 |
|      | InceptionV3 | 114.02 | 1 | 8176 |
| fp16 | resnet18 | 10.96 | 1.79 | 2546 |
|      | resnet50 | 31.28 | 1.67 | 4048 |
|      | resnet101 | 48.82 | 1.77 | 5068 |
|      | resnet152 | 68.78 | 1.82 | 6298 |
|      | InceptionV3 | 63.46 | 1.80 | 5512 |
| BMST c2 2to1 | resnet18 | 12.01 | 1.63 | 2904 |
|             | resnet50 | 29.09 | 1.80 | 4462 |
|             | resnet101 | 47.16 | 1.84 | 5450 |
|             | resnet152 | 67.68 | 1.85 | 6602 |
| BMST-fp16 | resnet18 | 8.91 | 2.20 | 2432 |
|           | resnet50 | 20.1 | 2.6 | 3474 |
|           | resnet101 | 34.05 | 2.54 | 4150 |
|           | resnet152 | 47.59 | 2.63 | 4876 |
|           | InceptionV3 | 38.73 | 2.94 | 4162 |

# Oxford Flowers 102 Dataset Results

| Mode | Model | Training Time/epoch (s)↑ | Speed-up↑ | Memory Usage↓ (MiB) |
|------|-------|---------------------------|------------|---------------------|
| base | resnet18 | 5.11 | 1 | 3668 |
|      | resnet50 | 9.4 | 1 | 6764 |
|      | resnet101 | 14.19 | 1 | 8562 |
|      | resnet152 | 19.19 | 1 | 9820 |
|      | InceptionV3 | 17.53 | 1 | 8176 |
| fp16 | resnet18 | 4.07 | 1.26 | 2546 |
|      | resnet50 | 6.85 | 1.37 | 4046 |
|      | resnet101 | 9.64 | 1.47 | 5068 |
|      | resnet152 | 12.22 | 1.57 | 6298 |
|      | InceptionV3 | 10.98 | 1.60 | 5512 |
| BMST c2 2to1 | resnet18 | 4.14 | 1.23 | 2902 |
|             | resnet50 | 6.25 | 1.50 | 4462 |
|             | resnet101 | 8.62 | 1.65 | 5450 |
|             | resnet152 | 11.33 | 1.69 | 6602 |
| BMST-fp16 | resnet18 | 3.7 | 1.38 | 2432 |
|           | resnet50 | 5.69 | 1.65 | 3474 |
|           | resnet101 | 7.6 | 1.87 | 4152 |
|           | resnet152 | 9.07 | 2.12 | 4876 |
|           | InceptionV3 | 7.49 | 2.34 | 4162 |

# miniImageNet Dataset Results

| Mode | Model | Training Time/epoch (s)↑ | Speed-up↑ | Memory Usage↓ (MiB) |
|------|-------|---------------------------|------------|---------------------|
| base | resnet18 | 33.87 | 1 | 4454 |
|      | resnet50 | 98.79 | 1 | 9872 |
|      | resnet101 | 160.55 | 1 | 8528 |
|      | resnet152 | 228.94 | 1 | 10004 |
|      | InceptionV3 | 233.66 | 1 | 9984 |
| fp16 | resnet18 | 21.06 | 1.61 | 3352 |
|      | resnet50 | 59.5 | 1.66 | 6292 |
|      | resnet101 | 102.87 | 1.56 | 5292 |
|      | resnet152 | 144.02 | 1.59 | 6590 |
|      | InceptionV3 | 143.42 | 1.63 | 5952 |
| BMST c2 2to1 | resnet18 | 20.66 | 1.64 | 3352 |
|             | resnet50 | 53.96 | 1.83 | 7562 |
|             | resnet101 | 91.71 | 1.75 | 5732 |
|             | resnet152 | 128.68 | 1.78 | 6976 |
| BMST-fp16 | resnet18 | 17.23 | 1.97 | 2716 |
|           | resnet50 | 35.49 | 2.78 | 4980 |
|           | resnet101 | 72.57 | 2.21 | 4254 |
|           | resnet152 | 98.82 | 2.32 | 5010 |
|           | InceptionV3 | 84.54 | 2.76 | 4580 |


*Note: The arrows (↑, ↓) indicate the direction of improvement, where lower training times and memory usage are better, and higher speed-up values indicate greater efficiency gains.*
[^1]: *Footnote text here, for example: "The results were obtained using an NVIDIA GeForce RTX 3080 GPU for testing purposes."*


# Trains a standard ResNet-18 on miniImageNet with a learning rate of 0.02 using GPU
python ntrain.py -net resnet18 -gpu -lr 0.02

# Trains a ResNet-18 with mixed precision and learning rate of 0.02 using GPU
python ntrain.py -net resnet18 -gpu -lr 0.02 -fp16

# Trains a BMST ResNet-18 with merging option 2, learning rate of 0.02 using GPU
python ntrain.py -net c2mixt_resnet18 -gpu -merge 2 -lr 0.02

# Trains a BMST ResNet-18 with merging option 2, mixed precision, and learning rate of 0.02 using GPU
python ntrain.py -net c2mixt_resnet18 -gpu -merge 2 -lr 0.02 -fp16


The -net option specifies the network architecture. For example, resnet18, resnet50, resnet101, resnet152, and inceptionv3.
The -gpu option indicates that the training should be performed on a GPU.
The -lr option sets the learning rate for the training process. All examples use a learning rate of 0.02.
The -fp16 option enables mixed precision training, which can help in reducing memory usage and potentially speeding up the training.


python ntrain.py -net resnet18 -gpu  -lr 0.02 

python ntrain.py -net resnet18 -gpu  -lr 0.02  -fp16 

python ntrain.py -net c2mixt_resnet18 -gpu -merge 2 -lr 0.02  

python ntrain.py -net c2mixt_resnet18 -gpu -merge 2 -lr 0.02  -fp16


python ntrain.py -net resnet50 -gpu  -lr 0.02 

python ntrain.py -net resnet50 -gpu  -lr 0.02  -fp16 

python ntrain.py -net c2mixt_resnet50 -gpu -merge 2 -lr 0.02  

python ntrain.py -net c2mixt_resnet50 -gpu -merge 2 -lr 0.02  -fp16


 
python ntrain.py -net resnet101 -gpu  -lr 0.02  
 
python ntrain.py -net resnet101 -gpu  -lr 0.02  -fp16  

python ntrain.py -net c2mixt_resnet101 -gpu -merge 2 -lr 0.02   

python ntrain.py -net c2mixt_resnet101 -gpu -merge 2 -lr 0.02  -fp16   


python ntrain.py -net resnet152 -gpu  -lr 0.02    

python ntrain.py -net resnet152 -gpu  -lr 0.02  -fp16    

python ntrain.py -net c2mixt_resnet152 -gpu -merge 2 -lr 0.02    
 
python ntrain.py -net c2mixt_resnet152 -gpu -merge 2 -lr 0.02  -fp16 





python ntrain.py -net inceptionv3 -gpu  -lr 0.02   

python ntrain.py -net inceptionv3 -gpu -lr 0.02  -fp16  
 
python ntrain.py -net c3mixt_inceptionv3 -gpu -merge 2 -lr 0.02  

python ntrain.py -net c3mixt_inceptionv3 -gpu -merge 2 -lr 0.02  -fp16  




# For the miniImageNet dataset, which uses 48x48 images, a batch size of 60 is specified to avoid running out of GPU memory. If the batch size is not specified in the command, the script might use a default batch size that could lead to GPU memory issues.


 
python ntrain.py -net resnet101 -gpu  -lr 0.02  -b 60
 
python ntrain.py -net resnet101 -gpu  -lr 0.02  -fp16  -b 60

python ntrain.py -net c2mixt_resnet101 -gpu -merge 2 -lr 0.02    -b 60

python ntrain.py -net c2mixt_resnet101 -gpu -merge 2 -lr 0.02  -fp16   -b 60


python ntrain.py -net resnet152 -gpu  -lr 0.02    -b 60

python ntrain.py -net resnet152 -gpu  -lr 0.02  -fp16    -b 60

python ntrain.py -net c2mixt_resnet152 -gpu -merge 2 -lr 0.02     -b 60
 
python ntrain.py -net c2mixt_resnet152 -gpu -merge 2 -lr 0.02  -fp16   -b 60





python ntrain.py -net inceptionv3 -gpu  -lr 0.02   -b 60

python ntrain.py -net inceptionv3 -gpu -lr 0.02  -fp16  -b 60
 
python ntrain.py -net c3mixt_inceptionv3 -gpu -merge 2 -lr 0.02   -b 60
 
python ntrain.py -net c3mixt_inceptionv3 -gpu -merge 2 -lr 0.02  -fp16   -b 60




