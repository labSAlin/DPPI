# DPPI

#### Codes for paper "Identification of All-to-all Protein-protein Interactions Based on Deep Hash Learning" by Yue Jiang, Yuxuan Wang, Lin Shen, Donald Adjeroh, Zhidong Liu and Jie Lin, which is submitted to BMC Bioinformatics 

## required installed softwares(running environment):  

1. python 3.6
2. Tensorflow 2.0

## Usage：

### Training

python train.py [dataset]

Note:

1. The dataset can be one member data of four species: C.elegan, Drosophila, E.coli,or Human
2. After training, the output model will be stored in the current folder: DPPI_Model/[dataset]

### Predicting

python predict.py [dataset]

Note: 

1. The output predicted result file named test.txt is stored  under the current folder: PPIdata/[dataset]

## Comment

1. These codes are developed on the base of the work: Elabd H, Bromberg Y, Hoarfrost A, Lenz T, Wendorff M. Amino acid encoding for deep learning applications .BMC Bioinformatics. 2020;21(1):1–14.  

