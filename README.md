## About DeepDTAF

DeepDTAF is a deep learning architecture, which integrates local and global features to predict the binding affinity between ligands and proteins.  

The benchmark dataset can be found in `./data/`. The DeepDTAF model is available in `./src/`. And the result will be generated in `./runs/`. See our paper for more details.

### Requirements:
- python 3.7
- cudatoolkit 10.1.243
- cudnn 7.6.0
- pytorch 1.4.0
- numpy 1.16.4
- scikit-learn 0.21.2
- pandas 0.24.2
- tensorboard 2.0.0
- scipy 1.3.0
- numba 0.44.1
- tqdm 4.32.1

The easiest way to install the required packages is to create environment with GPU-enabled version:
```bash
conda env create -f environment_gpu.yml
conda activate DeepDTAF_env
```

Then, install the apex in the DeepDTAF_env environment:
```bash
git clone https://github.com/NVIDIA/apex
cd apex
python setup.py install
```
Since the codes for apex package in the above website have been updated, you can also install it using our uploaded package.  


### Training & Evaluation

to train your own model
```bash
cd ./src/
python main.py
```
to see the result
```bash
tensorboard ../runs/DeepDTAF_<datetime>_<seed>/
```

### contact
Kaili Wang: kailiwang@csu.edu.cn
