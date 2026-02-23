# PS-Seg
The official code for PS-Seg, accoding to the following paper:

* Meng Han, Xiaochuan Ma, Xiangde Luo, Wenjun Liao, Shichuan Zhang, Shaoting Zhang,  Guotai Wang, 
[PS-seg: Learning from partial scribbles for 3D multiple abdominal organ segmentation.][paper_link] Neurocomputing, 672, April (2026): 132837. 

[paper_link]: https://www.sciencedirect.com/science/article/pii/S0925231226002341

BibTeX entry:

    @article{han2026ps_seg,
    author = {Meng Han and Xiaochuan Ma and Xiangde Luo and Wenjun Liao and Shichuan Zhang and Shaoting Zhang and  Guotai Wang},
    title = {{PS-seg: Learning from partial scribbles for 3D multiple abdominal organ segmentation}},
    year = {2026},
    url = {https://doi.org/10.1016/j.neucom.2026.132837},
    journal = {Neurocomputing},
    volume = {672},
    pages = {132837},
    }

### Overall Framework
The overall framework of PS-Seg：
![Overall](imgs/image.png)


# Dataset
* The WORD dataset can be downloaded from [WORD](https://github.com/HiLab-git/WORD?tab=readme-ov-file).
* The Synapse dataset can be downloaded from [Synapase](https://www.synapse.org/Synapse:syn3193805/files/)

# Usage with PyMIC
To facilitate the use of code and make it easier to compare with other methods, we have implemented PS-Seg in PyMIC, a Pytorch-based framework for annotation-efficient segmentation. The core modules of PS-Seg in PyMIC can be found [here][pymic_psseg]. It is suggested to use PyMIC for this experiment. In the following, we take the WORD dataset as an example for scribble-supervised segmentation.

[pymic_psseg]: https://github.com/HiLab-git/PyMIC/blob/master/pymic/net_run/weak_sup/wsl_psseg.py

### Step 0: Preparation
#### 0.1. Environment Setup. 
```sh
conda create -n PSSeg python=3.10
conda activate PSSeg
pip install -r requirements.txt
pip install pymic
```
#### 0.2. Dataset processing.
Preprocess WORD dataset by:
```sh
python data/preprocess_WORD.py
```
Generate scribble label by:
```sh
python data/scribble_generator.py
```

To speed up the training process, we convert the data into h5 files by:
```sh
python data/image2h5.py
```

### Step 1: Training 
The configurations including dataset, network, optimizer and hyper-parameters are contained in the configure file
`config/psseg_word.cfg`. PS-Seg needs a multi-decoder network, and it is defined in `networks/TDNet_3D.py`. A reimplementatin of this network has also been provided in PyMIC. 

Train the PS-Seg model by running:
```sh
python run.py train config/psseg_word.cfg
```

### Step 2: Test
Obtain predictions for testing images:
```
python run.py test config/psseg_word.cfg
```

### Step 3: Compare with other weakly supervised segmentation methods
PyMIC also provides implementation of several other weakly supervised methods (learning from scribbles). Please see [PyMIC_examples/seg_weak_sup/ACDC][PyMIC_example_link] for examples.

[PyMIC_example_link]:https://github.com/HiLab-git/PyMIC_examples/tree/main/seg_weak_sup/ACDC 

### Acknowledgement
The code of scribble-supervised learning framework is borrowed from [WSL4MIS](https://github.com/HiLab-git/WSL4MIS)