# LREN
We provide a PyTorch implementation of LREN: Low-Rank Embedded Network for Sample-Free Hyperspectral Anomaly Detection.

Framework of LREN:

![schematic_diagram](https://github.com/xdjiangkai/LREN/blob/main/schematic_diagram.png "Framework of LREN")

# Prerequisites

- Linux 18.04 LTS
- Python 3.7.8
- Pytorch 1.5.1
- CUDA 10.2
- Scipy 1.2.1

# Citation
If you use this code for your research, please cite:

Low-Rank Embedded Network for Sample-Free Hyperspectral Anomaly Detection.
Kai Jiang, Weiying Xie, Jie Lei, Tao Jiang, Yunsong Li. In AAAI 2021

# Running Code
In this code, you can run our models on on four [benchmark hyperspectral datasets](http://xudongkang.weebly.com/data-sets.html), including SanDiego, Hydice, Coast, and Pavia.
## Usage
run "python run_main_LREN.py" 
## Result
For the ease of reproducibility. We provide some of the experimental results and as belows:

|Dataset |AUC(P_d, P_f)  |AUC(P_f,\tau)  |
|:-----: |:----------:   |:-----------:  |
|SanDiego|0.9897         |0.0134         |
|Hydice  |0.9998         |0.0102         |
|Coast   |0.9982         |0.0276         |
|Pavia   |0.9925         |0.0433         |
|Average |0.9951         |0.0236         |


