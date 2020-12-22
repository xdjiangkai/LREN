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
# Result
## Hyperspectral Datasets
For the ease of reproducibility. We provide experimental results on hyperspectral datasets as belows:

|Dataset |AUC(P_d, P_f)  |AUC(P_f, \tau) |
|:-----: |:----------:   |:-----------:  |
|SanDiego|0.9897         |0.0134         |
|Hydice  |0.9998         |0.0102         |
|Coast   |0.9982         |0.0276         |
|Pavia   |0.9925         |0.0433         |
|Average |0.9951         |0.0236         |

## Extension
Our approach is based on the following three properties:
1. The background (i.e., the normal instances) still preserves a low-rank property lying in a low-dimensional manifold.
2. The presence probability of the anomaly is much lower than that of the background (i.e., the normal instances).
3. The latent representation serves the anomaly estimation, which optimally updates the parameters of the deep latent space.  

Thus, LREN is applicable to anomaly detection tasks that satisfy these three properties. We conducted experiments on [Outlier Detection DataSets (ODDS)](http://odds.cs.stonybrook.edu/#table1) to demonstrate the effectiveness of LREN in other anomaly detection tasks.

|Dataset   |AUC(P_d, P_f)  |AUC(P_f, \tau) |Precision|Recall   |F1       |
|:-----:   |:----------:   |:-----------:  |:-------:|:-------:|:-------:|
|Thyroid   |0.9910         |0.0980         |0.8571   |0.6452   |0.7362   |
|Arrhythmia|0.8353         |0.0490         |0.6389   |0.451    |0.5287   |
|Musk      |0.9734         |0.4263         |0.0730   |1        |0.1362   |

[comment]: <> (|KddCup99  |0.9951         |0.0236         ||||)
