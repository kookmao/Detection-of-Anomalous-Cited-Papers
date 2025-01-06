# CITAD: Detecting and Evaluating Anomalously Cited Papers Over Time using Anomaly Detection in Dynamic Graphs via Transformer 
This repo covers an reference implementation for the paper "[Anomaly detection in dynamic graphs via transformer](https://arxiv.org/pdf/2106.09876.pdf)" (TADDY).

[TADDY: Anomaly detection in dynamic graphs via transforme](https://github.com/yuetan031/TADDY_pytorch)

## Requirments
* Python==3.8
* PyTorch==1.7.1
* Transformers==3.5.1
* Scipy==1.5.2
* Numpy==1.19.2
* Networkx==2.5
* Scikit-learn==0.23.2

## Usage
### Step 0: Prepare Data
```
python 0_prepare_data.py --dataset five_year --anomaly_per 0.1
```

### Step 1: Train Model
```
python 1_train.py --dataset five_year --anomaly_per 0.1
```
