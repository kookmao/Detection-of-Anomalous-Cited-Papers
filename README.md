# CITAD: Detecting Anomalously Cited Papers using Anomaly Detection in Dynamic Graphs via Transformer 
This repo covers a reference implementation for the paper:
"[Anomaly detection in dynamic graphs via transformer(TADDY).](https://arxiv.org/pdf/2106.09876.pdf)"

## Requirments
* Python==3.8
* PyTorch==1.7.1
* Transformers==3.5.1
* Scipy==1.5.2
* Numpy==1.19.2
* Networkx==2.5
* Scikit-learn==0.23.2
```
pip install -r requitements.txt
```


# Usage
### Step 0: Prepare Data
```
python 0_prepare_data.py --dataset five_year --anomaly_per 0.1
```

### Step 1: Train Model
```
python 1_train.py --dataset five_year --anomaly_per 0.1
```


# Google Colab setup:
### Clone Repository:
```
!git clone https://github.com/kookmao/Detection-of-Anomalous-Cited-Papers
```
Refresh after running this cell:
```
!pip install -q condacolab
import condacolab
condacolab.install()
```
### Activate environment and install dependencies:
```
!conda create -n myenv python=3.8 -y
!conda init myenv
!conda activate myenv
```
```
!conda install -n myenv pytorch=1.7.1 torchvision torchaudio cudatoolkit=10.1 -c pytorch -y
!conda install -n myenv transformers=3.5.1 scipy=1.5.2 numpy=1.19.2 networkx=2.5 scikit-learn=0.23.2 -y
!conda install -n myenv matplotlib -y
!conda install -n myenv ipykernel -y
!conda install -n myenv pandas -y

%cd Detection-of-Anomalous-Cited-Papers
```
## Prepare Data:

```
!conda run -n myenv --live-stream python 0_prepare_data.py --dataset five_year --anomaly_per 0.1
```
## Train Data:
```
!conda run -n myenv --live-stream python 1_train.py --dataset five_year --anomaly_per 0.1
```
