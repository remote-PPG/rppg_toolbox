# Experiment
##
The project only publishes the project framework and models, and the algorithms for individual research reproduction are not uploaded and published.
## dataset loaer

## 运行

### 安装 conda，创建环境

```
# conda deactivate
# conda env remove -n rppg_toolbox -y

conda create -n rppg_toolbox python=3.9.19 pip=24.0 -y

conda activate rppg_toolbox

python -m pip install jupyter notebook -U

```
### 安装显卡驱动相应的 cuda、cuDNN

### 根据 cuda 安装相应 pytorch 的版本
* 不同版本的pytorch，加载训练好的模型可能会出错，可能需要参数```weights_only=False```

* Pytorch链接<a href='https://pytorch.org/get-started/previous-versions/'>pytorch version</a>

```
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1  pytorch-cuda=11.8 -c pytorch -c nvidia
```

### 安装其他依赖

```
pip install -r requirements.txt
# dlib
# conda install -c conda-forge dlib=19.24.0
```
### 编译安装支持的 ffmpeg
