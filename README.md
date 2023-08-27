# MomentDiff: Generative Video Moment Retrieval from Random to Real 
by 
Pandeng Li<sup>1</sup>, Chen-Wei Xie<sup>2</sup>, Hongtao Xie<sup>1</sup>, Liming Zhao<sup>2</sup>, Lei Zhang<sup>1</sup>, Yun Zheng<sup>2</sup>
, Deli Zhao<sup>2</sup>, Yongdong Zhang<sup>1</sup>

<sup>1</sup> University of Science and Technology of China, <sup>2</sup> DAMO Academy, Alibaba Group

	


[[Arxiv](https://arxiv.org/abs/2307.02869)]

The code will be released in October 2023.

----------

## Prerequisites
<b>0. Clone this repo</b>

<b>1. Prepare datasets</b>

<b>Charades-STA</b> : Download feature files for Charades-STA dataset.

VGG features and labels: Download [Charades-STA-VGG](https://github.com/TencentARC/UMT),

SF+C features: We followed Moment-DETR to use [Charades-STA-SF+C](https://github.com/linjieli222/HERO_Video_Feature_Extractor). 


<b>QVHighlights</b> : Download official feature files for QVHighlights dataset from Moment-DETR. 

SF+C features: Download [moment_detr_features.tar.gz](https://drive.google.com/file/d/1Hiln02F1NEpoW8-iPZurRyi-47-W2_B9/view?usp=sharing).
```
tar -xf path/to/moment_detr_features.tar.gz
```


<b>TACoS</b> : Prepare features for TACoS dataset. 

C3D features: : According to [VSLNet](https://github.com/26hzhang/VSLNet/tree/master/prepare), convert the pre-trained C3D visual features from [TALL](https://drive.google.com/uc?export=download&id=1zQp0aYGFCm8PqqHOh4UtXfy2U3pJMBeu).

<b>ActivityNet</b> : Prepare features for ActivityNet dataset. 

C3D features: : According to [VSLNet](https://github.com/26hzhang/VSLNet/tree/master/prepare), convert the pre-trained C3D visual features from [ActivityNet](http://activity-net.org/challenges/2016/download.html#c3d).


<b>2. Install dependencies.</b>

```
#使用conda python-3.7.16
conda create -n momentdiff python=3.7.16 
. activate
conda activate momentdiff
cd MomentDiff
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 torchtext==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt -i https://pypi.mirrors.ustc.edu.cn/simple
pip install tqdm ipython easydict tensorboard tabulate scikit-learn pandas -i https://pypi.mirrors.ustc.edu.cn/simple
```

## Charades-STA

### Training
Training with (VGG) and (SF+C) can be executed by running the shell below:
```
bash momentdiff/scripts/train_charades_vgg.sh 
bash momentdiff/scripts/train_charades_sf.sh 
```

Training on two anti-bias datasets can be executed by running the shell below:
```
bash momentdiff/scripts/train_anti_charades_len.sh 
bash momentdiff/scripts/train_anti_charades_Mom.sh 
```

Training on Charades-CD and ActivityNet-CD can be executed by running the shell below:
```
bash momentdiff/scripts/train_charades_CD.sh 
bash momentdiff/scripts/train_anet_CD.sh 
```




## LICENSE
The annotation files and many parts of the implementations are borrowed Moment-DETR.
Following, our codes are also under [MIT](https://opensource.org/licenses/MIT) license.
 
