| 第五届计图挑战赛赛道一：rank3

# Jittor 超声肿瘤乳腺癌Bi-Rads分级 

![主要结果](https://github.com/ZhangBY-Lab/jittor-shijieheping-breast_cancer_cls/blob/main/models/demo.jpg)


## 简介

本项目包含了第五届计图挑战赛计图 - 超声乳腺癌肿瘤Bi-Rads分级比赛的代码实现。本项目的核心任务是在jittor深度学习框架下开发乳腺癌超声图像Bi-rads分级模型，促进完成超声乳腺肿瘤的辅助诊断任务。
此项目基于jimm，构建了双backbone+feature fusion+ gem pooling的分类模型，并使用改进的符合课程学习思想的labelsmooth以及增强的混合focal-bce loss等获得复赛第三名。

## 安装 

本项目可在 1 张 3090 上运行，单折训练时间约为 1 小时，总训练时间为 5 小时。

#### 运行环境
- ubuntu 22.04 LTS
- python >= 3.10
- jittor >= 1.3.9.14

#### 安装依赖
执行以下命令安装 python 依赖
```
conda create -n jittor python=3.11.2

conda activate jittor 

pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124

pip install -r requirements.txt

python -c "import torch; print(torch.cuda.is_available())"
```

## 训练

单卡训练可运行以下命令：
```
python _train.sh
```


## 推理

生成测试集上的结果可以运行以下命令：

```
python _test.py
```

## 致谢


此项目模型层面的代码参考了(https://github.com/Jittor-Image-Models/Jittor-Image-Models )。


