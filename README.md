## 运行环境
conda create -name asr3.9 python=3.9<br/>
conda activate asr3.9<br/>
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/<br/>

## 模型训练数据集下载
由于模型训练数据集体积太大, 故放到百度网盘了.<br/>
沪语语音模型数据集下载地址:<br/>
https://pan.baidu.com/s/189-8oaCzCwp_qhYYOIuMtw?pwd=6666

## 使用的工具库
Huggingface的transformers和huggingsound。

## 沪语ASR模型
包含一个沪语ASR模型(沪语语音->沪语转写文本)和机器翻译模型(沪语转写文本->普通话文本)。

## 数据
包含Magichub开源数据集、喜马拉雅，中国语言网爬取的数据集、讯飞TTS生成的wav数据。

## 训练脚本
train.py 用于训练ASR模型

train_translation.py 用于训练MT模型

## 服务
使用fastapi进行整个模型的部署，运行run_service.sh部署。

