---
license: apache-2.0
datasets:
- reazon-research/reazonspeech
language:
- ja
metrics:
- cer
pipeline_tag: text-to-speech
---

# Japanese QuartzNet Large

ReazonSpeech Large(5000h)で学習したQuartzNet Large モデルです。

非常に軽量なモデルです。

# 学習状況
WandBのレポートに学習曲線などをまとめています。

[WandBレポート](https://wandb.ai/kinouchitakahiro/poetics_light_asr/reports/QuartzNet-reazon-speech-large---Vmlldzo5MjE0OTA5)

# 精度
こちらのASRモデルは　全て文字誤り率(CER)で評価しています。

Training set : 19.619 %

Validation set : 17.909 %

Test set :

Comming soon...