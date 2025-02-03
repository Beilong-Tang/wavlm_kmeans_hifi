## Multiple LibriSpeech speakers (random)

### Experiment 1 

#### Introduction

10 spks from librispeech train-clean-360 is used for training the kmeans: 

#### Inference

Random speakers will be chonse for inference.

#### Run

Firstly, download `librispeech_1_10spk.tar.gz` and extract it at `ckpt`. Output looks like `ckpt/librispeech_1_10spk/*.pt`.

Secondly, run:

```shell
python inference_audio.py
```

