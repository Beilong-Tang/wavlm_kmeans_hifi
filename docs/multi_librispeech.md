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
kmeans_dir=ckpt/librispeech_1_10spk
kmeans_ckpts=$(find $kmeans_dir -type f | tr '\n' ' ') # Get all the ckpts and separate them into spaces

python inference_audio.py --kmeans_path $kmeans_ckpts --audio_scp <audio_scp> --output_dir <output_dir> # Replace audio_scp and output_dir 
```