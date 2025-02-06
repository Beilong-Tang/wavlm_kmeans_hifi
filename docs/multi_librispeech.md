## Multiple LibriSpeech speakers (random)

### Experiment 1 

#### Introduction

10 spks from librispeech train-clean-360 is used for training the kmeans: 

#### Inference

Random speakers will be chosen for inference.

#### Run

Firstly, download `librispeech_1_10spk.tar.gz` and extract it at `ckpt`. Output looks like `ckpt/librispeech_1_10spk/*.pt`.

Secondly, run:

```shell
kmeans_dir=ckpt/librispeech_1_10spk
kmeans_ckpts=$(find -L $kmeans_dir -type f | tr '\n' ' ') # Get all the ckpts and separate them into spaces
python inference_audio_multi_kmeans.py --kmeans_path $kmeans_ckpts --audio_scp <audio_scp> --output_dir <output_dir> # Replace audio_scp and output_dir 
```



### Experiment 2

#### Introduction

100 spks from librispeech train-clean-{100,360}. (Note that the 100 speakers will be randomly chosen from the all spkears.)

#### Inference 

Random speakers will be chosen for inference. 


#### Run 

Firstly, download `librispeech_k_1024_config_all.tar.gz` and extract it at `ckpt`, output looks like `ckpt/librispeech_k_1024_config_all`.

Secondly, run 

```shell
kmeans_dir=ckpt/librispeech_k_1024_config_all
kmeans_ckpts=$(find -L $kmeans_dir -type f)
## Randomly select 100 speakers
spks=$(echo "$kmeans_ckpts" | shuf -n 100 --random-source=<(yes 42) | tr '\n' ' ')
echo $spks
python inference_audio_multi_kmeans.py --kmeans_path $spks --audio_scp <audio_scp> --output_dir <output_dir> # Replace audio_scp and output_dir 
```


### Experiment 3

#### Introduction

All spks from librispeech train-clean-{100,360}. (1165 speakers)

#### Inference 

Random speakers will be chosen for inference. 


#### Run 

Firstly, download `librispeech_k_1024_config_all.tar.gz` and extract it at `ckpt`, output looks like `ckpt/librispeech_k_1024_config_all`.

Secondly, run 

```shell
kmeans_dir=ckpt/librispeech_k_1024_config_all
kmeans_ckpts=$(find -L $kmeans_dir -type f | tr '\n' ' ') # Get all the ckpts and separate them into spaces
python inference_audio_multi_kmeans.py --kmeans_path $kmeans_ckpts --audio_scp <audio_scp> --output_dir <output_dir> # Replace audio_scp and output_dir
```