## WavLM_Kmeans_Hifi

Generate audio using Kmeans + Conformer (Detokenizer) + Hifi-Gan.

## Pre-requisite

1. Download our conformer ckpt from [here](https://drive.google.com/file/d/1a3LbfVxURgcy7oM3K-IZzQ_Gz6-W_msL/view?usp=sharing) to `./ckpt` folder.
2. Download Kmeans model from [here](https://drive.google.com/file/d/1ckxOx5MVxuHB_6qeEJo1Ae_c-8wZEgY5/view?usp=sharing) to `./ckpt` folder.
3. Download WavLM from [here](https://drive.google.com/file/d/12-cB34qCTvByWT-QtOcZaqwwO21FLSqU/view) to `./ckpt` folder.

The folder structure under ckpt should be:
```
ckpt/
├── g_02500000.pt
├── step160000_model.pth
└── WavLM-Large.pt
```

## Running

Put all the audio paths into a file. For example `audio.scp` should contain:
```
path/to/audio_1.wav
path/to/audio_2.wav
path/to/audio_3.wav
```
where each line stands for each audio source path.

Run

```shell
python inference_audio.py --audio_scp <audio_scp> --output_dir <output_dir> 
```


## Experiments

### LJSpeech

Kmeans and Detokenizer is trained on LJSpeech (Single speaker talking).

#### Pre-requisite

1. Download our conformer ckpt from [here](https://drive.google.com/file/d/1FRS-iKEbtwnRy9Ihyc1VGoM-j7JAFXdP/view?usp=sharing) to `./ckpt/LJSpeech` folder (or anywhere).
2. Download Kmeans model from [here](https://drive.google.com/file/d/1laO1yI35VTqxmfgv2opWqj2FMiJlwL3L/view?usp=sharing) to `./ckpt/LJSpeech` folder (or anywhere).

#### Running

```shell
python inference_audio.py --audio_scp <audio_scp> --output_dir <output_dir> --kmeans_path ./ckpt/LJSpeech/kmeans-cluster-1024-k_1024.pt --ckpt_path ./ckpt/LJSpeech/LJSpeech_k_1024_model.pt --config ./ckpt/LJSpeech/k_1024.yaml
```


### MSP-IMPROV

#### Pre-requisite

1. Download our conformer ckpt from [here](https://drive.google.com/file/d/1G_sD2-UkvzezsqGHLzc-H6Kirx489iR_/view?usp=sharing) to `./ckpt/MSP-IMPROV` folder (or anywhere).
2. Download Kmeans model from [here](https://drive.google.com/file/d/1UYdbNz0aquUsQWqLERbRxylz3C7XVODU/view?usp=sharing) to `./ckpt/MSP-IMPROV` folder (or anywhere).

#### Running

```shell
python inference_audio.py --audio_scp <audio_scp> --output_dir <output_dir> --kmeans_path ./ckpt/MSP-IMPROV/MSP_IMPROV_kmeans-cluster-1024-k_1024.pt --ckpt_path ./ckpt/MSP-IMPROV/MSP-Improv_k_1024_model.pt
```

## Inference using multiple models

If you follow the previous steps and the models and ckpts are downloaded
in the cooresponding folders. You can directly run

```shell
python inference_audio_fusion.py 
```

which will run the inference using __LibriSpeech__, __LJSpeech__, and __MSP_IMPROV__. 

If not, you can check the arguments in `inference_audio_fusion.py` and 
change the model ckpt directly. 

The `kmeans_path`, `ckpt_path` , `config`
are now a list of strings where each triple is a model with corresponding
kmeans ckpt and conformer ckpt. 

We randomly select audios from the scp to be output by any one of the model.


## Multiple LibriSpeech speakers (random)

### exp1 

#### Introduction

