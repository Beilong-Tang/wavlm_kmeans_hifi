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





