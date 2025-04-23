#!/bin/bash

##################################################
## CONFIG: librispeech base path and wavlm ckpt ##
##################################################

base_path=/SMIIPdata3/zbang/Corpus/librispeech/LibriSpeech
wavlm_ckpt=/DKUdata/tangbl/wavlm/WavLM-Large.pt

########################################
# Exp1: 10 randomly select 10 speakers #
########################################
# spks="14 90 126 176 166 204 209 216 240 242"
# num_proc=8
# gpus="cuda:1 cuda:2 cuda:3 cuda:4"
# config=exp/LibriSpeech_Multiple/config/k_1024.yaml

# ckpt_dir=exp/LibriSpeech_Multiple/ckpt/exp1_10

# python recipes/train_kmeans_librispeech.py --base_path $base_path \
#     --spk_list $spks --config $config --wavlm_ckpt $wavlm_ckpt \
#     --num_proc $num_proc --gpus $gpus --ckpt_dir $ckpt_dir

######################################################
## Exp2: run Kmeans on the whole librispeech dataset #
######################################################
config=exp/librispeech_multiple/config/k_1024.yaml
ckpt_dir=exp/librispeech_multiple/ckpt/librispeech_k_1024_config_all

### DDP 
num_proc=24
gpus="cuda:1 cuda:2 cuda:3 cuda:4 cuda:6 cuda:7"

## Data Path
tr_360=$(ls /SMIIPdata3/zbang/Corpus/librispeech/LibriSpeech/train-clean-360 | tr '\n' ' ') # train-clean-360
tr_100=$(ls /SMIIPdata3/zbang/Corpus/librispeech/LibriSpeech/train-clean-100 | tr '\n' ' ') # train-clean-100
spks="$tr_360 $tr_100"
python -u kmeans/train_kmeans_librispeech.py --base_path $base_path \
    --spk_list $spks --config $config --wavlm_ckpt $wavlm_ckpt \
    --num_proc $num_proc --gpus $gpus --ckpt_dir $ckpt_dir