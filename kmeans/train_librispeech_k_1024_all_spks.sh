#!/bin/bash
export CUDA_VISIBLE_DEVICES=3
##################################################
## CONFIG: librispeech base path and wavlm ckpt ##
##################################################

base_path=/app/raid/Voice-Privacy-Challenge-2024/corpora/LibriSpeech/train-clean-360 # libri path
wavlm_ckpt=ckpt/WavLM-Large.pt # WavLM Path

########################################


## Data Path
lib=$(ls ${base_path}  | tr '\n' ' ') 
config=kmeans/exp/librispeech_k_1024_all_spks/k_1024.yaml

ckpt_dir=kmeans/exp/ckpt/libri_every_10

spks="$lib"
echo $spks
python kmeans/train_kmeans_librispeech.py --base_path $base_path \
    --spk_list $spks --config $config --wavlm_ckpt $wavlm_ckpt  --ckpt_dir $ckpt_dir --bs 10



ckpt_dir=kmeans/exp/ckpt/libri_sep

spks="$lib"
echo $spks
python kmeans/train_kmeans_librispeech.py --base_path $base_path \
    --spk_list $spks --config $config --wavlm_ckpt $wavlm_ckpt  --ckpt_dir $ckpt_dir --bs 1

exit 0



ckpt_dir=kmeans/exp/ckpt/libri_every_20

spks="$lib"
echo $spks
python kmeans/train_kmeans_librispeech.py --base_path $base_path \
    --spk_list $spks --config $config --wavlm_ckpt $wavlm_ckpt  --ckpt_dir $ckpt_dir --bs 20




echo $num_spks
ckpt_dir=kmeans/exp/ckpt/spkall
python kmeans/train_kmeans_librispeech.py --base_path $base_path --spk_list $spks\
	--config $config --wavlm_ckpt $wavlm_ckpt  --ckpt_dir $ckpt_dir  --bs 921
