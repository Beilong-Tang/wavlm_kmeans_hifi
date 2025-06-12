#!/bin/bash
export CUDA_VISIBLE_DEVICES=4

flag=_spkall
out=anon_speech_mkf
conformer=ckpt/librispeech_conformer_e_50.pth

for dset in  libri_dev_enrolls libri_test_enrolls train-clean-360; do


python exp/multiple_kmeans_train_one_conformer/inference_audio_multi_kmeans.py \
    --audio_scp data/$dset/wav.scp \
    --output_dir $out/$dset \
    --kmeans_scp  kmeans$flag.scp \
    --ckpt $conformer \

done

flag=_libri_every_20

for dset in  libri_dev_{trials_f,trials_m} \
		libri_test_{trials_f,trials_m} \
		IEMOCAP_dev IEMOCAP_test; do

python exp/multiple_kmeans_train_one_conformer/inference_audio_multi_kmeans.py \
    --audio_scp data/$dset/wav.scp \
    --output_dir $out/$dset \
    --kmeans_scp  kmeans$flag.scp \
    --ckpt $conformer \

done
