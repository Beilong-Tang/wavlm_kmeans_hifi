#!/bin/bash

for dset in  libri_dev_{enrolls,trials_f,trials_m} \
		libri_test_{enrolls,trials_f,trials_m} \
		IEMOCAP_dev IEMOCAP_test train-clean-360; do


#python inference_audio.py --audio_scp data/$dset/wav.scp --output_dir anon_speech/$dset/  --kmeans_path ckpt/LJSpeech/kmeans-cluster-1024-k_1024.pt --ckpt_path ckpt/LJSpeech/LJSpeech_k_1024_model.pt --config ckpt/LJSpeech/k_1024.yaml
python inference_audio.py --audio_scp data/$dset/wav.scp --output_dir anon_speech/$dset/  --kmeans_path ./ckpt/MSP-IMPROV/MSP_IMPROV_kmeans-cluster-1024-k_1024.pt --ckpt_path ./ckpt/MSP-IMPROV/MSP-Improv_k_1024_model.pt

done
