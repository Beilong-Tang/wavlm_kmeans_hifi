
#### install and download VPC data

follow instructions https://github.com/Voice-Privacy-Challenge/Voice-Privacy-Challenge-2024 
download IEMOCAP data from https://drive.google.com/file/d/13Gue1-EwvpzK2bZEFpTSGKdB5oDhUjbf/view?usp=sharing
change filename path in data/IEMOCAP_test/wav.scp and data/IEMOCAP_dev/wav.scp



### generate speech
cd wavlm_kmeans_hifi/
ln -sr Voice-Privacy-Challenge-2024/data .
ln -sr Voice-Privacy-Challenge-2024/corpora .

```shell
bash 01_gen.sh
```


#### evaluate anonymized speech

change $anon_dir and $anon_suffix in 02_eval.sh and run
cp 02_eval.sh Voice-Privacy-Challenge-2024/

cd Voice-Privacy-Challenge-2024

```shell
bash 02_eval.sh
```


