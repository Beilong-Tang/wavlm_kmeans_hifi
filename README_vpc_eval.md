
### Install and download VPC data

follow instructions https://github.com/Voice-Privacy-Challenge/Voice-Privacy-Challenge-2024 
```
## Install

1. `git clone https://github.com/Voice-Privacy-Challenge/Voice-Privacy-Challenge-2024.git`
2. `./00_install.sh`
3. `source env.sh`

## Download data

`./01_download_data_model.sh` 
A password is required; please register to get the password.  
```

download IEMOCAP data from https://drive.google.com/file/d/13Gue1-EwvpzK2bZEFpTSGKdB5oDhUjbf/view?usp=sharing
change filename path in `data/IEMOCAP_test/wav.scp` and `data/IEMOCAP_dev/wav.scp`


### Generate speech
`cd wavlm_kmeans_hifi/`

`ln -sr Voice-Privacy-Challenge-2024/data .`

`ln -sr Voice-Privacy-Challenge-2024/corpora .`

```shell
bash 01_gen.sh
```


### Evaluate generated speech

change `$anon_dir` and `$anon_suffix` in `02_eval.sh` and
cp `02_eval.sh` to `Voice-Privacy-Challenge-2024/`

cd Voice-Privacy-Challenge-2024

```shell
bash 02_eval.sh
```


