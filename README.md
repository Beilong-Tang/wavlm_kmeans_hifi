
## Pre-requisite

1. Download our conformer ckpt from [here](https://drive.google.com/file/d/1E9NDTnsQp73bHu1Xn8-aTdPDqq1w0K5x/view?usp=sharing) to `./ckpt` folder.
2. Download Kmeans model from [here](https://drive.google.com/file/d/1pQx_nFZ-Y7v7B_NCGAheAyJ9UNzhyNda/view?usp=sharing) to `./ckpt` folder and uncompress.
3. Download WavLM from [here](https://drive.google.com/file/d/12-cB34qCTvByWT-QtOcZaqwwO21FLSqU/view) to `./ckpt` folder.

The folder structure under ckpt should be:
```
ckpt/
├── librispeech_conformer_e_50.pth
├── attacker/LibriSpeech_wavlm_k1000_L7.pt
├── user/*pt
└── WavLM-Large.pt
```

## Anonymize Testdata
 
```bash 00_test.sh```
The anonymized speech is saved to `testdata/anon/`


## Anonymize VPC data
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



### Generate speech
`cd wavlm_kmeans_hifi/`

`ln -sr Voice-Privacy-Challenge-2024/data .`

`ln -sr Voice-Privacy-Challenge-2024/corpora .`

```shell
bash 01_gen_vpc.sh
```


### Evaluate generated speech

change `$anon_dir` and `$anon_suffix` in `02_eval_vpc.sh` and
cp `02_eval_vpc.sh` to `Voice-Privacy-Challenge-2024/`

cd Voice-Privacy-Challenge-2024

```shell
bash 02_eval_vpc.sh
```

check the results from `exp/results_summary/result_for_rank_${anon_suffix}`

