Change the fields in [../kmeans/train_librispeech_k_1024_all_spks.sh](../kmeans/train_librispeech_k_1024_all_spks.sh):

- `base_path`: LibriSpeech base path
- `wavlm_ckpt`: Wavlm ckpt path
- `tr_360`: path to clean-360
- `tr_100`: path to clean-100


Run:

```sh
bash kmeans/train_librispeech_k_1024_all_spks.sh
```