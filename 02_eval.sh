#!/bin/bash

set -e

source env.sh


old_suffix=

anon_suffix=_llm2 #$TODO
submit_wav_dir=/home/xiaoxiao/anon_lm/wavlm_kmeans_hifi/anon_speech/ #$TODO
tar_dir=data


for dir in "${submit_wav_dir}"/*/; do
  # Get the base name of the directory without any suffix
  base=$(basename "$dir")

  # Construct the new name with anon_suffix
  new_dir="${submit_wav_dir}/${base}${anon_suffix}"

  # Rename the directory
  mv "$dir" "$new_dir"
done


mv ${submit_wav_dir}/* ${tar_dir}


#find data/train-clean-360${anon_suffix} -type f -name "*.flac" | while read -r file; do
#    mv "$file" "${file%.flac}.wav"
#done

### Variables

# Select the anonymization pipelinea
if [ -n "$1" ]; then
  anon_config=$1
else
  anon_config=configs/anon_mcadams.yaml
  # anon_config=configs/anon_sttts.yaml
  # anon_config=configs/anon_template.yaml
  # anon_config=configs/anon_asrbn.yaml
  # anon_config=configs/anon_nac.yaml
fi
echo "Using config: $anon_config"

#force_compute=
force_compute='--force_compute True'

# JSON to modify run_evaluation(s) configs, see below
eval_overwrite="{"

### Anonymization + Evaluation:

# find the $anon_suffix (data/dataset_$anon_suffix) = to where the anonymization produces the data files
if [[ $anon_suffix ]]; then
  eval_overwrite="$eval_overwrite \"anon_data_suffix\": \"$anon_suffix\"}"
fi


# Perform libri dev+test & IEMOCAP dev+test pre evaluation using pretrained ASR/ASV/SER models
python run_evaluation.py --config $(dirname ${anon_config})/eval_pre.yaml --overwrite "${eval_overwrite}" ${force_compute}

# Train post ASV using anonymized libri-360 and perform libri dev+test post evaluation
# ASV training takes ~2hours
python run_evaluation.py --config $(dirname ${anon_config})/eval_post.yaml --overwrite "${eval_overwrite}" ${force_compute}

# Merge results
results_summary_path_orig=$(python3 -c "from hyperpyyaml import load_hyperpyyaml; f = open('$(dirname ${anon_config})/eval_pre.yaml'); print(load_hyperpyyaml(f, ${eval_overwrite}).get('results_summary_path', ''))")
results_summary_path_anon=$(python3 -c "from hyperpyyaml import load_hyperpyyaml; f = open('$(dirname ${anon_config})/eval_post.yaml'); print(load_hyperpyyaml(f, ${eval_overwrite}).get('results_summary_path', ''))")
[[ "$results_summary_path_anon" == *"_test_tool"* ]] && exit 0

results_exp=exp/results_summary
mkdir -p ${results_exp}
{ cat "${results_summary_path_orig}"; echo; cat "${results_summary_path_anon}"; } > "${results_exp}/result_for_rank${anon_suffix}"
zip ${results_exp}/result_for_submission${anon_suffix}.zip -r exp/asr/*${anon_suffix} exp/asr/*${anon_suffix}.csv exp/ser/*${anon_suffix}.csv exp/results_summary/*${anon_suffix}* exp/asv_orig/*${anon_suffix} exp/asv_orig/*${anon_suffix}.csv exp/asv_anon${anon_suffix} > /dev/null
