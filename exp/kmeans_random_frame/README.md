# Kmeans random frame

For each frame of the speech, randomly select a kmeans model and infer on it. 

## Pre-requisite

Firstly, generate a `kmeans.scp` where each line is a name and path to a kmeans model. For example: 
```
kmeans_1 /path/to/kmeans_1.pt
kmeans_2 /path/to/kmeans_2.pt
kmeans_3 /path/to/kmeans_3.pt
```

To do so, you can put all kmeans models in one folder and run

```shell
KMEANS_DIR="<kmeans_dir_path>" # path to your target kmeans directorys
OUTPUT_FILE="kmeans.scp"
> $OUTPUT_FILE
# Loop through all files
find "$KMEANS_DIR" -type f | while read -r file; do
    abs_path=$(realpath "$file")
    filename=$(basename "$file")
    stem="${filename%.*}"
    echo "$stem $abs_path" >> "$OUTPUT_FILE"
done
```

## Run 

### Running with all kmeans models

1. go to repository root directory
2. Run:

```shell
python exp/kmeans_random_frame/inference_audio.py \
    --audio_scp <audio_scp> \
    --output_dir <output_dir> \
    --kmeans_scp <path_to_kmeans_scp> \
    --stride 1 # Controls the consecutive frames to use the same kmeans model
```

Note that `stride` is an `int` that controls the consecutive frames to use the same kmeans model.

### Running with randomly chosen number of kmeans models

Loading 1000ish kmeans models can take about 35 mins, therefore, you can also infer using randomly chosen kmeans models from the scp by specifying `num_kmeans` field.


1. go to repository root directory
2. Run

```shell
python exp/kmeans_random_frame/inference_audio.py \
    --audio_scp <audio_scp> \
    --output_dir <output_dir> \
    --kmeans_scp <path_to_kmeans_scp> \
    --stride 1 \
    --num_kmeans <number of kmeans models to choose from> # This field controls the number of kmeans models to randomly choose from during inference. 
```









