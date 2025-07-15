import argparse
import os
import sys

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

from models.wavlm.WavLMWrapper import WavLMWrapper as WavLM

import kmeans.kmeans_utils as K
import random
import torch
import numpy as np
import yaml
from random import shuffle

import glob

SEED = 1234

def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path", type=str, required=True, help="Base path of the Librispeech Train directory")
    parser.add_argument("--spk_list", nargs="+", help="List of spk ids to train Kmeans on")
    parser.add_argument("--config", type=str, required=True, help="Config path for kmeans")
    parser.add_argument("--ckpt_dir", type=str, default=None, help="Checkpoint directory (optional)")
    parser.add_argument("--wavlm_ckpt", default=True, help="WavLM checkpoint")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use, e.g., cuda:0 or cpu")
    parser.add_argument("--bs", type=str, default=1)

    return parser.parse_args()

def main(args):
    setup_seed(SEED)

    if args.ckpt_dir is None:
        args.ckpt_dir = os.path.join(
            os.path.dirname(os.path.dirname(args.config)),
            "ckpt",
            os.path.basename(args.config).replace(".yaml", ""),
        )
    os.makedirs(args.ckpt_dir, exist_ok=True)

    run(args)

def _get_librispeech_spk_utterance(base_path, spk: int):
    return glob.glob(f"{base_path}/{spk}/*/*.flac")
    #return glob.glob(f"{base_path}/*/{spk}/*/*.flac")

def run(args):
    spks = args.spk_list
    device = args.device
    print(f"Processing spks: {spks} on device {device}")

    wavlm = WavLM(args.wavlm_ckpt)
    wavlm.to(device)

    with open(args.config, "r") as file:
        config: dict = yaml.safe_load(file)


    shuffled_spks = spks.copy()
    shuffle(shuffled_spks)  # Shuffle speaker order

    scps = []
    batch_size = int(args.bs)
    for i in range(0, len(shuffled_spks), batch_size):
        batch = shuffled_spks[i:i + batch_size]
        print(f"Processing random batch {i//batch_size + 1}: {batch}")
        

        for _spk in batch:
            new_scps = _get_librispeech_spk_utterance(args.base_path, _spk)
            print(f"Processing spk {_spk} with {len(new_scps)} utterances...")
            scps.extend(new_scps)
        
        shuffle(scps)
        print(len(scps))
        kmeans_model = K.fetch_kmeans_model(
            n_clusters=config["n_clusters"],
            init=config["init"],
            max_iter=config["max_iter"],
            batch_size=config["batch_size"],
            tol=config["tol"],
            max_no_improvement=config["max_no_improvement"],
            n_init=config["n_init"],
            reassignment_ratio=config["reassignment_ratio"],
            random_state=SEED,
            checkpoint_path=os.path.join(
                args.ckpt_dir, f"kmeans-cluster-{config['n_clusters']}-{_spk}.pt"
            ),
        )

        K.train(
            kmeans_model,
            scps,
            args.ckpt_dir,
            _spk,
            ssl_model=wavlm,
            device=device,
            kmeans_batch_size=config["batch_size"],
        )

if __name__ == "__main__":
    args = parse_args()
    main(args)
