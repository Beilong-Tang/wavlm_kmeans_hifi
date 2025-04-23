import argparse
# import torch
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
import torch.multiprocessing as mp
import glob

SEED = 1234


def setup_seed(seed, rank=0):
    SEED = int(seed) + rank
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return SEED


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_path",
        type=str,
        required=True,
        help="""
            The base path of the Librispeech Train directory
            """,
    )
    parser.add_argument("--spk_list", nargs="+", help="List of spk ids to train Kmeans on")
    parser.add_argument("--config", type=str, required=True, help="The config path specifically for kmeans")
    parser.add_argument(
        "--ckpt_dir", type=str, default = None, help="The checkpoint directory, and it can be None"
    )
    ## WavLM Related
    parser.add_argument(
        "--wavlm_ckpt",
        default=True,
    )
    ## DDP Related
    parser.add_argument(
        "--num_proc", type=int, default=8, help="total number of procedures to run tasks in parallel"
    )
    parser.add_argument(
        "--gpus", nargs="+", default=["cuda:0", "cuda:1", "cuda:2", "cuda:3"]
    )

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
    if args.num_proc > 1:
        mp.spawn(run, args=(args,), nprocs=args.num_proc, join = True)
    else:
        run(0, args)
    pass


def _get_librispeech_spk_utterance(base_path, spk:int):
    return glob.glob(f"{base_path}/*/{spk}/*/*.flac")

def run(rank, args):
    spks = args.spk_list[rank::args.num_proc] # 
    device = args.gpus[rank % len(args.gpus)] # The device to run experiments on 
    print(f"rank {rank} processing spks: {spks} on device {device}")

    ## Load WavLM Model ## 
    wavlm = WavLM(args.wavlm_ckpt)
    wavlm.to(device)

    for _spk in spks:
        scps = _get_librispeech_spk_utterance(args.base_path, int(_spk))
        print(f"processing spk {_spk} with num {len(scps)}... on rank {rank}")
        ## Shuffle it for randomness
        shuffle(scps)
        
        with open(args.config, "r") as file:
            config: dict = yaml.safe_load(file)

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
            ssl_model = wavlm,
            device = device,
            kmeans_batch_size=config["batch_size"],
        )

if __name__ == "__main__":
    args = parse_args()
    main(args)
    pass
