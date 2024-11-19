import argparse
import torch.multiprocessing as mp
import torch
import os
import sys

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

from utils import setup_seed
from utils import get_source_list

from models.wavlm.WavLMWrapper import WavLMWrapper as WavLM
from model import WavLMKmeansConformer
import torchaudio
import tqdm
import yaml
import random

SEED = 1234


def load_model(
    config_path: str, kmeans_path: str, hifi_config: str, ckpt_path: str, device: str
) -> WavLMKmeansConformer:
    if config_path.lower() == "none":
        model = WavLMKmeansConformer(kmeans_path=kmeans_path, hifi_config=hifi_config)
    else:
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
        model = WavLMKmeansConformer(
            **config, kmeans_path=kmeans_path, hifi_config=hifi_config
        )
    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt)
    model.to(device)
    model.eval()
    return model


def split_list(a: list, size: int):
    res = []
    for i in range(0, size):
        res.append(list(a[i::size]))
    return res


def inference(rank: int, args: argparse.Namespace):
    device = args.gpus[rank % len(args.gpus)]
    source_list = list(
        get_source_list(args.audio_scp)[rank :: args.num_proc]
    )  # list of audio paths
    random.shuffle(source_list)
    wavlm = WavLM(args.wavlm_ckpt).to(device)
    print(f"rank {rank} got data number {len(source_list)}")
    print(f"output directory {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)
    assert len(args.kmeans_path) == len(args.ckpt_path) and len(args.ckpt_path) == len(
        args.config
    )  # make sure kmeans_path, ckpt_path, config is the same length
    source_res = split_list(source_list, len(args.kmeans_path))
    for _kmeans, _ckpt, _config, _source in zip(
        args.kmeans_path, args.ckpt_path, args.config, source_res
    ):
        model = load_model(_config, _kmeans, args.hifi_config, _ckpt, device)
        with torch.no_grad():
            for s in tqdm.tqdm(_source):
                audio, rate = torchaudio.load(s)
                audio = audio.to(device)  # [1,T]
                audio_hat = model.inference_audio(audio, wavlm).cpu()
                filename = s.split("/")[-1].replace(".flac", ".wav")
                output_path = os.path.join(args.output_dir, filename)
                torchaudio.save(output_path, audio_hat, rate)
                pass
    pass


def main(args):
    # os.makedirs(args.)
    setup_seed(SEED)
    mp.spawn(inference, args=(args,), nprocs=args.num_proc, join=True)
    print("inference done")
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audio_scp",
        type=str,
        required=True,
        help="""
            The file has a list of source paths, where each path is on each line. It looks like:
            ...
            >>> path/to/wav/1.wav
            >>> path/to/wav/2.wav
            ...
            """,
    )
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--wavlm_ckpt", type=str, default="./ckpt/WavLM-Large.pt")
    parser.add_argument(
        "--kmeans_path",
        nargs="+",
        default=[
            "./ckpt/LibriSpeech_wavlm_k1000_L7.pt",
            "./ckpt/LJSpeech/kmeans-cluster-1024-k_1024.pt",
            "./ckpt/MSP-IMPROV/MSP_IMPROV_kmeans-cluster-1024-k_1024.pt",
        ],
    )
    parser.add_argument(
        "--ckpt_path",
        nargs="+",
        default=[
            "./ckpt/step160000_model.pth",
            "./ckpt/LJSpeech/LJSpeech_k_1024_model.pt",
            "./ckpt/MSP-IMPROV/MSP-Improv_k_1024_model.pt",
        ],
    )
    parser.add_argument(
        "--config",
        nargs="+",
        default=["none", "./ckpt/LJSpeech/K_1024.yaml", "none"],
        help="model config file, none stands for default",
    )
    parser.add_argument(
        "--hifi_config", type=str, default="./hifigan_config_v1_wavlm.json"
    )
    parser.add_argument(
        "--num_proc", type=int, default=8, help="total number of procedures"
    )
    parser.add_argument(
        "--gpus", nargs="+", default=["cuda:0", "cuda:1", "cuda:2", "cuda:3"]
    )
    args = parser.parse_args()
    main(args)
    pass
