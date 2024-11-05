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

SEED = 1234

def inference(rank: int, args: argparse.Namespace):
    device = args.gpus[rank % len(args.gpus)]
    source_list = get_source_list(args.audio_scp)[
        rank :: args.num_proc
    ]  # list of audio paths
    wavlm = WavLM(args.wavlm_ckpt).to(device)
    model = WavLMKmeansConformer(
        kmeans_path=args.kmeans_path, hifi_config=args.hifi_config
    )
    ckpt = torch.load(args.ckpt_path)
    model.load_state_dict(ckpt)
    model.to(device)
    model.eval()
    print(f"rank {rank} got data number {len(source_list)}")
    print(f"output directory {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)
    with torch.no_grad():
        for s in tqdm.tqdm(source_list):
            s: str
            audio, rate = torchaudio.load(s)
            audio = audio.to(device)  # [1,T]
            audio_hat = model.inference_audio(audio, wavlm).cpu()
            output_path = os.path.join(args.output_dir, s.split("/")[-1])
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
    parser.add_argument("--kmeans_path", type=str, default="./ckpt/LibriSpeech_wavlm_k1000_L7.pt")
    parser.add_argument("--ckpt_path", type=str, default = "./ckpt/step160000_model.pth")
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
