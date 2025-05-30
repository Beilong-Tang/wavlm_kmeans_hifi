import argparse
import torch.multiprocessing as mp
import torch
import os
import sys
from pathlib import Path

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
sys.path.append(str(Path(os.path.abspath(__file__)).parent.parent.parent))

from utils import setup_seed
from utils import get_source_list

from models.wavlm.WavLMWrapper import WavLMWrapper as WavLM
from models.multiple_kmeans import MultipleKmeans
from model import WavLMKmeansConformer
import torchaudio
import tqdm
import yaml

SEED = 1234

def inference(rank: int, args: argparse.Namespace):
    device = args.gpus[rank % len(args.gpus)]
    source_list = get_source_list(args.audio_scp)[
        rank :: args.num_proc
    ]  # list of audio paths
    wavlm = WavLM(args.wavlm_ckpt).to(device)
    wavlm.eval()

    kmeans_path = get_source_list(args.kmeans_scp)[0] # This one does not matter
    if args.config is not None:
        with open(args.config, "r") as file:
            config = yaml.safe_load(file)
        model = WavLMKmeansConformer(**config, 
                                     kmeans_path=kmeans_path, 
                                     hifi_config=args.hifi_config)
    else:
        model = WavLMKmeansConformer(
            kmeans_path=kmeans_path, 
            hifi_config=args.hifi_config
        )
    ckpt = torch.load(args.ckpt_path)
    model.load_state_dict(ckpt)
    model.to(device)
    model.eval()

    multi_kmeans = MultipleKmeans(args.kmeans_scp, stride = args.stride, num_kmeans = args.num_kmeans)
    multi_kmeans.to(device)
    multi_kmeans.eval()

    print(f"rank {rank} got data number {len(source_list)}")
    print(f"output directory {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)
    with torch.no_grad():
        for s in tqdm.tqdm(source_list):
            s: str
            audio, rate = torchaudio.load(s)
            audio = audio.to(device)  # [1,T]
            emb = wavlm(audio) # [1,T,E]
            kmeans_emb = multi_kmeans.random_infer(emb) #[1,T,E]
            audio_hat = model.inference_emb(kmeans_emb).cpu() #
            filename = s.split("/")[-1].replace('.flac','.wav')
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
    # kmeans config
    parser.add_argument("--kmeans_scp", type=str, required=True, 
                        help="""
                             The file has a list of kmeans paths, where each path is on each line. It looks like:
                             ...
                             >>> kmeans_1 path/to/kmeans_1.pt
                             >>> kmeans_2 path/to/kmeans_2.pt
                             ...
                             """,)
    parser.add_argument("--num_kmeans", type = int, default= None, help='number of kmeans model to randomly choose from. If unspecified, use all kmeans models.')
    parser.add_argument("--stride", type = int, default = 1, help = "number of consecutive frames from the same kmeans model")
    parser.add_argument("--ckpt_path", type=str, default = "./ckpt/step160000_model.pth")
    parser.add_argument("--config", type=str, default = None, help = "model config file")
    parser.add_argument(
        "--hifi_config", type=str, default="./hifigan_config_v1_wavlm.json"
    )
    parser.add_argument(
        "--num_proc", type=int, default=4, help="total number of procedures"
    )
    parser.add_argument(
        "--gpus", nargs="+", default=["cuda:0", "cuda:1", "cuda:2", "cuda:3"]
    )
    args = parser.parse_args()
    print(args.gpus)
    main(args)
    pass
