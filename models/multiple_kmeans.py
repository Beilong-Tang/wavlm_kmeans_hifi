import tqdm

import torch
import random
import torch.nn as nn

from utils import get_source_list
from models.kmeans import KMeansQuantizer


class MultipleKmeans(nn.Module):
    def __init__(self, kmeans_scp, stride=1, num_kmeans = None):
        """
        Arguments:
            kmeans_scp: path to kmeans scp
            stride: determine number of consecutive frames from the same kmeans model
            num_kmeans: If specified, use certain number of randomly chosen kmeans models to infer. Default: None. Use all kmeans model. 
        """
        super().__init__()

        kmeans_path = get_source_list(kmeans_scp)
        if num_kmeans is not None:
            random.shuffle(kmeans_path)
            kmeans_path = kmeans_path[:num_kmeans]
        kmeans_list = []
        print(f"loading kmeans model of length {len(kmeans_path)}")
        for _p in tqdm.tqdm(kmeans_path, desc="[Loading Kmeans Model]"):
            kmeans_list.append(KMeansQuantizer(_p))
        self.kmeans_list = nn.ModuleList(kmeans_list)
        self.stride = stride

    def random_infer(self,emb):
        """
        Arguments:
            emb: wavlm_emb [1,T,E]
        Returns:
            kmeans_discrete_emb: [1,T,E]
        """
        res = []
        for i in range(0, emb.size(1), self.stride):
            _single_emb = emb[:,i:i+self.stride] # [1,T',E]
            kmeans_model = self.kmeans_list[random.randint(0, len(self.kmeans_list)-1)] # randomly select kmeans model
            out = kmeans_model.emb(kmeans_model(_single_emb)) # [1,T',E]
            res.append(out)
        return torch.cat(res, dim=1) # [1,T,E]
