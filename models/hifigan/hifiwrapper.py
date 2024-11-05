from .models import Generator
import torch
import json
import torch.nn as nn


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class HifiGan(nn.Module):
    def __init__(self, config, path=None):
        super().__init__()
        with open(config) as f:
            data = f.read()
        json_config = json.loads(data)
        h = AttrDict(json_config)
        model = Generator(h)
        if path is not None:
            ckpt = torch.load(path, map_location="cpu")
            model.load_state_dict(ckpt["generator"])
            model.eval()
        for p in model.parameters():
            p.requires_grad = False
        self.model = model

    def forward(self, x):
        """[B, T', E] -> [B, T] (audio)"""
        with torch.no_grad():
            output = self.model(x).squeeze(1)
        return output
