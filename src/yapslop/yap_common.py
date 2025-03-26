from os import getenv

import torch

DEVICE: str = getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

# using dict to avoid unnecessary imports
initial_speakers = [
    {
        "name": "Seraphina",
        "description": "Tech entrepreneur. Uses technical jargon, speaks confidently",
    },
]
