import os

import torch
import torchaudio
from huggingface_hub import hf_hub_download

from csm.generator import load_csm_1b

out_dir = "audio_outputs"
os.makedirs(out_dir, exist_ok=True)


model_path = hf_hub_download(repo_id="sesame/csm-1b", filename="ckpt.pt")
generator = load_csm_1b(model_path, "cuda" if torch.cuda.is_available() else "cpu")

audio = generator.generate(
    text="Hello from yapslop.",
    speaker=0,
    context=[],
    max_audio_length_ms=10_000,
)

torchaudio.save(f"{out_dir}/audio.wav", audio.unsqueeze(0).cpu(), generator.sample_rate)
