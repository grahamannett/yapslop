from typing import Generator as GeneratorType

import torch
import torchaudio

from yapslop.convo_dto import Segment

from csm.generator import (
    CSM_1B_GH_WATERMARK,
    Model,
    hf_hub_download,
    load_llama3_tokenizer,
    load_watermarker,
    loaders,
    watermark,
)
from csm.generator import Generator as CSMGenerator


class Generator(CSMGenerator):
    use_wm: bool = False

    def __init__(
        self,
        model: Model,
    ):
        self._model = model
        self._model.setup_caches(1)

        self._text_tokenizer = load_llama3_tokenizer()

        device = next(model.parameters()).device
        mimi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
        mimi = loaders.get_mimi(mimi_weight, device=device)
        mimi.set_num_codebooks(32)
        self._audio_tokenizer = mimi
        # this is the only real difference between the two generators
        self._watermarker = load_watermarker(device=device) if self.use_wm else None

        self.sample_rate = mimi.sample_rate
        self.device = device

    def watermark(self, audio: torch.Tensor) -> torch.Tensor:
        # This applies an imperceptible watermark to identify audio as AI-generated.
        # Watermarking ensures transparency, dissuades misuse, and enables traceability.
        # Please be a responsible AI citizen and keep the watermarking in place.
        # If using CSM 1B in another application, use your own private key and keep it secret.
        audio, wm_sample_rate = watermark(self._watermarker, audio, self.sample_rate, CSM_1B_GH_WATERMARK)
        return torchaudio.functional.resample(audio, orig_freq=wm_sample_rate, new_freq=self.sample_rate)

    @torch.inference_mode()
    def generate(
        self,
        text: str,
        speaker: int,
        context: list[Segment],
        max_audio_length_ms: float = 90_000,
        temperature: float = 0.9,
        topk: int = 50,
    ) -> torch.Tensor:
        self._model.reset_caches()

        max_audio_frames = int(max_audio_length_ms / 80)
        tokens, tokens_mask = [], []
        for segment in context:
            segment_tokens, segment_tokens_mask = self._tokenize_segment(segment)
            tokens.append(segment_tokens)
            tokens_mask.append(segment_tokens_mask)

        gen_segment_tokens, gen_segment_tokens_mask = self._tokenize_text_segment(text, speaker)
        tokens.append(gen_segment_tokens)
        tokens_mask.append(gen_segment_tokens_mask)

        prompt_tokens = torch.cat(tokens, dim=0).long().to(self.device)
        prompt_tokens_mask = torch.cat(tokens_mask, dim=0).bool().to(self.device)

        samples = []
        curr_tokens = prompt_tokens.unsqueeze(0)
        curr_tokens_mask = prompt_tokens_mask.unsqueeze(0)
        curr_pos = torch.arange(0, prompt_tokens.size(0)).unsqueeze(0).long().to(self.device)

        # max_seq_len = 2048 - max_audio_frames # had tried with flipped but idk
        max_seq_len = max_audio_frames  # if its less than 163_840 this will be <0?
        if curr_tokens.size(1) >= max_seq_len:
            raise ValueError(f"Inputs too long, must be below max_seq_len - max_audio_frames: {max_seq_len}")

        for _ in range(max_audio_frames):
            sample = self._model.generate_frame(curr_tokens, curr_tokens_mask, curr_pos, temperature, topk)
            if torch.all(sample == 0):
                break  # eos

            samples.append(sample)

            curr_tokens = torch.cat([sample, torch.zeros(1, 1).long().to(self.device)], dim=1).unsqueeze(1)
            curr_tokens_mask = torch.cat(
                [torch.ones_like(sample).bool(), torch.zeros(1, 1).bool().to(self.device)], dim=1
            ).unsqueeze(1)
            curr_pos = curr_pos[:, -1:] + 1

        audio = self._audio_tokenizer.decode(torch.stack(samples).permute(1, 2, 0)).squeeze(0).squeeze(0)

        if self.use_wm:
            audio = self.watermark(audio)

        return audio

    @torch.inference_mode()
    def stream_generate(
        self,
        text: str,
        speaker: int,
        context: list[Segment],
        max_audio_length_ms: float = 90_000,
        temperature: float = 0.9,
        topk: int = 50,
        chunk_size: int = 1000,  # Size of audio chunks to yield in milliseconds
    ) -> GeneratorType[torch.Tensor, None, None]:
        """Stream audio generation, yielding chunks as they're generated.

        Args:
            text: Text to generate audio for
            speaker: Speaker ID
            context: List of previous conversation segments
            max_audio_length_ms: Maximum length of audio to generate
            temperature: Sampling temperature
            topk: Top-k sampling parameter
            chunk_size: Size of audio chunks to yield in milliseconds

        Yields:
            torch.Tensor: Audio chunks of size chunk_size
        """
        pass


def load_csm_1b(device: str = "cuda") -> Generator:
    model = Model.from_pretrained("sesame/csm-1b")
    model.to(device=device, dtype=torch.bfloat16)

    generator = Generator(model)
    return generator
