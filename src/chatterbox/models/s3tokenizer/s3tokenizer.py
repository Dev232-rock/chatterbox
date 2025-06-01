from typing import List, Tuple, Union
import numpy as np
import librosa
import torch
import torch.nn.functional as F
from s3tokenizer.utils import padding
from s3tokenizer.model_v2 import (
    S3TokenizerV2,
    ModelConfig,
)

# Sampling rate of the inputs to S3TokenizerV2
S3_SR = 16_000
S3_HOP = 160  # 100 frames/sec
S3_TOKEN_HOP = 640  # 25 tokens/sec
S3_TOKEN_RATE = 25
SPEECH_VOCAB_SIZE = 6561


class S3Tokenizer(S3TokenizerV2):
    """
    A subclass of s3tokenizer.S3TokenizerV2 with improvements:
    - integrated `forward` for batch input
    - custom `log_mel_spectrogram` using registered buffers
    - padding to fit 25Hz token rate
    """

    ignore_state_dict_missing = ("_mel_filters", "window")

    def __init__(
        self,
        name: str = "speech_tokenizer_v2_25hz",
        config: ModelConfig = ModelConfig(),
        verbose: bool = False,
    ):
        super().__init__(name)

        self.n_fft = 400
        self.verbose = verbose

        _mel_filters = librosa.filters.mel(
            sr=S3_SR,
            n_fft=self.n_fft,
            n_mels=config.n_mels
        )
        self.register_buffer(
            "_mel_filters",
            torch.FloatTensor(_mel_filters),
        )

        self.register_buffer(
            "window",
            torch.hann_window(self.n_fft),
        )

    def pad(self, wavs: List[Union[np.ndarray, torch.Tensor]], sr: int) -> List[torch.Tensor]:
        """
        Pads each waveform so its length is a multiple of 40ms (25 tokens/sec).

        Args:
            wavs: List of waveforms (np.ndarray or torch.Tensor)
            sr: Sampling rate (should be 16kHz)

        Returns:
            List[torch.Tensor]: Padded waveforms
        """
        processed_wavs = []
        for wav in wavs:
            if isinstance(wav, np.ndarray):
                wav = torch.from_numpy(wav)
            if wav.dim() == 1:
                wav = wav.unsqueeze(0)

            if sr != S3_SR:
                raise ValueError(f"Expected sampling rate {S3_SR}, got {sr}")

            n_tokens = np.ceil((wav.shape[1] / sr) * S3_TOKEN_RATE)
            intended_wav_len = int(n_tokens * (sr / S3_TOKEN_RATE))
            wav = F.pad(wav, (0, intended_wav_len - wav.shape[-1]), mode="constant", value=0)
            processed_wavs.append(wav)
        return processed_wavs

    def _prepare_audio(self, wavs: List[Union[np.ndarray, torch.Tensor]]) -> List[torch.Tensor]:
        """Normalize and shape audio for processing."""
        processed_wavs = []
        for wav in wavs:
            if isinstance(wav, np.ndarray):
                wav = torch.from_numpy(wav)
            if wav.dim() == 1:
                wav = wav.unsqueeze(0)
            # Optional: normalize
            wav = wav - wav.mean()
            wav = wav / (wav.std() + 1e-9)
            processed_wavs.append(wav)
        return processed_wavs

    @torch.no_grad()
    def forward(
        self,
        wavs: Union[List[Union[np.ndarray, torch.Tensor]], torch.Tensor],
        accelerator: 'Accelerator' = None,
        max_len: int = None,
    ) -> Tuple[torch.Tensor, torch.LongTensor]:
        """
        Forward pass to compute speech tokens from input audio.

        Args:
            wavs: Batch of waveforms at 16kHz (list of arrays or torch.Tensor)
            accelerator: Accelerator instance for DDP (optional)
            max_len: Max number of output tokens to keep (truncates spectrograms)

        Returns:
            Tuple of:
                - speech_tokens: [B, T]
                - speech_token_lens: [B]
        """
        processed_wavs = self._prepare_audio(wavs)
        mels = []
        for wav in processed_wavs:
            wav = wav.to(self.device)
            mel = self.log_mel_spectrogram(wav)
            if max_len is not None:
                mel = mel[..., :max_len * 4]  # 4 mel-frames per token
            mels.append(mel.squeeze(0))  # [F, T]

            if self.verbose:
                print(f"[DEBUG] Mel shape: {mel.shape}")

        mels, mel_lens = padding(mels)
        tokenizer = self if accelerator is None else accelerator.unwrap_model(self)

        speech_tokens, speech_token_lens = tokenizer.quantize(mels, mel_lens.to(self.device))
        return speech_tokens.long().detach(), speech_token_lens.long().detach()

    def log_mel_spectrogram(
        self,
        audio: torch.Tensor,
        padding: int = 0,
    ) -> torch.Tensor:
        """
        Compute the log-Mel spectrogram of the input audio.

        Args:
            audio: 1D or 2D audio tensor (B=1, T)
            padding: Right-side zero padding (optional)

        Returns:
            log-Mel spectrogram: [1, n_mels, n_frames]
        """
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        audio = audio.to(self.device)
        if padding > 0:
            audio = F.pad(audio, (0, padding))

        stft = torch.stft(
            audio,
            self.n_fft,
            S3_HOP,
            window=self.window.to(self.device),
            return_complex=True
        )
        magnitudes = stft[..., :-1].abs() ** 2
        mel_spec = self._mel_filters.to(self.device) @ magnitudes
        log_spec = torch.clamp(mel_spec, min=1e-10).log10()
        log_spec = torch.maximum(log_spec, log_spec.amax(dim=-1, keepdim=True) - 8.0)
        log_spec = (log_spec + 4.0) / 4.0
        return log_spec

    def inverse(self, tokens: torch.LongTensor) -> torch.Tensor:
        """
        (Optional placeholder) Inverse tokens back to waveform — not implemented.

        Args:
            tokens: [B, T] speech tokens

        Returns:
            torch.Tensor: reconstructed audio (NotImplemented)
        """
        raise NotImplementedError("Inverse token-to-audio not implemented.")
