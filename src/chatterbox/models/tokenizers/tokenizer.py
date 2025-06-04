import logging
import torch
from tokenizers import Tokenizer
from typing import List, Union

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Special tokens
SOT = "[START]"
EOT = "[STOP]"
UNK = "[UNK]"
SPACE = "[SPACE]"
SPECIAL_TOKENS = [SOT, EOT, UNK, SPACE, "[PAD]", "[SEP]", "[CLS]", "[MASK]"]

class EnTokenizer:
    def __init__(self, vocab_file_path: str):
        self.tokenizer: Tokenizer = Tokenizer.from_file(vocab_file_path)
        self.check_vocabset_special_tokens()

    def check_vocabset_special_tokens(self):
        """Ensure required special tokens exist in vocab."""
        vocab = self.tokenizer.get_vocab()
        for token in [SOT, EOT]:
            if token not in vocab:
                raise ValueError(f"Required special token '{token}' not found in vocabulary.")
        logger.info("All required special tokens are present in the vocabulary.")

    def encode(self, txt: str, add_special_tokens: bool = True, verbose: bool = False) -> List[int]:
        """
        Clean and encode a single string into token IDs.
        Optionally add special start and stop tokens.
        """
        original_txt = txt
        txt = txt.replace(' ', SPACE)
        if add_special_tokens:
            txt = f"{SOT} {txt} {EOT}"

        if verbose:
            logger.debug(f"Original text: {original_txt}")
            logger.debug(f"Preprocessed text: {txt}")

        code = self.tokenizer.encode(txt)
        ids = code.ids

        if verbose:
            logger.debug(f"Token IDs: {ids}")

        return ids

    def decode(self, seq: Union[List[int], torch.Tensor], skip_special_tokens: bool = False) -> str:
        """
        Decode a sequence of token IDs back into a human-readable string.
        """
        if isinstance(seq, torch.Tensor):
            seq = seq.cpu().numpy().tolist()

        txt = self.tokenizer.decode(seq, skip_special_tokens=skip_special_tokens)
        txt = txt.replace(' ', '')  # remove tokenizer-introduced spaces
        txt = txt.replace(SPACE, ' ')
        txt = txt.replace(EOT, '')
        txt = txt.replace(SOT, '')
        txt = txt.replace(UNK, '[UNK]')
        return txt.strip()

    def text_to_tokens(self, text: str) -> torch.Tensor:
        """
        Convert raw input text to a PyTorch tensor of token IDs.
        """
        token_ids = self.encode(text)
        return torch.IntTensor(token_ids).unsqueeze(0)

    def tokens_to_tensor(self, tokens: List[int]) -> torch.Tensor:
        return torch.IntTensor(tokens).unsqueeze(0)

    def tensor_to_tokens(self, tensor: torch.Tensor) -> List[int]:
        return tensor.squeeze(0).tolist()

    def tokens_to_text(self, tokens: Union[List[int], torch.Tensor]) -> str:
        return self.decode(tokens)

    def batch_encode(self, texts: List[str], add_special_tokens: bool = True) -> List[List[int]]:
        return [self.encode(txt, add_special_tokens=add_special_tokens) for txt in texts]

    def batch_decode(self, sequences: List[Union[List[int], torch.Tensor]], skip_special_tokens: bool = False) -> List[str]:
        return [self.decode(seq, skip_special_tokens=skip_special_tokens) for seq in sequences]

    def __call__(self, text: str) -> torch.Tensor:
        return self.text_to_tokens(text)
