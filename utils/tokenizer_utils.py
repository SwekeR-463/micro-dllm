from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer


SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[MASK]"]


def train_bpe_tokenizer(
    input_path: str = "data/stories.txt",
    tokenizer_path: str = "artifacts/tokenizer/tokenizer.json",
    vocab_size: int = 2048,
    min_frequency: int = 2,
) -> Tokenizer:
    src = Path(input_path)
    if not src.exists():
        raise FileNotFoundError(f"Tokenizer training data not found: {src}")

    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=SPECIAL_TOKENS,
    )
    tokenizer.train(files=[str(src)], trainer=trainer)
    tokenizer.save(tokenizer_path)
    return tokenizer


def load_tokenizer(tokenizer_path: str = "artifacts/tokenizer/tokenizer.json") -> Tokenizer:
    path = Path(tokenizer_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Tokenizer file not found: {path}. Run train_tokenizer.py first."
        )
    return Tokenizer.from_file(str(path))
