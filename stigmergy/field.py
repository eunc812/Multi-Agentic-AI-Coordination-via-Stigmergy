import torch
from torch import Tensor


class SharedField:
    """Passive shared field: a frequency table over knowledge-base tokens.

    It does not aggregate vectors. Agents deposit token indices; the field
    tallies their frequency, fades old counts by a decay factor each round,
    and exposes the most frequent token as the current *dominant* token.
    """

    def __init__(self, num_tokens: int, decay: float = 0.9):
        self.counts = torch.zeros(num_tokens)
        self.decay = decay  # keep-fraction per round (framework example: 0.8)

    def is_empty(self) -> bool:
        return self.counts.max().item() < 1e-8

    def dominant(self) -> int | None:
        """Most frequent token, or None if the field is empty."""
        return None if self.is_empty() else int(self.counts.argmax())

    def deposit(self, tokens) -> None:
        """Tally a batch of deposited token indices (list or 1-D tensor)."""
        if tokens is None or len(tokens) == 0:
            return
        idx = torch.as_tensor(tokens, dtype=torch.long)
        self.counts += torch.bincount(idx, minlength=self.counts.numel()).float()

    def evaporate(self) -> None:
        """Fade all token frequencies by the decay factor."""
        self.counts *= self.decay
