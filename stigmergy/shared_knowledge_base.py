import torch
import torch.nn.functional as F
from torch import Tensor

# Category names (CIFAR-10) — used only as human-readable labels.
CIFAR10 = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]


class SharedKnowledgeBase:
    """Shared, immutable vocabulary of category directions.

    Categories are synthetic random unit vectors (no language model), fixed by
    a seed so every run sees the same knowledge base. All agents and the field
    share this single table.
    """

    def __init__(self, num_tokens: int = 10, dim: int = 512,
                 seed: int = 0, names: list[str] | None = None,
                 orthonormal: bool = False):
        g = torch.Generator().manual_seed(seed)
        E = F.normalize(torch.randn(num_tokens, dim, generator=g), dim=1)  # (M, dim)
        if orthonormal:
            # exactly orthonormal category directions (requires num_tokens <= dim);
            # cross-category cosine becomes exactly 0 instead of a small residual.
            q, _ = torch.linalg.qr(E.t())   # (dim, M) orthonormal columns
            E = q.t()
        self.E = E
        self.M, self.dim = self.E.shape
        self.names = names if names is not None else [f"c{i}" for i in range(num_tokens)]

    def tokenize(self, belief: Tensor) -> int:
        """Nearest category index by cosine similarity."""
        return int((self.E @ F.normalize(belief, dim=0)).argmax())

    def tokenize_batch(self, B: Tensor) -> Tensor:
        """Vectorized tokenize for a batch of beliefs (N, dim) -> (N,)."""
        return (F.normalize(B, dim=1) @ self.E.t()).argmax(dim=1)

    def embed(self, token: int) -> Tensor:
        """Unit direction of a category."""
        return self.E[token]
