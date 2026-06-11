import torch
import torch.nn.functional as F
from torch import Tensor

from .shared_knowledge_base import SharedKnowledgeBase


class Agent:
    """Stigmergic agent operating over a shared token knowledge base.

    Per round the agent (1) absorbs a local observation, (2) reads the field's
    dominant token and pulls its belief toward that category's direction,
    (3) tokenizes its updated belief to the nearest knowledge-base category, and
    (4) selectively deposits that token.

    Read-before-deposit: the agent deposits the token of its belief *after*
    integrating the field. Depositing the pre-read (stale) token instead
    creates a phase lag that drives a limit-cycle oscillation of the dominant
    token under synchronous updates; reading first removes it and yields clean
    winner-take-all convergence.

    Selective-deposit note: the send/silent decision compares the agent's
    *continuous* belief to the dominant token's direction. With synthetic
    near-orthogonal categories a token-vs-token distance is degenerate
    (0 or 1); using the continuous belief makes theta a graded knob. The
    deposited symbol itself is still the tokenized belief.
    """

    def __init__(self, agent_id: int, kb: SharedKnowledgeBase,
                 beta: float = 0.05,
                 alpha: float = 0.1,
                 theta: float = 0.3):
        self.id = agent_id
        self.kb = kb
        self.beta = beta    # observation weight
        self.alpha = alpha  # field reading weight
        self.theta = theta  # gating threshold (cosine distance)
        self.belief = torch.zeros(kb.dim)

    def _cosine_distance(self, a: Tensor, b: Tensor) -> float:
        return 1.0 - F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()

    def step(self, obs: Tensor, dominant: int | None) -> int | None:
        """One round update. Returns the deposited token index, or None."""
        # (1) absorb local observation
        if self.beta > 0:
            self.belief = (1 - self.beta) * self.belief + self.beta * obs
            self.belief = F.normalize(self.belief, dim=0)

        # (2) read dominant token, pull belief toward it
        if dominant is not None:
            self.belief = (1 - self.alpha) * self.belief + self.alpha * self.kb.embed(dominant)
            self.belief = F.normalize(self.belief, dim=0)

        # (3) tokenize updated belief
        my_token = self.kb.tokenize(self.belief)

        # (4) selective deposit (continuous belief vs dominant direction)
        if dominant is None:
            deposit = my_token  # field empty -> seed it
        elif self._cosine_distance(self.belief, self.kb.embed(dominant)) > self.theta:
            deposit = my_token
        else:
            deposit = None

        return deposit
