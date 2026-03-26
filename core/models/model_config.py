from typing import Optional, Dict


class ModelConfig:
    """
    Configuration for the plainLM Model.

    Args:
        vocab_size:                      Vocabulary size.
        dim:                             Hidden / model dimension.
        n_layers:                        Number of transformer blocks.
        n_heads:                         Number of heads used by the sequence mixer.
        seq_len:                         Context length (not required for pure RNNs).
        state_mixer:                     MLP class: 'mlp', 'glu', or 'mlp_relu_sq'.
        state_mixer_config:              Extra kwargs forwarded to the state-mixer constructor.
                                         Must include 'hidden_dim'.
        sequence_mixer:                  Primary sequence-mixer key (see SEQUENCE_MIXERS in model.py).
        sequence_mixer_config:           Extra kwargs forwarded to the primary mixer constructor.
        secondary_sequence_mixer:        Optional secondary mixer key for hybrid architectures.
        secondary_sequence_mixer_config: Extra kwargs for the secondary mixer constructor.
        secondary_every:                 Period of secondary-mixer layers (1-indexed layer count).
                                         E.g. secondary_every=2 places secondary mixers at
                                         0-indexed layers 1, 3, 5, …
        tie_embeddings:                  Share lm_head weights with the token embedding table.
        rmsnorm_eps:                     Epsilon for all RMSNorm layers.
        use_rope:                        Legacy flag.  FLA attention layers manage their own
                                         rotary embeddings internally; this flag is kept for
                                         backward compatibility but has no effect on FLA layers.
    """

    def __init__(
        self,
        vocab_size: int,
        dim: int,
        n_layers: int,
        n_heads: int,
        seq_len: Optional[int] = None,
        state_mixer: str = 'glu',
        state_mixer_config: Optional[Dict] = None,
        sequence_mixer: str = 'gdn',
        sequence_mixer_config: Optional[Dict] = None,
        secondary_sequence_mixer: Optional[str] = None,
        secondary_sequence_mixer_config: Optional[Dict] = None,
        secondary_every: Optional[int] = None,
        tie_embeddings: bool = False,
        rmsnorm_eps: float = 1e-6,
        use_rope: bool = False,
    ):
        self.vocab_size = vocab_size
        self.dim = dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.seq_len = seq_len

        self.state_mixer = state_mixer
        self.state_mixer_config = state_mixer_config or {}

        self.sequence_mixer = sequence_mixer
        self.sequence_mixer_config = sequence_mixer_config or {}

        self.secondary_sequence_mixer = secondary_sequence_mixer
        self.secondary_sequence_mixer_config = secondary_sequence_mixer_config or {}
        self.secondary_every = secondary_every

        self.tie_embeddings = tie_embeddings
        self.rmsnorm_eps = rmsnorm_eps
        self.use_rope = use_rope

    @property
    def head_dim(self) -> int:
        """Dimension per head."""
        if self.dim % self.n_heads != 0:
            raise ValueError(
                f"dim={self.dim} must be divisible by n_heads={self.n_heads}"
            )
        return self.dim // self.n_heads
