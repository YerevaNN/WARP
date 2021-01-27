from overrides import overrides

import torch

from allennlp.modules.seq2vec_encoders import Seq2VecEncoder
from allennlp.nn.util import get_final_encoder_states


@Seq2VecEncoder.register("at")
class AtIndexPooler(Seq2VecEncoder):
    def __init__(self, embedding_dim: int, index: int = 1):
        super().__init__()
        self._embedding_dim = embedding_dim
        self._index = index

    @overrides
    def get_input_dim(self) -> int:
        return self._embedding_dim

    @overrides
    def get_output_dim(self) -> int:
        return self._embedding_dim

    @overrides
    def forward(self, tokens: torch.Tensor, mask: torch.BoolTensor = None):
        # `tokens` is assumed to have shape (batch_size, sequence_length, embedding_dim).
        # `mask` is assumed to have shape (batch_size, sequence_length) with all 1s preceding all 0s.
        if self._index >= 0:
            return tokens[:, self._index, :]
        else:  # [CLS] at the end
            if mask is None:
                raise ValueError(
                    "Must provide mask for [MASK] tokens with fixed negative index."
                )
            if self._index != -1:
                raise NotImplementedError(
                    "Only -1 negative index is supported as of now."
                )
            return get_final_encoder_states(tokens, mask)
