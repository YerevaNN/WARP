from typing import List, Optional, Tuple, Union
from overrides import overrides

from torch import Tensor
from torch.nn import Module, init
from torch.nn.modules.container import ParameterDict
from torch.nn.parameter import Parameter
from torch.nn.modules.sparse import Embedding

from allennlp.common.from_params import FromParams

from transformers.tokenization_utils import PreTrainedTokenizer

from .tokenizers import ArpTokenizer


class ArpInjector(Module, FromParams):
    def __init__(self,
                 embedder: Embedding,
                 tokenizer: PreTrainedTokenizer,
                 prompts: List[Union[int, str]] = None,
                 prompt_better_init: bool = False):
        super().__init__()

        self.embedder = embedder
        self.embedding_dim = embedder.embedding_dim

        if prompts is None:
            prompts = [50261, 50262, 50263]

        self.prompt_to_id, _ = ArpTokenizer.prepare_prompts(prompts, tokenizer=tokenizer)
        # We skip [MASK] tokens and treat the rest as prompts
        self.prompt_to_id.pop(tokenizer.mask_token, tokenizer.mask_token_id)

        self.prompt_params = ParameterDict({
            prompt: Parameter(Tensor(self.embedding_dim))
            for prompt in self.prompt_to_id
        })
        self.prompt_better_init = prompt_better_init
        self.reset_params()

    def reset_params(self):
        if self.prompt_better_init:
            mean = self.embedder.weight.mean()
            std = self.embedder.weight.std()
        else:
            mean = 0.
            std = 1.

        for param in self.prompt_params.values():
            init.normal_(param, mean=mean, std=std)

    @overrides
    def forward(self, input: Tensor) -> Tensor:
        embeddings = self.embedder(input)

        for prompt, idx in self.prompt_to_id.items():
            mask = (input == idx)
            embeddings[mask] = self.prompt_params[prompt]

        return embeddings
