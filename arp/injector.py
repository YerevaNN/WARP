from typing import List, Optional, Union
from overrides import overrides

import logging

import torch
from torch.nn import Module, init
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.sparse import Embedding
from torch.nn.modules.container import ParameterDict

from transformers.tokenization_utils import PreTrainedTokenizer

from allennlp.common import Lazy
from allennlp.common import FromParams
from allennlp.modules import FeedForward

from .tokenizers import ArpTokenizer, PromptTemplateConfig

logger = logging.getLogger(__name__)


class ArpInjector(Module, FromParams):
    def __init__(
        self,
        embedder: Embedding,
        tokenizer: PreTrainedTokenizer,
        prompts: PromptTemplateConfig,
        prompt_better_init: Union[bool, str] = False,
        frozen_prompts: bool = False,
        optimized_prompts: bool = False,
        reparameterization: FeedForward = None,
        dropout: float = None,
    ):
        super().__init__()

        self.embedder = embedder
        self.embedding_dim = embedder.embedding_dim
        self.tokenizer = tokenizer
        self.frozen_prompts = frozen_prompts
        self.prompt_params_reparam = reparameterization
        self.prompt_better_init = prompt_better_init

        self.prompts = prompts
        self.prompt_template = ArpTokenizer.prepare_prompts(
            prompts, tokenizer=tokenizer, default_init=prompt_better_init
        )
        self.prompt_to_id = self.prompt_template.to_id
        self.prompt_to_init = self.prompt_template.to_init_id

        if dropout:
            self.dropout = torch.nn.Dropout(dropout)
        else:
            self.dropout = None

        self.optimized_prompts: Optional[List[str]] = None
        if optimized_prompts:
            self.optimized_prompts = [
                prompt for prompt in self.prompt_to_id if prompt != tokenizer.mask_token
            ]

        self.prompt_params: Union[Parameter, ParameterDict]
        if self.optimized_prompts is not None:
            self.prompt_params = Parameter(
                torch.Tensor(len(self.optimized_prompts), self.embedding_dim)
            )
        else:
            self.prompt_params = ParameterDict(
                {
                    prompt: Parameter(torch.Tensor(self.embedding_dim))
                    for prompt in self.prompt_to_id
                    if prompt != tokenizer.mask_token
                }
            )
            if reparameterization:
                raise NotImplementedError

        if frozen_prompts:
            self.freeze_prompts()

        self.reset_params()

    def freeze_prompts(self):
        if self.optimized_prompts is not None:
            self.prompt_params.requires_grad = False
            return

        for key, param in self.prompt_params.items():
            param.requires_grad = False

    def reset_params(self):
        if self.prompt_better_init is True:
            mean = self.embedder.weight.mean().item()
            std = self.embedder.weight.std().item()
        else:
            mean = 0.0
            std = 1.0

        if self.optimized_prompts is not None:
            for idx, token in enumerate(self.optimized_prompts):
                init_with = self.prompt_to_init.get(token)
                # If no init-with-word is given we initialize as `Normal(mean, std)`
                if init_with is None:
                    init.normal_(self.prompt_params[idx], mean=mean, std=std)
                    continue
                # Otherewise we'll init it with word embeddings
                with torch.no_grad():
                    self.prompt_params[idx][:] = self.embedder.weight[init_with]
        else:
            for token, param in self.prompt_params.items():
                init_with = self.prompt_to_init.get(token)
                # If no init-with-word is given we initialize as `Normal(mean, std)`
                if init_with is None:
                    init.normal_(param, mean=mean, std=std)
                    continue
                # Otherewise we'll init it with word embeddings
                with torch.no_grad():
                    param.data[:] = self.embedder.weight[init_with]

    def optimized_forward(self, input: torch.Tensor) -> torch.Tensor:

        assert isinstance(self.prompt_params, Parameter)

        # Shape: (batch_size, num_tokens, embedding_dim)
        embeddings = self.embedder(input)

        # Shape: (num_prompts, embedding_dim)
        prompt_params = self.prompt_params
        if self.prompt_params_reparam is not None:
            prompt_params = F.relu(prompt_params)
            prompt_params = self.prompt_params_reparam(prompt_params)

        if self.dropout is not None:
            prompt_params = self.dropout(prompt_params)

        if len(self.prompt_params) > 0:
            embeddings[:, 2 : 2 + len(self.prompt_params)] = prompt_params

        return embeddings

    @overrides
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.optimized_prompts is not None:
            return self.optimized_forward(input)

        assert isinstance(self.prompt_params, ParameterDict)

        # Shape: (batch_size, num_tokens, embedding_dim)
        embeddings = self.embedder(input)

        for prompt, idx in self.prompt_to_id.items():
            if prompt == self.tokenizer.mask_token:
                continue
            # Shape: (batch_size, num_tokens)
            mask = input == idx
            # embeddings[mask] = self.prompt_params[prompt]
            embeddings = torch.where(
                mask.unsqueeze(dim=-1), self.prompt_params[prompt], embeddings
            )

        return embeddings
