from typing import Any, Dict, Optional, Union
from overrides import overrides

import logging

import torch

from allennlp.common.lazy import Lazy
from allennlp.data.tokenizers import PretrainedTransformerTokenizer
from allennlp.modules.token_embedders.token_embedder import TokenEmbedder

from transformers import AutoModelForMaskedLM, XLNetConfig

from .injector import ArpInjector

logger = logging.getLogger(__name__)


@TokenEmbedder.register("pretrained_mlm")
class PretrainedMLM(TokenEmbedder):

    authorized_missing_keys = [r"position_ids$"]

    def __init__(
        self,
        model_name: str,
        *,
        max_length: int = None,
        train_parameters: Union[bool, str] = True,
        arp_injector: Union[Lazy[ArpInjector], ArpInjector],
        on_logits: bool = False,
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
        transformer_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()

        if transformer_kwargs is None:
            transformer_kwargs = {}

        self.transformer_model_for_mlm = AutoModelForMaskedLM.from_pretrained(
            model_name,
            **transformer_kwargs,
        )

        self.config = self.transformer_model_for_mlm.config
        self._max_length = max_length
        self.output_dim = self.config.hidden_size

        tokenizer = PretrainedTransformerTokenizer(
            model_name,
            tokenizer_kwargs=tokenizer_kwargs,
        )
        self._num_added_start_tokens = len(tokenizer.single_sequence_start_tokens)
        self._num_added_end_tokens = len(tokenizer.single_sequence_end_tokens)
        self._num_added_tokens = self._num_added_start_tokens + self._num_added_end_tokens

        self.train_parameters = train_parameters
        if train_parameters is not True:
            for key, param in self.named_parameters():
                last_layer_name = f'layer.{self.num_hidden_layers - 1}.attention'
                if train_parameters != 'only_prompts':
                    if train_parameters == 'last_layer_only' and last_layer_name in key or '.lm_head.' in key:
                        continue
                param.requires_grad = False

        # Now, let's inject ARP if neccessary
        transformer_embeddings = self.embeddings_layer
        if isinstance(arp_injector, Lazy):
            arp_injector_ = arp_injector.construct(embedder=transformer_embeddings.word_embeddings,
                                                   tokenizer=tokenizer.tokenizer)
        else:
            arp_injector_ = arp_injector
        transformer_embeddings.word_embeddings = arp_injector_

        self.config.output_hidden_states = True
        self.on_logits = on_logits

    @overrides
    def state_dict(self, destination, prefix, keep_vars):
        states = super().state_dict(prefix=prefix, keep_vars=keep_vars)

        logger.warning(f'saving {self.train_parameters}')
        if self.train_parameters == 'only_prompts':
            keys_to_remove = [key for key in states if '.prompt_params.' not in key]
            for key in keys_to_remove:
                del states[key]

        for key, value in states.items():
            destination[key] = value

        return destination

    # def state_dict(self, destination=None, prefix='', keep_vars=False):
    #     if destination is None:
    #         destination = OrderedDict()
    #         destination._metadata = OrderedDict()
    #     destination._metadata[prefix[:-1]] = local_metadata = dict(version=self._version)
    #     self._save_to_state_dict(destination, prefix, keep_vars)
    #     for name, module in self._modules.items():
    #         if module is not None:
    #             module.state_dict(destination, prefix + name + '.', keep_vars=keep_vars)
    #     for hook in self._state_dict_hooks.values():
    #         hook_result = hook(self, destination, prefix, local_metadata)
    #         if hook_result is not None:
    #             destination = hook_result
    #     return destination

    @property
    def transformer_model(self):
        # TODO fix the hard-coded segment
        return self.transformer_model_for_mlm.roberta

    @property
    def num_hidden_layers(self):
        return self.config.num_hidden_layers

    @property
    def embeddings_layer(self):
        return self.transformer_model.embeddings

    @overrides
    def get_output_dim(self):
        if self.on_logits:
            return self.config.vocab_size
        return self.output_dim

    def _number_of_token_type_embeddings(self):
        if isinstance(self.config, XLNetConfig):
            return 3  # XLNet has 3 type ids
        elif hasattr(self.config, "type_vocab_size"):
            return self.config.type_vocab_size
        else:
            return 0

    @overrides
    def forward(
        self,
        token_ids: torch.LongTensor,
        mask: torch.BoolTensor,
        type_ids: Optional[torch.LongTensor] = None,
        segment_concat_mask: Optional[torch.BoolTensor] = None,
    ) -> torch.Tensor:  # type: ignore
        """
        # Parameters

        token_ids: `torch.LongTensor`
            Shape: `[batch_size, num_wordpieces if max_length is None else num_segment_concat_wordpieces]`.
            num_segment_concat_wordpieces is num_wordpieces plus special tokens inserted in the
            middle, e.g. the length of: "[CLS] A B C [SEP] [CLS] D E F [SEP]" (see indexer logic).
        mask: `torch.BoolTensor`
            Shape: [batch_size, num_wordpieces].
        type_ids: `Optional[torch.LongTensor]`
            Shape: `[batch_size, num_wordpieces if max_length is None else num_segment_concat_wordpieces]`.
        segment_concat_mask: `Optional[torch.BoolTensor]`
            Shape: `[batch_size, num_segment_concat_wordpieces]`.

        # Returns

        `torch.Tensor`
            Shape: `[batch_size, num_wordpieces, embedding_size]`.

        """
        # Some of the huggingface transformers don't support type ids at all and crash when you supply
        # them. For others, you can supply a tensor of zeros, and if you don't, they act as if you did.
        # There is no practical difference to the caller, so here we pretend that one case is the same
        # as another case.
        if type_ids is not None:
            max_type_id = type_ids.max()
            if max_type_id == 0:
                type_ids = None
            else:
                if max_type_id >= self._number_of_token_type_embeddings():
                    raise ValueError("Found type ids too large for the chosen transformer model.")
                assert token_ids.shape == type_ids.shape

        transformer_mask = segment_concat_mask if self._max_length is not None else mask
        assert transformer_mask is not None
        # Shape: [batch_size, num_wordpieces, embedding_size],
        # or if self._max_length is not None:
        # [batch_size * num_segments, self._max_length, embedding_size]

        # We call this with kwargs because some of the huggingface models don't have the
        # token_type_ids parameter and fail even when it's given as None.
        # Also, as of transformers v2.5.1, they are taking FloatTensor masks.
        parameters = {"input_ids": token_ids, "attention_mask": transformer_mask.float()}
        if type_ids is not None:
            parameters["token_type_ids"] = type_ids

        parameters["return_dict"] = True
        transformer_output = self.transformer_model_for_mlm(**parameters)


        if self.on_logits:
            return transformer_output.logits

        return transformer_output.hidden_states[-1]
