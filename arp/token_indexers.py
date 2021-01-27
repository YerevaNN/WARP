from typing import Dict, List, Optional
from overrides import overrides

import logging

from transformers.tokenization_utils import PreTrainedTokenizer

from allennlp.data import Vocabulary
from allennlp.data import Token
from allennlp.data import TokenIndexer
from allennlp.data.token_indexers import PretrainedTransformerIndexer
from allennlp.data import IndexedTokenList

from .tokenizers import ArpTokenizer, PromptTemplate, PromptTemplateConfig

logger = logging.getLogger(__name__)


@TokenIndexer.register("pretrained_transformer_permute")
class PretrainedTransformerPermuteIndexer(PretrainedTransformerIndexer):
    def __init__(self, optimize_prompts: PromptTemplateConfig = None, **kwargs):
        super().__init__(**kwargs)
        self._tokenizer: PreTrainedTokenizer
        self._max_positions = self._tokenizer.model_max_length

        self.prompt_template: Optional[PromptTemplate] = None
        if optimize_prompts is not None:
            self.prompt_template = ArpTokenizer.prepare_prompts(
                optimize_prompts, tokenizer=self._tokenizer
            )

    @overrides
    def tokens_to_indices(
        self, tokens: List[Token], vocabulary: Vocabulary
    ) -> IndexedTokenList:

        output: IndexedTokenList = super().tokens_to_indices(tokens, vocabulary)

        padding_idx = self._tokenizer.pad_token_id
        token_ids: List[int] = output["token_ids"]
        mask = output["mask"]
        type_ids = output["type_ids"]
        position_ids = [i + padding_idx + 1 for i, _ in enumerate(token_ids)]

        mask_token_id = self._tokenizer.mask_token_id

        if self.prompt_template is not None:
            assert mask_token_id in token_ids, "Optimized mode is only supported on MLM"
            original_positions: Dict[int, int] = dict()
            for position, token_id in enumerate(token_ids):
                if token_id not in original_positions:
                    original_positions[token_id] = position

            head_ids: List[int] = []
            head_ids.append(original_positions.pop(self._tokenizer.cls_token_id))
            head_ids.append(original_positions.pop(mask_token_id))
            for prompt_token in self.prompt_template.tokens:
                if prompt_token is None:
                    continue
                if prompt_token.text_id == mask_token_id:
                    continue
                original_position: int = original_positions.pop(prompt_token.text_id)
                head_ids.append(original_position)

            tail_ids: List[int] = []
            for position, token_id in enumerate(token_ids):
                if token_id in original_positions:
                    tail_ids.append(position)

            if len(head_ids) + len(tail_ids) != len(token_ids):
                logger.warning("Some prompt tokens are used in the real text")
                tail_ids = [
                    idx for idx, _ in enumerate(token_ids) if idx not in head_ids
                ]

            permutation = head_ids + tail_ids
            token_ids = [token_ids[idx] for idx in permutation]
            mask = [mask[idx] for idx in permutation]
            type_ids = [type_ids[idx] for idx in permutation]
            position_ids = [
                min(position_ids[idx], self._max_positions - 1) for idx in permutation
            ]

        else:
            raise NotImplementedError
            # Now, let's permute
            # Let's find `[MASK]` and switch it with places position#1
            mask_idx = self._tokenizer.mask_token_id
            if mask_idx in token_ids:
                pos = token_ids.index(mask_idx)
                token_ids[1], token_ids[pos] = token_ids[pos], token_ids[1]
                mask[1], mask[pos] = mask[pos], mask[1]
                type_ids[1], type_ids[pos] = type_ids[pos], type_ids[1]
                position_ids[1], position_ids[pos] = position_ids[pos], position_ids[1]

        return {
            "token_ids": token_ids,
            "mask": mask,
            "type_ids": type_ids,
            "position_ids": position_ids,
        }

    @overrides
    def get_empty_token_list(self) -> IndexedTokenList:
        output = super().get_empty_token_list()
        output["position_ids"] = []
        return output
