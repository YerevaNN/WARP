from typing import Dict, List, Optional, Tuple, Union

from allennlp.data.tokenizers import PretrainedTransformerTokenizer
from allennlp.data.tokenizers.token import Token
from allennlp.data.tokenizers.tokenizer import Tokenizer

from transformers.tokenization_utils import PreTrainedTokenizer


@Tokenizer.register('arp')
class ArpTokenizer(PretrainedTransformerTokenizer):
    def __init__(self,
                 *args,
                 add_special_tokens: bool = True,
                 prompts: Union[int, List[Union[str, int]]] = None,
                 **kwargs):
        super().__init__(*args,
                         add_special_tokens=False,
                         **kwargs)
        self._arp_add_special_tokens = add_special_tokens

        if prompts is None:
            prompts = 1
        if isinstance(prompts, int):
            prompts = [50261, 50262, 50263][:prompts]

        _, self.prompts = self.prepare_prompts(prompts, tokenizer=self.tokenizer)

    @classmethod
    def prepare_prompts(cls,
                        prompts: List[Union[str, int]],
                        tokenizer: PreTrainedTokenizer) -> Tuple[Dict[str, int], List[Token]]:
        prompt_to_id: Dict[str, int] = dict()
        prompt_tokens: List[Token] = []
        for prompt in prompts:
            if isinstance(prompt, int):
                # If the index is -1 we treat it as `[MASK]` token
                if prompt > 0:
                    prompt_id = prompt
                else:
                    assert tokenizer.mask_token_id is not None
                    prompt_id = tokenizer.mask_token_id
                prompt_token = tokenizer.convert_ids_to_tokens(prompt_id)
            else:
                prompt_token = prompt
                prompt_id = tokenizer.convert_tokens_to_ids(prompt_token)

            prompt_to_id[prompt_token] = prompt_id

            prompt_tokens.append(Token(
                text=prompt_token,
                text_id=prompt_id,
                type_id=0,
                idx=None,
                idx_end=None,
            ))

        return prompt_to_id, prompt_tokens

    @overrides
    def tokenize(self, text: str) -> List[Token]:
        tokenized = super().tokenize(text)
        if not self._arp_add_special_tokens:
            return tokenized
        return self.add_special_tokens(tokenized)

    def add_special_tokens(self,
                           tokens1: List[Token],
                           tokens2: Optional[List[Token]] = None
                           ) -> List[Token]:
        prepend_tokens = []
        for token in self.prompts:
            prepend_tokens.append(token)
        return super().add_special_tokens(prepend_tokens + tokens1, tokens2)
