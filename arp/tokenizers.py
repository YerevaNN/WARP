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
                 prompts: List[Optional[Union[str, int, Tuple[str, str]]]] = None,
                 as_one_segment: bool = False,
                 **kwargs):
        super().__init__(*args,
                         add_special_tokens=False,
                         **kwargs)
        self._arp_add_special_tokens = add_special_tokens

        if prompts is None:
            prompts = []

        _, self.prompts = self.prepare_prompts(prompts, tokenizer=self.tokenizer)

        self.as_one_segment = as_one_segment

    @classmethod
    def prepare_prompts(cls,
                        prompts: List[Optional[Union[str, int, Tuple[str, str]]]],
                        tokenizer: PreTrainedTokenizer) -> Tuple[Dict[str, int], List[Optional[Token]]]:
        prompt_to_id: Dict[str, int] = dict()
        prompt_to_init: Dict[str, str] = dict()
        prompt_tokens: List[Optional[Token]] = []
        for prompt in prompts:
            if prompt is None:
                prompt_tokens.append(None)
                continue

            if isinstance(prompt, int):
                # If the index is -1 we treat it as `[MASK]` token
                if prompt > 0:
                    prompt_id = prompt
                else:
                    assert tokenizer.mask_token_id is not None
                    prompt_id = tokenizer.mask_token_id
                prompt_token = tokenizer.convert_ids_to_tokens(prompt_id)
            elif isinstance(prompt, str):
                prompt_token = prompt
                prompt_id = tokenizer.convert_tokens_to_ids(prompt_token)
            else:
                raise NotImplementedError  # TODO

            prompt_to_id[prompt_token] = prompt_id

            prompt_tokens.append(Token(
                text=prompt_token,
                text_id=prompt_id,
                type_id=0,
                idx=None,
                idx_end=None,
            ))

        return prompt_to_id, prompt_tokens

    def tokenize(self, text: str) -> List[Token]:
        tokenized = super().tokenize(text)
        if not self._arp_add_special_tokens:
            return tokenized
        return self.add_special_tokens(tokenized)

    def add_special_tokens(self,
                           tokens1: List[Token],
                           tokens2: Optional[List[Token]] = None
                           ) -> List[Token]:
        prompt_tokens = iter(self.prompts)

        pre_tokens: List[Token] = []
        for token in prompt_tokens:
            if token is None:
                break
            pre_tokens.append(token)

        sep_tokens: List[Token] = []
        for token in prompt_tokens:
            if token is None:
                break
            sep_tokens.append(token)

        post_tokens: List[Token] = []
        for token in prompt_tokens:
            if token is None:
                raise NotImplementedError
            post_tokens.append(token)

        if self.as_one_segment:
            if tokens2 is None:
                tokens2 = []
            return super().add_special_tokens(pre_tokens + tokens1 + sep_tokens + tokens2 + post_tokens)

        assert not sep_tokens
        assert not post_tokens
        return super().add_special_tokens(pre_tokens + tokens1, tokens2)
