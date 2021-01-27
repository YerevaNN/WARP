from typing import Dict, List, NamedTuple, Optional, Tuple, Union

from transformers.tokenization_utils import PreTrainedTokenizer

from allennlp.data.tokenizers import Token, Tokenizer
from allennlp.data.tokenizers import PretrainedTransformerTokenizer


PromptTokenConfig = Union[str, int, Tuple[str, str]]
PromptTemplateConfig = List[Optional[PromptTokenConfig]]


class PromptTemplate(NamedTuple):
    tokens: List[Optional[Token]]
    to_id: Dict[str, int]
    to_init_id: Dict[str, int]


@Tokenizer.register("arp")
class ArpTokenizer(PretrainedTransformerTokenizer):
    def __init__(
        self,
        *args,
        add_special_tokens: bool = True,
        prompts: PromptTemplateConfig = None,
        as_one_segment: bool = False,
        **kwargs
    ):
        super().__init__(*args, add_special_tokens=False, **kwargs)
        self._arp_add_special_tokens = add_special_tokens

        if prompts is None:
            prompts = []

        # if hasattr(self.tokenizer, "sp_model"):
        #     metaspace = SPIECE_UNDERLINE
        # else:
        #     metaspace = "Ġ"
        # self.metaspace = metaspace

        self.prompts = self.prepare_prompts(prompts, tokenizer=self.tokenizer).tokens
        self.as_one_segment = as_one_segment

    @classmethod
    def get_space_aware_token(
        cls, token: str, tokenizer: PreTrainedTokenizer, **kwargs
    ) -> str:
        """
        Convert the given string to a single word/sentence-piece token.
        We might want to reverse-engineer the tokenizer to analyze the
        behaviour because every single implementation differs dramatically.
        """
        if token in ["[MASK]", "<mask>"]:
            return tokenizer.mask_token

        # This trick is developed by professionals in the laboratory conditions
        # !! DO NOT TRY THIS AT HOME !!
        z, q, w = tokenizer.tokenize("zq w", **kwargs)
        # We expect that `q` should always be joined to the `z`
        join_seq, *tail_q = q.rpartition("q")
        assert tail_q == ["q", ""]
        # Similarly, `w` token should always start with whitespace
        space_seq, *tail_w = w.partition("w")
        assert tail_w == ["w", ""]
        # Only if the token starts with space then we treat it as a separate token
        if not token.startswith(" "):
            return join_seq + token

        *head, token = token.partition(" ")
        assert head == ["", " "]
        return space_seq + token

        # Here we also check whether the first token contains a whitespace
        prefix_space = z == q

    @classmethod
    def prepare_prompts(
        cls,
        prompts: PromptTemplateConfig,
        tokenizer: PreTrainedTokenizer,
        default_init: Union[str, bool] = None,
    ) -> PromptTemplate:
        if not isinstance(default_init, str):
            default_init = None

        prompt_tokens: List[Optional[Token]] = []
        prompt_to_id: Dict[str, int] = dict()
        prompt_to_init: Dict[str, int] = dict()
        for prompt in prompts:
            if prompt is None:
                prompt_tokens.append(None)
                continue
            # We initialize with the default initializer unless stated otherwise
            init_with = default_init

            # If an tuple/list is given, the second argument is the custom initializer
            if isinstance(prompt, (tuple, list)):
                prompt, init_with = prompt

            # If an integer is given, we need to covert it into a token
            if isinstance(prompt, int):
                # If the index is -1 we treat it as `[MASK]` token
                if prompt < 0:
                    # assert tokenizer.mask_token_id is not None
                    # prompt = tokenizer.mask_token_id
                    prompt += tokenizer.vocab_size

                assert (
                    prompt not in tokenizer.all_special_ids
                ), "Do not hardcode special IDs"
                # Then we convert it into a wordpiece
                prompt = tokenizer.convert_ids_to_tokens(prompt)

            assert isinstance(prompt, str)

            if prompt in ["[MASK]", "<mask>"]:
                prompt = tokenizer.mask_token

            prompt = cls.get_space_aware_token(prompt, tokenizer)
            # if prompt.startswith(' '):
            #     prompt = self.metaspace + prompt[1:]

            # TODO let's make sure t
            prompt_id: int = tokenizer.convert_tokens_to_ids(prompt)
            assert (
                prompt_id != tokenizer.unk_token_id
            ), "Using UNK not implemented yet, may be tricky"

            prompt_tokens.append(
                Token(
                    text=prompt,
                    text_id=prompt_id,
                    type_id=0,
                    idx=None,
                    idx_end=None,
                )
            )
            prompt_to_id[prompt] = prompt_id
            if init_with is not None:
                # if init_with in ['[MASK]', '<mask>']:
                #     init_with = tokenizer.mask_token
                init_with = cls.get_space_aware_token(init_with, tokenizer)
                prompt_to_init[prompt] = tokenizer.convert_tokens_to_ids(init_with)

        return PromptTemplate(prompt_tokens, prompt_to_id, prompt_to_init)

    def tokenize(self, text: str) -> List[Token]:
        tokenized = super().tokenize(text)
        if not self._arp_add_special_tokens:
            return tokenized
        return self.add_special_tokens(tokenized)

    def add_special_tokens(
        self, tokens1: List[Token], tokens2: Optional[List[Token]] = None
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
            return super().add_special_tokens(
                pre_tokens + tokens1 + sep_tokens + tokens2 + post_tokens
            )

        if tokens2 is None:
            return super().add_special_tokens(
                pre_tokens + tokens1 + sep_tokens + post_tokens
            )
        else:
            return super().add_special_tokens(
                pre_tokens + tokens1 + sep_tokens, tokens2 + post_tokens
            )
