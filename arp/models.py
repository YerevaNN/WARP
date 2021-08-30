from typing import Any, Dict, List, Optional, Union
from overrides import overrides

import json
import logging

import torch
import torch.nn
import torch.nn.functional

from allennlp.common.lazy import Lazy
from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.models.model import Model
from allennlp.modules import FeedForward, Seq2VecEncoder, TextFieldEmbedder
from allennlp.nn import InitializerApplicator, util
from allennlp.nn.regularizers import RegularizerApplicator
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import CategoricalAccuracy, Metric

from .injector import ArpInjector
from .tokenizers import ArpTokenizer

logger = logging.getLogger(__name__)


# @Model.register("arp_classifier")
class ArpClassifier(Model):
    def __init__(
        self,
        vocab: Vocabulary,
        text_field_embedder: TextFieldEmbedder,
        seq2vec_encoder: Seq2VecEncoder,
        feedforward: Optional[FeedForward] = None,
        dropout: float = None,
        num_labels: int = None,
        label_namespace: str = "labels",
        namespace: str = "tokens",
        initializer: InitializerApplicator = InitializerApplicator(),
        regularizer: RegularizerApplicator = None,
        serialization_dir: Optional[str] = None,
        classifier_bias: bool = True,
        classifier_trainable: bool = True,
        classifier_init: Dict[str, str] = None,
        metrics: Optional[List[Metric]] = None,
    ) -> None:
        super().__init__(
            vocab, regularizer=regularizer, serialization_dir=serialization_dir
        )
        self._text_field_embedder = text_field_embedder
        self._seq2vec_encoder = seq2vec_encoder
        self._feedforward = feedforward
        self._label_namespace = label_namespace
        self._namespace = namespace

        self._metrics = metrics or []
        self._reports: Dict[str, Any] = {}

        if feedforward is not None:
            self._classifier_input_dim = feedforward.get_output_dim()
        else:
            self._classifier_input_dim = self._seq2vec_encoder.get_output_dim()

        self._dropout: Optional[torch.nn.Dropout] = None
        if dropout:
            self._dropout = torch.nn.Dropout(dropout)

        if num_labels is None:
            num_labels = vocab.get_vocab_size(namespace=self._label_namespace)
        if num_labels <= 1:
            logger.warning("Treating as a regression task!")
            num_labels = 1

        self._num_labels = num_labels

        self._loss: Union[torch.nn.CrossEntropyLoss, torch.nn.MSELoss]
        self._accuracy: Optional[CategoricalAccuracy]
        if num_labels > 1:
            self._loss = torch.nn.CrossEntropyLoss()
            self._accuracy = CategoricalAccuracy()
        else:
            self._loss = torch.nn.MSELoss()
            self._accuracy = None

        self._classification_layer = torch.nn.Linear(
            self._classifier_input_dim, self._num_labels, bias=classifier_bias
        )
        if classifier_init is not None:
            injector: ArpInjector = self._text_field_embedder._token_embedders[
                "tokens"
            ].embeddings_layer.word_embeddings
            tokenizer = injector.tokenizer
            lm_head = self._text_field_embedder._token_embedders["tokens"].lm_head
            for class_name, init_with in classifier_init.items():
                class_idx = self.vocab.get_token_index(
                    class_name, namespace=self._label_namespace
                )
                init_with = ArpTokenizer.get_space_aware_token(init_with, tokenizer)
                token_idx = tokenizer.convert_tokens_to_ids(init_with)
                with torch.no_grad():
                    self._classification_layer.weight[
                        class_idx
                    ] = injector.embedder.weight[token_idx]
                    if self._classification_layer.bias is not None:
                        self._classification_layer.bias[class_idx] = lm_head.bias[
                            token_idx
                        ]

        if not classifier_trainable:
            self._classification_layer.weight.requires_grad = False

        initializer(self)

    @classmethod
    def from_partial_objects(
        cls,
        vocab: Vocabulary,
        text_field_embedder: TextFieldEmbedder,
        seq2vec_encoder: Lazy[Seq2VecEncoder],
        feedforward: Lazy[FeedForward] = None,
        dropout: float = None,
        num_labels: int = None,
        label_namespace: str = "labels",
        namespace: str = "tokens",
        initializer: InitializerApplicator = InitializerApplicator(),
        regularizer: RegularizerApplicator = None,
        serialization_dir: Optional[str] = None,
        classifier_bias: bool = True,
        classifier_trainable: bool = True,
        classifier_init: Dict[str, str] = None,
        metrics: Optional[Union[Metric, List[Metric]]] = None,
    ):
        embedding_dim = text_field_embedder.get_output_dim()

        seq2vec_encoder_ = seq2vec_encoder.construct(embedding_dim=embedding_dim)
        embedding_dim = seq2vec_encoder_.get_output_dim()

        feedforward_: Optional[FeedForward]
        if feedforward is not None:
            feedforward_ = feedforward.construct(input_dim=embedding_dim)
            embedding_dim = feedforward_.get_output_dim()
        else:
            feedforward_ = None

        return cls(
            vocab=vocab,
            text_field_embedder=text_field_embedder,
            seq2vec_encoder=seq2vec_encoder_,
            feedforward=feedforward_,
            dropout=dropout,
            num_labels=num_labels,
            label_namespace=label_namespace,
            namespace=namespace,
            initializer=initializer,
            regularizer=regularizer,
            serialization_dir=serialization_dir,
            classifier_bias=classifier_bias,
            classifier_trainable=classifier_trainable,
            classifier_init=classifier_init,
            metrics=metrics,
        )

    def forward(  # type: ignore
        self, tokens: TextFieldTensors = None, label: torch.IntTensor = None, **metadata
    ) -> Dict[str, torch.Tensor]:

        """
        # Parameters

        tokens : `TextFieldTensors`
            From a `TextField`
        label : `torch.IntTensor`, optional (default = `None`)
            From a `LabelField`

        # Returns

        An output dictionary consisting of:

            - `logits` (`torch.FloatTensor`) :
                A tensor of shape `(batch_size, num_labels)` representing
                unnormalized log probabilities of the label.
            - `probs` (`torch.FloatTensor`) :
                A tensor of shape `(batch_size, num_labels)` representing
                probabilities of the label.
            - `loss` : (`torch.FloatTensor`, optional) :
                A scalar loss to be optimised.
        """
        if tokens is None:
            tokens = metadata.pop("sentence")

        token_embeddings = self._text_field_embedder(tokens)

        mask = get_text_field_mask(tokens)

        text_embeddings = self._seq2vec_encoder(token_embeddings, mask=mask)

        if self._dropout:
            text_embeddings = self._dropout(text_embeddings)

        if self._feedforward is not None:
            text_embeddings = self._feedforward(text_embeddings)

        logits = self._classification_layer(text_embeddings)
        output_dict = {"logits": logits}
        if self._num_labels > 1:
            probs = torch.nn.functional.softmax(logits, dim=-1)
            output_dict["probs"] = probs

        for key in ["idx", "pair_id"]:
            output_dict[key] = metadata.get(key, [None] * len(logits))
        output_dict["token_ids"] = util.get_token_ids_from_text_field_tensors(tokens)

        if label is not None:
            if self._num_labels > 1:
                loss = self._loss(logits, label.long().view(-1))
                output_dict["loss"] = loss

                assert self._accuracy is not None
                self._accuracy(logits, label)

                # Shape: (batch_size,)
                predictions = logits.argmax(axis=-1)
                # Shape: (batch_size,)
                references = label

            else:
                # Shape: (batch_size,)
                predictions = logits.squeeze(-1)
                # Shape: (batch_size,)
                references = label
                loss = self._loss(logits.squeeze(-1), label)
                output_dict["loss"] = loss

            for metric in self._metrics:
                metric(predictions, references)

        return output_dict

    @overrides
    def make_output_human_readable(
        self, output_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, Union[List, torch.Tensor]]:
        """
        Does a simple argmax over the probabilities, converts index to string label, and
        add `"label"` key to the dictionary with the result.
        """
        predictions = output_dict["logits"]

        if self._num_labels <= 1:
            return {
                "index": output_dict["idx"],
                "prediction": predictions.squeeze(-1).tolist(),
            }

        class_ids: List[int] = []
        class_names: List[str] = []
        for prediction in predictions.cpu():
            label_idx: int = prediction.argmax(dim=-1).item()
            label_str = self.vocab.get_index_to_token_vocabulary(
                self._label_namespace
            ).get(label_idx, str(label_idx))
            class_ids.append(label_idx)
            class_names.append(label_str)
        return {
            "index": output_dict["idx"],
            "prediction": class_ids,
            "label": class_names,
        }

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:

        if reset:
            ...
            # self.on_batch(batch_number=0)

        metrics = {}
        if self._accuracy is not None:
            metrics["accuracy"] = self._accuracy.get_metric(reset)
        for metric in self._metrics:
            for key, value in metric.get_metric(reset).items():
                metrics[key] = value
        if reset:
            metrics.update(self._reports)

        if reset:
            metric_order = [
                "pearson",
                "spearmanr",
                "matthews_correlation",
                "f1",
                "accuracy",
            ]
            for metric_key in metric_order:
                val_metric = metrics.get(metric_key)
                if val_metric is not None:
                    break
            else:
                raise NotImplementedError

            metrics["val_metric"] = val_metric

        return metrics

    def load_state_dict(
        self,
        state_dict: Union[Dict[str, torch.Tensor], Dict[str, torch.Tensor]],
        strict: bool = False,
    ) -> None:
        outputs = super().load_state_dict(state_dict, strict=strict)
        return outputs

        predictions_list = [predictions]
        classes = []
        for prediction in predictions_list:
            label_idx = prediction.argmax(dim=-1).item()
            label_str = self.vocab.get_index_to_token_vocabulary(self._label_namespace).get(
                label_idx, str(label_idx)
            )
            classes.append(label_str)
        output_dict["label"] = classes
        tokens = []
        for instance_tokens in output_dict["token_ids"]:
            tokens.append(
                [
                    self.vocab.get_token_from_index(token_id.item(), namespace=self._namespace)
                    for token_id in instance_tokens
                ]
            )
        output_dict["tokens"] = tokens
        return output_dict

    def load_state_dict(self, state_dict: Union[Dict[str, torch.Tensor], Dict[str, torch.Tensor]],
                        strict: bool = False):
        return super().load_state_dict(state_dict, strict=strict)

    default_predictor = "glue-numeric"


Model.register("arp_classifier", constructor="from_partial_objects")(ArpClassifier)
Model.register("warp_classifier", constructor="from_partial_objects")(ArpClassifier)
