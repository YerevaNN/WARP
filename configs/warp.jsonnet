local libs = {
    hparams: import 'hparams.libsonnet'
};
local L = libs;


local hparams = L.hparams.get({
    as_one_segment: true,
    batch_size: null,
    batches_per_epoch: null,
    benchmark: 'glue',
    classifier_bias: true,
    classifier_init: null,
    classifier_trainable: true,
    cross_validation: false,
    dataset: null,
    dropout: null,
    ensure_whitespace_between: true,
    eval_mode: true,
    grad_norm: 1.0,
    lr: 2e-5,
    max_tokens: 1024,
    max_tokens_sq: null,
    max_batch_size: null,
    num_epochs: 10,
    num_gradient_accumulation_steps: 1,
    num_samples: null,
    on_logits: 'pre_decoder_layer_norm',
    pad_to_multiple_of: 8,
    padding_noise: 0.1,
    patience: null,
    pooling_index: 1,
    pretrained_prompts: null,
    prompt_better_init: true,
    prompt_dropout: null,
    prompts: [null, -2, '<mask>', -3, null],
    reorder_optimized: true,
    seed: 133,
    str_cut_frac: 0.06,
    train_parameters: 'only_prompts',
    train_set: 'train',
    transformer_model: 'roberta-large',
    use_amp: true,
    validation_metric: '+val_metric',
    validation_set: null,
    weight_decay: 0,
});
local H = hparams;

{

    "random_seed": (H.seed * 100 + 70) % 65536,
    "numpy_seed": (H.seed * 10 + 7) % 65536,
    "pytorch_seed": (H.seed) % 65536,

    "dataset_reader": {
        "type": "huggingface",
        "path": H.benchmark,
        "name": if H.cross_validation then (H.dataset + '_cross_validation_' + H.seed) else H.dataset,
        "ensure_whitespace_between": H.ensure_whitespace_between,
        "tokenizer": {
            "type": "arp",
            "model_name": H.transformer_model,
            "add_special_tokens": false,
            "prompts": H.prompts,
            "as_one_segment": H.as_one_segment
        },
        "token_indexers": {
            "tokens": {
                "type": if H.reorder_optimized
                        then "pretrained_transformer_permute"
                        else "pretrained_transformer",
                "model_name": H.transformer_model,
                "token_min_padding_length": 0,
                "pad_to_multiple_of": H.pad_to_multiple_of
            } + (if H.reorder_optimized then {
                "optimize_prompts": H.prompts
            } else {})
        },
        "max_instances": H.num_samples
    },

    "validation_dataset_reader": self.dataset_reader {
        "max_instances": null
    },

    "train_data_path": H.train_set,
    "validation_data_path": if H.validation_set != null then H.validation_set else (
        if H.dataset == 'mnli' then 'validation_matched' else 'validation'
    ),


    "model": {
        "type": "warp_classifier",
        "text_field_embedder": {
            "token_embedders": {
                "tokens": {
                    "type": "pretrained_mlm",
                    "model_name": H.transformer_model,
                    "train_parameters": H.train_parameters,
                    "arp_injector": {
                        "prompts": H.prompts,
                        "prompt_better_init": H.prompt_better_init,
                        "optimized_prompts": H.reorder_optimized,
                        "dropout": H.prompt_dropout
                    },
                    "on_logits": H.on_logits,
                    "eval_mode": H.eval_mode
                }
            }
        },
        "seq2vec_encoder": if H.pooling_index != null then {
            "type": "at",
            "index": H.pooling_index
        } else {
            "type": "boe",
            "averaged": true
        },
        "dropout": H.dropout,
        "namespace": "tags",
        "classifier_bias": H.classifier_bias,
        "classifier_trainable": H.classifier_trainable,
        "classifier_init": H.classifier_init,
        "metrics": [
            {
                "type": "huggingface",
                "path": H.benchmark,
                "name": H.dataset,
            },
        ],
        "initializer": if H.pretrained_prompts == null then {} else {
            "regexes": [
                ["\\.prompt_params", {
                    "type": "pretrained",
                    "weights_file_path": H.pretrained_prompts,
                    "parameter_name_overrides": {
                    }
                }]
            ]
        },
    },

    "data_loader": {
        "batch_sampler": if H.batch_size == null then {
            "type": "max_tokens_sampler",
            "max_tokens" : if H.max_tokens_sq != null then H.max_tokens_sq else H.max_tokens,
            "square_complexity": (H.max_tokens_sq != null),
            "max_batch_size": H.max_batch_size
        } else {
            "type": "bucket",
            "batch_size": H.batch_size,
            "padding_noise": H.padding_noise
        },
        "batches_per_epoch": H.batches_per_epoch
    },

    "validation_data_loader": {
        "batch_sampler": {
            "type": "max_tokens_sampler",
            "max_tokens" : if H.transformer_model != 'albert-xxlarge-v2' then 4096 else 512
    },


    "trainer": {
        "use_amp": H.use_amp,
        "grad_norm": H.grad_norm,
        "num_epochs": H.num_epochs,
        "validation_metric": H.validation_metric,
        "patience": H.patience,
        "num_gradient_accumulation_steps": H.num_gradient_accumulation_steps,
        "learning_rate_scheduler": if H.str_cut_frac > 0 then {
            "type": "slanted_triangular",
            "cut_frac": H.str_cut_frac
        } else null,
        "optimizer": {
            "type": "huggingface_adamw",
            "lr": H.lr,
            "weight_decay": H.weight_decay,
        },
        "trainer_callbacks": [
            {
                "type": "aim",
                "hparams": H + {"extras": L.hparams.extras}

            },
            "track_epoch_callback",
        ],
        "checkpointer": {
            "num_serialized_models_to_keep": 1
        },
    }
}
