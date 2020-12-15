local libs = {
    hparams: import 'hparams.libsonnet'
};
local L = libs;


local hparams = L.hparams.get({
    transformer_model: 'roberta-large',
    prompts: [],
    on_logits: false,
    pooling_index: 0,
    train_parameters: 'last_layer_only',
    seed: 133,
    lr: 2e-5,
    num_epochs: 10,
    str_cut_frac: 0.06,
    weight_decay: 0.1,
    max_tokens: 1536,
    batch_size: null,
    num_gradient_accumulation_steps: 1,
    dropout: null,
    // dataset_version: 1,
    prompt_better_init: false,
    as_one_segment: false,
    batches_per_epoch: null,
    pad_to_multiple_of: 1,
    eval_mode: false,
    padding_noise: 0.1,
    dataset: "mnli",
    use_amp: true
});
local H = hparams;

local paths = {
    "mnli": {
        "train_data_path": "https://allennlp.s3.amazonaws.com/datasets/multinli/multinli_1.0_train.jsonl",
        "validation_data_path": "https://allennlp.s3.amazonaws.com/datasets/multinli/multinli_1.0_dev_matched.jsonl",
        "test_data_path": "https://allennlp.s3.amazonaws.com/datasets/multinli/multinli_1.0_dev_mismatched.jsonl",
    },
    "rte": {
        "train_data_path": "https://storage.googleapis.com/public.storage.yerevann.com/datasets/nlp/superglue_data/RTE/train.jsonl",
        "validation_data_path": "https://storage.googleapis.com/public.storage.yerevann.com/datasets/nlp/superglue_data/RTE/val.jsonl",
        "test_data_path": "https://storage.googleapis.com/public.storage.yerevann.com/datasets/nlp/superglue_data/RTE/test.jsonl",
    },
    "few-rte": self.rte {
        "train_data_path": "https://storage.googleapis.com/public.storage.yerevann.com/datasets/nlp/FewGLUE/RTE/train.jsonl"
    },

    "cb": {
        "train_data_path": "https://storage.googleapis.com/public.storage.yerevann.com/datasets/nlp/superglue_data/CB/train.jsonl",
        "validation_data_path": "https://storage.googleapis.com/public.storage.yerevann.com/datasets/nlp/superglue_data/CB/val.jsonl",
        "test_data_path": "https://storage.googleapis.com/public.storage.yerevann.com/datasets/nlp/superglue_data/CB/test.jsonl",
    },
    "few_cb": self.cb {
        "train_data_path": "https://storage.googleapis.com/public.storage.yerevann.com/datasets/nlp/FewGLUE/CB/train.jsonl"
    },
};

{
    "hparams": H,

    "random_seed": (H.seed * 100 + 70) % 65536,
    "numpy_seed": (H.seed * 10 + 7) % 65536,
    "pytorch_seed": (H.seed) % 65536,

    "dataset_reader": {
        "type": "snli",
        "tokenizer": {
            "type": "arp",
            "model_name": H.transformer_model,
            "add_special_tokens": false,
            "prompts": H.prompts,
            "as_one_segment": H.as_one_segment,
        },
        "token_indexers": {
            "tokens": {
                "type": "pretrained_transformer",
                "model_name": H.transformer_model,
                // "max_length": 512,
                "token_min_padding_length": 0,
                "pad_to_multiple_of": H.pad_to_multiple_of
            }
        },
    },

    "validation_dataset_reader": self.dataset_reader,

    "train_data_path": null, // "https://allennlp.s3.amazonaws.com/datasets/multinli/multinli_1.0_train.jsonl",
    "validation_data_path": null, //"https://allennlp.s3.amazonaws.com/datasets/multinli/multinli_1.0_dev_matched.jsonl",
    "test_data_path": null, //"https://allennlp.s3.amazonaws.com/datasets/multinli/multinli_1.0_dev_mismatched.jsonl",

    "model": {
        "type": "arp_classifier",
        "text_field_embedder": {
            "token_embedders": {
                "tokens": {
                    "type": "pretrained_mlm",
                    "model_name": H.transformer_model,
                    "train_parameters": H.train_parameters,
                    "arp_injector": {
                        "prompts": H.prompts,
                        "prompt_better_init": H.prompt_better_init
                    },
                    "on_logits": H.on_logits,
                    "eval_mode": H.eval_mode
                }
            }
        },
        "seq2vec_encoder": {
            "type": "at",
            "index": H.pooling_index
        },
        "dropout": H.dropout,
        "namespace": "tags",

        // "fp16": true
    },

    "data_loader": {
        "batch_sampler": if H.batch_size == null then {
            "type": "max_tokens_sampler",
            "max_tokens" : H.max_tokens
        } else {
            // "type": "basic",
            // "batch_size": 8,
            // "sampler": "sequential",
            // "drop_last": false
            "type": "bucket",
            "batch_size": H.batch_size,
            "padding_noise": H.padding_noise
        },
        // "pin_memory": true,
        "batches_per_epoch": H.batches_per_epoch
    },

    "validation_data_loader": {
        "batch_sampler": {
            "type": "max_tokens_sampler",
            "max_tokens" : 4096
            // "type": "basic",
            // "batch_size": 8,
            // "sampler": "sequential",
            // "drop_last": false
        },
        // "pin_memory": true,
        // "batches_per_epoch": H.batches_per_epoch
    },


    "trainer": {
        "use_amp": H.use_amp,
        "num_epochs": H.num_epochs,
        "validation_metric": "+accuracy",
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
            },
        ],
        "checkpointer": {
            "num_serialized_models_to_keep": 0
        },
    }
} + paths[H.dataset]
