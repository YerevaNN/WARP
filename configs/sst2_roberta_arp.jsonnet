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
    num_gradient_accumulation_steps: 1,
    dropout: null,
    dataset_version: 1,
    prompt_better_init: false
});
local H = hparams;



{
    "hparams": H,

    "random_seed": (H.seed * 100 + 70) % 65536,
    "numpy_seed": (H.seed * 10 + 7) % 65536,
    "pytorch_seed": (H.seed) % 65536,

    "dataset_reader": {
        "type": "sst_tokens",
        "use_subtrees": true,
        "granularity": "2-class",
        "token_indexers": {
            "tokens": {
                "type": "pretrained_transformer",
                "model_name": H.transformer_model
            }
        },
        "tokenizer": {
            "type": "arp",
            "model_name": H.transformer_model,
            "prompts": H.prompts
        },
        "version": H.dataset_version,
        "original_rt_snippets": "https://raw.githubusercontent.com/hithisisdhara/stanford_1/master/original_rt_snippets.txt"
    },
    "validation_dataset_reader": self.dataset_reader + {
        "use_subtrees": false
    },

    "train_data_path": "https://allennlp.s3.amazonaws.com/datasets/sst/train.txt",
    "validation_data_path": "https://allennlp.s3.amazonaws.com/datasets/sst/dev.txt",
    "test_data_path": "https://allennlp.s3.amazonaws.com/datasets/sst/test.txt",

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
                    "on_logits": H.on_logits
                }
            }
        },
        "seq2vec_encoder": {
            "type": "at",
            "index": H.pooling_index
        },
        "dropout": H.dropout,
        "namespace": "tags"
    },

    "data_loader": {
        "batch_sampler": {
            "type": "max_tokens_sampler",
            "max_tokens" : H.max_tokens
        },
        "pin_memory": true
    },

    "trainer": {
        "use_amp": true,
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
}
