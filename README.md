# WARP
Word-level Adversarial ReProgramming

## Training


# Linear Probing
```sh
for DATASET in 'cola' 'sst2' 'mrpc' 'qqp' 'stsb' 'mnli' 'rte' 'wnli' 'qnli'
do
    export HPARAMS='{
        "dataset": "'$DATASET'",
        "lr": 0.0001,
        "num_epochs": 20,
        "prompts": [],
        "reorder_optimized": false,
        "max_batch_size": 8,
        "max_tokens_sq": 262144, "on_logits":  false, "pooling_index":  null, "seed":  1}'
    python -m allennlp train \
    -s .aim/baseline-linear-${DATASET} configs/warp.jsonnet
done
```

# WARP_0
```sh
for DATASET in 'cola' 'sst2' 'mrpc' 'qqp' 'stsb' 'mnli' 'rte' 'wnli' 'qnli'
do
    export HPARAMS='{
        "dataset": "'$DATASET'",
        "lr": 0.0001,
        "num_epochs": 20,
        "prompts": [null, "<mask>"],
        "reorder_optimized": true,
        "max_batch_size": 8,
        "max_tokens_sq": 262144,
        "on_logits": "pre_decoder_layer_norm",
        "pooling_index": 1,
        "seed": 1
    }'
    python -m allennlp train \
    -s .aim/baseline-warp_0-${DATASET} configs/warp.jsonnet
done
```

## Training WARP

```sh
export DATASET="rte"
export HPARAMS='{
    "benchmark":"super_glue",
    "classifier_init":null,
    "dataset":"'$DATASET'",
    "ensure_whitespace_between":false,
    "lr":0.001,
    "max_batch_size":8,
    "max_tokens_sq":262144,
    "num_epochs":30,
    "prompt_better_init":"<mask>",
    "prompts":[-10,-11,-12,-13,-14,null,-15,-16,-17,-18,-19,"<mask>",-20,-21,-22,-23,-24,null,-25,-26,-27,-28,-29],
    "seed":1,
    "transformer_model":"roberta-large"
}'
python -m allennlp train \
-s .aim/t-${DATASET} configs/warp.jsonnet
```

## WARP_init
## Few-Shot Experiments
```sh
export HPARAMS='{
    "benchmark":"super_glue",
    "classifier_init": {
        "entailment": " yes",
        "not_entailment": " instead"
    },
    "dataset":"few_rte",
    "eval_mode":false,
    "lr":0.001,
    "max_batch_size":2,
    "max_tokens_sq":131072,
    "num_epochs":100,
    "num_gradient_accumulation_steps":2,
    "prompt_better_init": "[PAD]",
    "prompts":[-10,-11,[-14,"\""],null,[-15,"\""],  [-16, "?"], "<mask>", [-20, ","], null, [-29, "!"],-30,-31],
    "seed":3,
    "str_cut_frac":0,
    "transformer_model":"albert-xxlarge-v2",
    "validation_metric": null
}'
python -m allennlp train \
-s .aim/t-${DATASET}-`date +%s` configs/warp.jsonnet
```


```sh
export HPARAMS='{
   "benchmark":"super_glue",
   "classifier_init":{
      "entailment":" yes",
      "not_entailment":" instead"
   },
   "dataset":"few_rte",
   "grad_norm":1,
   "lr":0.001,
   "max_batch_size":2,
   "max_tokens_sq":131072,
   "num_epochs":30,
   "num_gradient_accumulation_steps":2,
   "prompt_better_init":"[PAD]",
   "prompts":[-10,-11,[-14,"\""],null,[-15,"\""],[-16,"?"],"<mask>",[-20,","],null,[-29,"!"],-30,-31],
   "seed":1,
   "str_cut_frac":0.06,
   "transformer_model":"albert-xxlarge-v2",
   "validation_metric":"+training_val_metric"
}'
python -m allennlp train \
-s .aim/t-${DATASET}-`date +%s` configs/warp.jsonnet
```

## Evaluation

```sh
python -m allennlp predict \
  --silent --use-dataset-reader --cuda-device 0 \
  --batch-size 50 \
  --predictor glue --output-file v0.1/AX.tsv /data/arp/.aim/H-93ae5ae9 ax/test
```

```sh
python -m allennlp predict \
  --silent --use-dataset-reader --cuda-device 0 \
  --batch-size 50 \
  --predictor glue --output-file v0.1/MNLI-m.tsv /data/arp/.aim/H-93ae5ae9 test_matched
```
