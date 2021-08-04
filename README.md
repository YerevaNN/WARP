# 🌀 WARP: Word-level Adversarial ReProgramming
This repository contains code for ACL'2021 Paper [WARP: Word-level Adversarial ReProgramming](https://arxiv.org/abs/2101.00121).

<img src="https://svgshare.com/i/XpG.svg">
<sup>WARP adds a few trainable embeddings around the input, which causes the masked language model to predict the sentiment of the sentence in the SST-2 task.</sup>

Transfer learning from pretrained language models recently became the dominant approach for solving many NLP tasks. A common approach to transfer learning for multiple tasks that maximize parameter sharing trains one or more task-specific layers on top of the language model.

In this paper, we present an alternative approach based on adversarial reprogramming, which extends earlier work on automatic prompt generation. Adversarial reprogramming attempts to learn task-specific word embeddings that, when concatenated to the input text, instruct the language model to solve the specified task.

Using up to 25K trainable parameters per task, this approach outperforms all existing methods with up to 25M trainable parameters on the public leaderboard of the GLUE benchmark. Our method, initialized with task-specific human-readable prompts, also works in a few-shot setting, outperforming GPT-3 on two SuperGLUE tasks with just 32 training samples.


# Few-Shot Results
<table>
  <tr>
    <th rowspan="2">Set</th>
    <th rowspan="2">Model</th>
    <th colspan="2">CB</th>
    <th>RTE</th>
  </tr>
  <tr>
    <td align="center"><b>F<sub>1</sub></b></td>
    <td align="center"><b>Acc.</b></td>
    <td align="center"><b>Acc.</b></td>
  </tr>
  <tr>
    <td rowspan="7" align="center">dev</td>
  </tr>
  <tr>
    <td>GPT-3 Small</td>
    <td align="right">26.1</td>
    <td align="right">42.9</td>
    <td align="right">52.3</td>
  </tr>
  <tr>
    <td>GPT-3 Med</td>
    <td align="right">40.4</td>
    <td align="right">58.9</td>
    <td align="right">48.4</td>
  </tr>
  <tr>
    <td>GPT-3</td>
    <td align="right">57.2</td>
    <td align="right">82.1</td>
    <td align="right">72.9</td>
  </tr>
  <tr>
    <td>PET (ALBERT)</td>
    <td align="right">59.4</td>
    <td align="right">85.1</td>
    <td align="right">69.8</td>
  </tr>
  <tr>
    <td>iPET (ALBERT)</td>
    <td align="right">92.4</td>
    <td align="right">92.9</td>
    <td align="right">74.0</td>
  </tr>
  <tr>
    <td>WARP<sub>init</sub> (ALBERT) </td>
    <td align="right">84.0</td>
    <td align="right">87.5</td>
    <td align="right">71.8</td>
  </tr>
  <tr>
    <td rowspan="6" align="center">test</td>
  </tr>
  <tr>
    <td>GPT-3                             </td>
    <td align="right">52.0</td>
    <td align="right">75.6</td>
    <td align="right">69.0</td>
  </tr>
  <tr>
    <td>PET (ALBERT)                      </td>
    <td align="right">60.2</td>
    <td align="right">87.2</td>
    <td align="right">67.2</td>
  </tr>
  <tr>
    <td>iPET (ALBERT)                     </td>
    <td align="right">79.9</td>
    <td align="right">88.8</td>
    <td align="right">70.8</td>
  </tr>
  <tr>
    <td>WARP<sub>init</sub> (ALBERT) </td>
    <td align="right">70.2</td>
    <td align="right">82.4</td>
    <td align="right">69.1</td>
  </tr>
</table>
<sup>Results on SuperGLUE benchmark. The results for the test set are obtained from SuperGLUE evaluation server.
We only show systems performing in a similar few-shot training setup using 32 examples.</sup>

# Setup
The code requires YerevaNN's internal version of `allennlp`
```
git clone https://github.com/YerevaNN/allennlp
git checkout warp
pip install .
```

# Training


### Linear Probing
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

### WARP_0
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

## Citation
If you want to refer to our work use this bibTeX:
```
@inproceedings{hambardzumyan-etal-2021-warp,
    title = "{WARP}: {W}ord-level {A}dversarial {R}e{P}rogramming",
    author = "Hambardzumyan, Karen  and
      Khachatrian, Hrant  and
      May, Jonathan",
    booktitle = "Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.acl-long.381",
    doi = "10.18653/v1/2021.acl-long.381",
    pages = "4921--4933"
}
```
