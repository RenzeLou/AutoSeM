A reproduction for paper -- AutoSeM: Automatic Task Selection and Mixing in Multi-Task Learning [[paper](https://aclanthology.org/N19-1355/)] [[origin code](https://github.com/HanGuo97/AutoSeM)].


## Preparations
### 1. Build Environment

```bash
conda create -n autosem python=3.6.12
```

and then install all dependencies:

```bash
pip install -r requirements.txt
```

In case of an error interrupt, use the following command:

```bash
while read requirement; do pip install $requirement; done < requirements.txt
```

### 2. Data Processing

All the datasets have already been processed and split (see `data/GLUE`), however, you can do it by yourself as long as you:

1. Use the ``.ipyng`` file under each directory to process specific task, you can use ``jupyter notebook`` and the dependency is simple ``python3`` .After running the notebook, you will obtain ``{train/dev/test}.json`` files and a ``data_info.json``. 
2. Use the ``format.py`` to reprocess all the ``.json`` files, then you will get ``{train/dev/test}_format.json`` files.

### 3. Download Pre-trained Resources

 In our experiments, we need **ELMO** and **BERT family** (i.e., bert-base-uncased, bert-base-cased, bert-large-uncased, bert-large-cased, roberta-base, roberta-large).

- **ELMO**: We use allennlp's original [ELMO model](https://allennlp.org/elmo), you should put `weight.hdf5` and `option.json` to ``cache/elmo/``.
- **BERT**: We use [huggingface](https://github.com/huggingface)/**[transformers](https://github.com/huggingface/transformers)**, the BERT family can be download automatically (see the doc of huggingface). Pls cache all the pre-trained resources at ``cache/``.

## Auto Task Selection

```bash
python main.py --tasks MRPC_CoLA_MNLIMatched_QNLI_QQP_RTE_SST-2_WNLI --random_seed 42 --setting 0 --encoder_type 1 --learning_rate 0.001 --stage 1 --cuda 7 --max_steps 200 --train_batch_size 16 --train_batch_num 10 --eval_batch_size 8 --steps_per_eval 8 
```

## setting 1&2

*all auxilary task use same ratiio (1:1:1)*

*setting 2 will use the auxilary tasks chosen from auto task selection*

```bash
python main.py --task CoLA_MNLIMatched_MRPC_QNLI_QQP_RTE_SST-2_WNLI --random_seed 42 --setting 1 --encoder_type 1 --learning_rate 0.001 --stage 3 --cuda 1 --max_steps 200 --train_batch_size 16 --train_batch_num 50 --eval_batch_size 8 --steps_per_eval 40 
```

## setting 3

*training with auto ratio selection*

```bash
python main.py --task RTE_QQP_MNLIMatched --random_seed 42 --setting 3 --encoder_type 1 --learning_rate 0.001 --stage 2 --cuda 1 --max_steps 200 --train_batch_size 16 --train_batch_num 20 --eval_batch_size 8 --steps_per_eval 40 --trial_num 10
```

