#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: RenzeLou

import time
import argparse
import os
import json
import datetime
import numpy as np
import csv
from tqdm import tqdm, trange

parser = argparse.ArgumentParser()

# # split parameter group
# TRAIN = parser.add_argument_group(title="training args")
# MODEL_STRUCT = parser.add_argument_group(title="model structure")
# MTL = parser.add_argument_group(title="multi task learning param")
#
# # Training
# TRAIN.add_argument("--tasks",
#                    type=str, required=True, metavar="RTE_QNLI_MRPC_SST_CoLA",
#                    help="String, indicate chosen task split by '_',the first task will be take as primary one")
# TRAIN.add_argument("--max_steps",
#                    type=int, default=200, help="max tuning round of stage 1")
# TRAIN.add_argument("--train_batch_size",
#                    type=int, default=32, help="training (optimization) batch_size, reduce it if using BERT")
# TRAIN.add_argument("--train_batch_num", type=int, default=10,
#                    help="batch num of each training step")
# # TRAIN.add_argument("--batch_num_per_step",
# #                    type=int, help="training (optimization) batch_num")
# TRAIN.add_argument("--eval_batch_size",
#                    type=int, default=16, help="evaluation batch_size")
# TRAIN.add_argument("--steps_per_eval",
#                    type=int, default=1, help="1, when stage 1")
# TRAIN.add_argument("--random_sampler", action='store_true', help="whether use random sampler when training/evaluation")
# TRAIN.add_argument("--logdir",
#                    type=str, default="./cache")
# TRAIN.add_argument("--random_seed",
#                    type=int, default=42)
# TRAIN.add_argument("--cuda",
#                    type=str, default="0")
## warm up model on main task, this operation is important to CoLA
# TRAIN.add_argument("--warm_up_step", type=int, default=0, help="the wram up steps on main task")
#
# # Inference
# parser.add_argument("--infer", action="store_true", help="whether to predict on test set")
# parser.add_argument("--model_file_path", type=str, default=None, help="saved model parameters' path, used in inference")
# # other
# parser.add_argument("--setting", type=int,
#                     required=True, help="result save dir. " +
#                                         "0: no dir (auto task selection)" +
#                                         "1-3: setting 1-3")
#
# # -----------------------------------------
# # HYPER-PARAMETERS
#
# # Model (use default parameters from original codes)
# MODEL_STRUCT.add_argument("--encoder_type",
#                           type=int, required=True,
#                           metavar="1", help="1:RNN-LSTM\n2:BERT-base-cased")
# MODEL_STRUCT.add_argument("--full_name", type=str, default="bert-base-cased",
#                           help="name of bert variant, only take effect when use bert as encoder")
# # if chose rnn-lstm, then activate the parameters below
# # else use BERT, we will use the parameters similar to huggingface:
# # https://github.com/huggingface/transformers/blob/master/examples/text-classification/run_glue.py
# MODEL_STRUCT.add_argument("--embedding_dim",
#                           type=int, default=1024)
# # we use ELMO original so the dim is 1024, see at
# # https://allennlp.org/elmo
# MODEL_STRUCT.add_argument("--num_units",
#                           type=int, default=512)
# MODEL_STRUCT.add_argument("--num_layers",
#                           type=int, default=2)
# MODEL_STRUCT.add_argument("--dropout_rate",
#                           type=float, default=0.5)
# MODEL_STRUCT.add_argument("--learning_rate",
#                           type=float, default=0.001)
# MODEL_STRUCT.add_argument("--use_hidden_state", action="store_true",
#                           help="use hidden states of LSTM, otherwise use output, which is same as original paper")
#
# # MTL
# MTL.add_argument("--stage", type=int,required=True, help="1:auto task selection\n" +
#                                                       "2:auto mix ratios\n" +
#                                                       "3:run auxiliary task with ratio 1:1 straightforward")
# MTL.add_argument("--decay_ratio",
#                  type=float, default=0.3, help="no stationary MRB (\math{r} mentioned in paper)")
# MTL.add_argument("--trial_num", type=int, default=10, help="trial num of stage 2")
# # -----------------------------------------

# debug ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# split parameter group
TRAIN = parser.add_argument_group(title="training args")
MODEL_STRUCT = parser.add_argument_group(title="model structure")
MTL = parser.add_argument_group(title="multi task learning param")

TRAIN.add_argument("--tasks",
                   type=str, default="RTE",
                   metavar="CoLA_MNLIMatched_MRPC_QNLI_QQP_RTE_SST-2_WNLI",
                   help="String, indicate chosen task split by '_',the first task will be take as primary one")  # _CoLA_MRPC_QNLI_SST-2_WNLI
TRAIN.add_argument("--max_steps",
                   type=int, default=200, help="max tuning round of stage 1")
TRAIN.add_argument("--train_batch_size",
                   type=int, default=2, help="training (optimization) batch_size, reduce it if using BERT")
TRAIN.add_argument("--train_batch_num", type=int, default=50,
                   help="batch num of each training step")
# TRAIN.add_argument("--batch_num_per_step",
#                    type=int, help="training (optimization) batch_num")
TRAIN.add_argument("--eval_batch_size",
                   type=int, default=1, help="evaluation batch_size")
TRAIN.add_argument("--steps_per_eval",
                   type=int, default=20, help="1, when stage 1")
TRAIN.add_argument("--random_sampler", type=bool, default=True,
                   help="whether use random sampler when training/evaluation")  # action='store_true',
TRAIN.add_argument("--logdir",
                   type=str, default="./cache")
TRAIN.add_argument("--random_seed",
                   type=int, default=42)
TRAIN.add_argument("--cuda",
                   type=str, default="7")
# warm up model on main task, this operation is important to CoLA
TRAIN.add_argument("--warm_up_step", type=int, default=0, help="the wram up steps on main task")

# Inference
parser.add_argument("--infer", action="store_true", help="whether to predict on test set")
parser.add_argument("--model_file_path", type=str, default=None, help="saved model parameters' path, used in inference")
# other
parser.add_argument("--setting", type=int,
                    default=1, help="result save dir. " +
                                    "0: no dir (auto task selection)" +
                                    "1-3: setting 1-3")

# -----------------------------------------
# HYPER-PARAMETERS

# Model (use default parameters from original codes)
MODEL_STRUCT.add_argument("--encoder_type",
                          type=int, default=2,
                          metavar="1", help="1:RNN-LSTM\n2:BERT")
MODEL_STRUCT.add_argument("--full_name", type=str, default="bert-base-cased",
                          help="name of bert variant, only take effect when use bert as encoder")

# if chose rnn-lstm, then activate the parameters below
# else use BERT, we will use the parameters similar to huggingface:
# https://github.com/huggingface/transformers/blob/master/examples/text-classification/run_glue.py
MODEL_STRUCT.add_argument("--use_hidden_state", type=bool, default=False,
                          help="use hidden states of LSTM, otherwise use output")  # same as their code, use output of LSTM
MODEL_STRUCT.add_argument("--embedding_dim",
                          type=int, default=1024)
# we use ELMO original so the dim is 1024, see at
# https://allennlp.org/elmo
MODEL_STRUCT.add_argument("--num_units",
                          type=int, default=512)
MODEL_STRUCT.add_argument("--num_layers",
                          type=int, default=2)
MODEL_STRUCT.add_argument("--dropout_rate",
                          type=float, default=0.5)
MODEL_STRUCT.add_argument("--learning_rate",
                          type=float, default=5e-5, help="lstm:0.001; bert:5e-5")

# MTL
MTL.add_argument("--stage", type=int, default=3, help="1:auto task selection\n" +
                                                      "2:auto mix ratios\n" +
                                                      "3:run auxiliary task with ratio 1:1 straightforward")
MTL.add_argument("--decay_ratio",
                 type=float, default=0.3, help="no stationary MRB (\math{r} mentioned in paper)")
MTL.add_argument("--trial_num", type=int, default=10, help="trial num of stage 2")
# debug ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# # -----------------------------------------

args, unparsed = parser.parse_known_args()
if unparsed:
    raise ValueError(unparsed)

# check some argparse (save path)
if args.stage == 1:
    assert args.setting == 0, "auto task selection is setting 0"
elif args.stage == 2:
    assert args.setting == 3, "auto mix ratios is setting 3"
elif args.stage == 3:
    assert args.setting in [1, 2], "ratio 1:1... is setting 1/2"
else:
    raise KeyError("invalid stage!")

os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda

from utils.tool_box import generate_logger
from model.MultiTaskModel import MultiTaskBaseMoudle, AutoTaskSelectMoudle
from utils.tool_box import seed_torch

# save latest training info which name with datetime
# detail info, level == 1
now = datetime.datetime.now()
date_time = datetime.datetime.strftime(now, '%Y_%m_%d_%H_%M')
logger = generate_logger(log_dir=args.logdir, log_name=date_time + ".log")
logger.info(args)

save_path = "./checkpoints/"
save_path += "LSTM" if args.encoder_type == 1 else args.full_name
if args.setting != 0:
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    save_path = os.path.join(save_path, "setting" + str(args.setting))
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    save_path = os.path.join(save_path, args.tasks)

if not os.path.exists(save_path):
    os.mkdir(save_path)


# two running strategy
# ============================================================================================
def MTL_mix_ratios(mix_ratios: list, save_result: bool = True):
    ''' train multi task model with pre-defined mix ratios '''
    # set random seed
    seed_torch(args.random_seed)
    begin = time.time()
    flatten_ratios = [idx for idx, ratio in enumerate(mix_ratios) for _ in range(ratio)]
    # create base model
    model_args = {"task_name": args.tasks, "train_batch_size": args.train_batch_size,
                  "eval_batch_size": args.eval_batch_size,
                  "train_batch_num": args.train_batch_num, "random_sampler": args.random_sampler,
                  "encoder_type": args.encoder_type, "hidden": args.use_hidden_state,
                  "embedding_dim": args.embedding_dim, "num_units": args.num_units, "num_layers": args.num_layers,
                  "dropout_rate": args.dropout_rate, "learning_rate": args.learning_rate,
                  "full_model_name": args.full_name}
    model = MultiTaskBaseMoudle(**model_args)
    model._build(logger)
    logger.info("===> training begin <===")
    best_eval_score = dict()
    best_test_ans = None

    for i in tqdm(range(args.max_steps), desc="Step"):
        step = model.get_step()
        remenber = step % sum(mix_ratios)
        task_idx = flatten_ratios[remenber]
        if step < args.warm_up_step:
            task_idx = 0
        model._update_task(task_idx)
        loss = model._train()
        logger.info("loss:{}".format(loss))

        if i % args.steps_per_eval == 0 and i != 0:
            logger.info("===> evaluation begin <===")
            eval_score = model._eval()
            primary_metric = list(eval_score.values())[0]  # primary metric (F_1 & spearman)
            eval_history = model.get_history()
            logger.info("eval score: {}".format(eval_score))
            logger.info("eval_history:{}".format(eval_history))
            best_his_score = max(eval_history) if len(eval_history) > 0 else -1
            if primary_metric >= best_his_score:
                ans = None
                if save_result:
                    # save model
                    model._save(save_path)
                    logger.info("model saved to %s" % save_path)
                    # save score
                    score_name = os.path.join(save_path, args.tasks[0] + ".json")
                    with open(score_name, "w") as f:
                        json.dump(eval_score, f)
                    logger.info("best eval score saved to %s" % score_name)

                best_eval_score = eval_score

            model.append_history(primary_metric)

    # predict on test data and write to file
    file_name = os.path.join(save_path, args.tasks[0] + ".tsv")
    load_flag = model._load(save_path)
    if load_flag:
        logger.info("model load from %s" % save_path)
        best_test_ans = model._test(file_name)
        logger.info("test predictions saved to %s" % file_name)
    else:
        logger.warn("model loading failed!")

    logger.info("multi task training with mix ratio end!")

    end = time.time()
    run_time = (end - begin) / 60.
    # save running time
    if save_result:
        with open(os.path.join(save_path, "running_time.txt"), "w") as f:
            f.write("\n========\nTotal time consumption: %.3f (minutes)\n==========" % run_time)
    logger.info("\n========\nTotal time consumption: %.3f (minutes)\n==========" % run_time)

    return best_eval_score, best_test_ans, run_time


def MTL_auto_selection(observe: bool = False, save_result: bool = True):
    ''' use MAB to select auxiliary tasks, write task utility to a file '''
    # set random seed
    seed_torch(args.random_seed)
    begin = time.time()
    super_args = {"task_name": args.tasks, "train_batch_size": args.train_batch_size,
                  "eval_batch_size": args.eval_batch_size,
                  "train_batch_num": args.train_batch_num, "random_sampler": args.random_sampler,
                  "encoder_type": args.encoder_type, "hidden": args.use_hidden_state,
                  "embedding_dim": args.embedding_dim, "num_units": args.num_units, "num_layers": args.num_layers,
                  "dropout_rate": args.dropout_rate, "learning_rate": args.learning_rate,
                  "full_model_name": args.full_name}

    model_args = {"super_args": super_args, "prior_alpha": 1, "prior_beta": 1, "decay_rate": args.decay_ratio}
    # create base model
    model = AutoTaskSelectMoudle(**model_args)
    model._build(logger)
    logger.info("===> training begin <===")

    for i in tqdm(range(args.max_steps), desc="Step"):
        loss = model._train()
        print("loss:{}".format(loss))

        if i % args.steps_per_eval == 0 and i != 0:
            logger.info("===> evaluation begin <===")
            eval_score = model._eval()
            primary_metric = list(eval_score.values())[0]  # primary metric (F_1 & spearman)
            eval_history = model.get_history()
            logger.info("eval score: {}".format(eval_score))
            logger.info("eval_history:{}".format(eval_history))
            # update auto task selector
            next_task, sampled_means, sample_history = model._update(primary_metric)
            logger.info("--> now, task {} have been chosen".format(next_task))
            if observe:
                logger.info(
                    "========\nsampled_means:{}\nsample_history:{}\n==========".format(sampled_means, sample_history))
            model.append_history(primary_metric)

    end = time.time()
    run_time = (end - begin) / 60.

    # get task utility
    task_utility = model.get_utility()
    if save_result:
        task_utility["time_consume"] = "%.3f minutes" % run_time
    save_name = "_".join(args.tasks) + ".json"
    with open(os.path.join(save_path, save_name), "w") as f:
        json.dump(task_utility, f)
    logger.info("final task utility:{}".format(task_utility))
    logger.info("which has already saved to %s" % save_name)
    logger.info("auto task selection end!")
    logger.info("\n========\nTotal time consumption: %.3f (minutes)\n==========" % run_time)


# ================================================================================================
# main function
logger.info("begin experiment, chose stage {} ".format(args.stage))
args.tasks = args.tasks.split("_")
if args.stage == 1:
    # auto task selection
    if args.steps_per_eval != 1:
        args.steps_per_eval = 1
        logger.warn("You are running stage 1, set 'steps_per_eval' to 1 now!")
    MTL_auto_selection(observe=True, save_result=True)
elif args.stage == 2:
    # auto mix ratios via greedy trial
    mix_ratios_list = []
    while True:
        ratios = []
        for _ in range(len(args.tasks)):
            ratios.append(np.random.randint(10))
        if ratios.count(0) > len(args.tasks) - 2 or ratios[0] == 0:
            continue
        mix_ratios_list.append(ratios)
        if len(mix_ratios_list) == args.trial_num:
            break
    best_eval_score, best_test_ans, best_run_time, best_ratios = None, None, 0, None
    best_score = -1
    for i, mix_ratios in enumerate(mix_ratios_list):
        logger.info("~~~~~~~~~~ The {} trial ~~~~~~~~~~~~~~~~".format(i + 1))
        logger.info("~~~~~~~~~~ mix ratios :{} ~~~~~~~~~~~~~~~~".format(mix_ratios))
        eval_score, test_ans, run_time = MTL_mix_ratios(mix_ratios, save_result=True)
        score = list(eval_score.values())[0]
        if score > best_score:
            best_score = score
            best_eval_score = eval_score
            best_test_ans = test_ans
            best_run_time = run_time
            best_ratios = mix_ratios
    score_name = os.path.join(save_path, args.tasks[0] + "_best.json")
    with open(score_name, "w") as f:
        json.dump(best_eval_score, f)
    logger.info("===================================================")
    logger.info("overall best eval score :{}".format(best_eval_score))
    logger.info("overall best eval score saved to %s" % score_name)

    file_name = os.path.join(save_path, args.tasks[0] + "_best.tsv")
    with open(file_name, 'w', newline='') as f:
        tsv_w = csv.writer(f, delimiter='\t')
        tsv_w.writerow(['index', 'prediction'])
        tsv_w.writerows(best_test_ans)
    logger.info("===================================================")
    logger.info("overall best test predictions saved to %s" % file_name)

    with open(os.path.join(save_path, "best_running_time.txt"), "w") as f:
        f.write("\n========\nTotal time consumption: %.3f (seconds)\n==========" % best_run_time)

    logger.info("===================================================")
    logger.info("The final mix ratios is :{}".format(best_ratios))
    with open(os.path.join(save_path, "best_mix_ratios.json"), "w") as f:
        json.dump(best_ratios, f)
elif args.stage == 3:
    # training and eval model with default 1:1:1... ratios (setting 1&2)
    mix_ratios = [1] * len(args.tasks)
    best_eval_score, best_test_ans, run_time = MTL_mix_ratios(mix_ratios, save_result=True)
