### this file is broken because of time and ram comsuming on MNLI data

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: RenzeLou
# a implementation of AutoSem (multi task select model)
# Multi task learning with multi arms bandits
import os
import json
from overrides import overrides

import math
import numpy as np
import warnings
import csv
import logging
import pickle
from tqdm import tqdm
from nltk.tokenize import WordPunctTokenizer

import torch
from transformers.data.processors.glue import glue_convert_examples_to_features
from transformers.data.processors.utils import InputFeatures
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from allennlp.modules.elmo import Elmo, batch_to_ids

from multitask.task_constant import TASK_INFO
from utils.data_loader import myDataIter, simple_data_loader
from utils.processor import Processor, InputFeatures_ELMO, InputFeatures_BERT, InputExample
from model.LanguageModel import NeuralEncoder, RnnLstmEncoder, BertBaseEncoder

from sklearn.metrics import f1_score, accuracy_score, matthews_corrcoef
from scipy.stats import spearmanr, kendalltau, pearsonr


class MultiTaskBaseMoudle(object):
    def __init__(self, task_name: list, train_batch_size: int, eval_batch_size: int,
                 train_batch_num: int, random_sampler: bool, encoder_type=1, hidden=False,
                 embedding_dim=1024, num_units=512, num_layers=2, dropout_rate=0.5, learning_rate=0.001):
        # check task name
        for name in task_name:
            assert name in TASK_INFO["ALL_TASK"], "task %s is not in pre-defined list!" % name
        # check model choice
        assert encoder_type in [1, 2], "invalid model choice"

        self._task_name = task_name
        self._task_idx = 0
        self._random = random_sampler
        self._encoder_type = encoder_type
        self._global_step = 0
        self._eval_history = []

        self._train_loader = []
        self._dev_loader = []
        self._test_loader = []
        self._train_batch_size = train_batch_size
        self._train_batch_num = train_batch_num
        self._eval_batch_size = eval_batch_size
        self._test_batch_size = 8

        # some LSTM model hy-param are useless when use BERT as encoder
        if self._encoder_type == 1:
            # two layers LSTM
            self._model = RnnLstmEncoder(embedding_dim, num_units, num_layers, dropout_rate, learning_rate, task_name,
                                         hidden)
        elif self._encoder_type == 2:
            # TODO:BERT
            self._model = BertBaseEncoder()
        else:
            raise KeyError("invalid model choice")

        self._idx2labels = self._model._idx2labels
        self._processor = Processor(self._idx2labels, self._encoder_type)

    def _build(self, logger: logging):
        ''' build multi task data loader and initialize model structure (encoder & dense layer)

        :return: NULL
        '''
        logger.info("=== Build model ===")
        self._model._build()

        logger.info("=== Build data ===")
        # only build eval/test iter on main task
        for i, task in enumerate(self._task_name):
            train_it, dev_it, test_it = self._buld_data(task, i, self._random, logger, self._train_batch_size,
                                                        self._eval_batch_size)
            self._train_loader.append(train_it)
            self._dev_loader.append(dev_it)
            self._test_loader.append(test_it)

    def _update_task(self, task_idx: int):
        ''' used as an external call interface '''
        self._task_idx = task_idx

    # def _load_data(self, subset: int):
    #     ''' load next batch data {"index","seq1","seq2","label"} according to specific task and subset (train/dev/test)
    #     :param: subset -> 1:train; 2:dev; 3:test
    #     :return: None, if one epoch end, else return batch data
    #     '''
    #     batch = None
    #     if subset == 1:
    #         # train data
    #         train_batch = self._train_loader[self._task_idx].next()  # training data don't need to stop
    #         batch = train_batch
    #     elif subset == 2:
    #         # dev data
    #         dev_batch = self._dev_loader[0].next()
    #         if not self._dev_loader[0].if_reset():
    #             batch = dev_batch
    #     else:
    #         # test data
    #         test_batch = self._test_loader[0].next()
    #         if not self._test_loader[0].if_reset():
    #             batch = test_batch
    #
    #     return batch

    def _train(self):
        ''' one step training model according to mix_ratios

        :return: loss
        '''
        all_loss = []
        train_loader = self._train_loader[self._task_idx]
        i = 0
        for i, batch in enumerate(train_loader):
            loss = self._model._train(batch, self._task_idx)
            all_loss.append(loss)
            if i == self._train_batch_num - 1:
                break
        if i < self._train_batch_num - 1:
            for j, batch in enumerate(train_loader):
                loss = self._model._train(batch, self._task_idx)
                all_loss.append(loss)
                if j == self._train_batch_num - 2 - i:
                    break
        self._global_step += 1

        return np.mean(all_loss)

    def _eval(self):
        '''  evaluate model on main task's whole dev data.

        :return: score -> dict
        '''
        pred = []
        labels = []
        dev_loader = tqdm(self._dev_loader[0])
        for dev_batch in dev_loader:
            batch_pred, batch_label, _, _ = self._model._eval(dev_batch, 0)
            pred += batch_pred
            labels += batch_label

        score = dict()
        main_task = self._task_name[0]
        if main_task in TASK_INFO['REGRESS']:
            score['spear'] = spearmanr(pred, labels)[0]
            score['pearson'] = pearsonr(pred, labels)[0]
        else:
            if main_task == "CoLA":
                score['matthews'] = matthews_corrcoef(labels, pred)
                score['f_1'] = f1_score(pred, labels, average='macro')
            else:
                score['f_1'] = f1_score(pred, labels, average='macro')
                score['acc'] = accuracy_score(pred, labels)

        # check NAN
        for k, v in score.items():
            if math.isnan(v):
                warnings.warn("The eval score is NAN, which is not expected (set to 0.0 now)")
                score[k] = 0.0

        return score

    def _test(self, file_path):
        ''' fetch all test predictions and write to a .tsv

        :return: ans -> list
        '''
        pred_ori = []
        index = []
        test_loader = tqdm(self._test_loader[0])
        for test_batch in test_loader:
            _, _, batch_pred_ori, batch_index = self._model._eval(test_batch, 0)
            pred_ori += batch_pred_ori
            index += batch_index

        assert len(pred_ori) == len(index), "unknown error!"

        ans = []
        for i in range(len(pred_ori)):
            ans.append([str(index[i]), str(pred_ori[i])])
        # write the result to a .tsv file
        with open(file_path, 'w', newline='') as f:
            tsv_w = csv.writer(f, delimiter='\t')
            tsv_w.writerow(['index', 'prediction'])
            tsv_w.writerows(ans)

        return ans

    def seq_to_ids(self, data, max_seq_len=512, lower=False):
        assert isinstance(data, list)
        if data[0] == '-1':
            return list(map(int,data))
        else:
            # use nltk to tokenize sentence
            with torch.no_grad():
                tokenizer = WordPunctTokenizer()
                new_data = []
                for sentence in data:
                    sentence = sentence.lower() if lower else sentence
                    sen_tok = tokenizer.tokenize(sentence)[:max_seq_len]
                    new_data.append(sen_tok)
                character_ids = batch_to_ids(new_data)

            return character_ids.detach().numpy().tolist()


    def _convert_features(self, input_examples: any, label_list: list, task_idx: int):
        output_mode = 'regression' if self._task_name[task_idx] in TASK_INFO["REGRESS"] else 'classification'
        if self._encoder_type == 1:
            # feed to elmo, get embeddings
            assert isinstance(input_examples, InputExample)
            index = input_examples.guid
            label = input_examples.label

            character_a = self.seq_to_ids(input_examples.text_a)
            character_b = self.seq_to_ids(input_examples.text_b)

            instance_num = len(character_a)
            input_features = []
            for i in range(instance_num):
                input_features.append(InputFeatures_ELMO(int(index[i]),character_a[i],character_b[i],label[i]))
            # gpu = True
            # with torch.no_grad():
            #     self._model._elmo.cuda()
            #     character_a = character_a.cuda()
            #     result = self._model._elmo(character_a)
            #     embed_1 = result["elmo_representations"][0]
            #     if character_b is not None:
            #         character_b = character_b.cuda()
            #         result = self._model._elmo(character_b)
            #         embed_2 = result["elmo_representations"][0]
            #     else:
            #         embed_2 = None
            # embed_1, mask_1 = self._model.fetch_elmo(character_a)
            # embed_2, mask_2 = self._model.fetch_elmo(character_b)
            #
            # instance_num = embed_1.shape[0]
            # embed_1 = embed_1.detach()
            # embed_1 = embed_1.cpu() if gpu else embed_1
            # if embed_2 is not None:
            #     assert embed_2.shape[0] == instance_num
            #     embed_2 = embed_2.detach()
            #     embed_2 = embed_2.cpu() if gpu else embed_2
            # if label is not None:
            #     assert len(label) == instance_num
            #
            # input_features = []
            # for i in range(instance_num):
            #     e1 = embed_1[i]
            #     e1 = e1.numpy().tolist()
            #     if embed_2 is not None:
            #         e2 = embed_2[i]
            #         e2 = e2.numpy().tolist()  # [max_seq_len,1024]
            #     else:
            #         e2 = -1
            #     if label[0] != '-1':
            #         lb = label[i]
            #     else:
            #         lb = '-1'
            #     idx = int(index[i])
            #     example = InputFeatures_ELMO(idx, e1, e2, lb)
            #     input_features.append(example)
        else:
            assert type(input_examples) == list
            input_features_ori = glue_convert_examples_to_features(input_examples, self._model.tokenizer,
                                                                   TASK_INFO["BERT_MAX_LEN"],
                                                                   label_list=label_list, output_mode=output_mode)
            assert len(input_features_ori) == len(input_examples)
            input_features = []
            for i in range(len(input_examples)):
                input_ids = input_features_ori[i].input_ids
                attention_mask = input_features_ori[i].attention_mask
                token_type_ids = input_features_ori[i].token_type_ids
                label = input_features_ori[i].label
                ins = InputFeatures_BERT(int(input_examples[i].guid), input_ids, attention_mask, token_type_ids, label)
                input_features.append(ins)

        return input_features

    def _build_loader(self, features, output_mode, batch_size, random: bool = False):
        if self._encoder_type == 2:
            all_guid = torch.tensor([f.guid for f in features], dtype=torch.int)
            all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
            all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
            all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
            if output_mode == "classification":
                all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
            else:
                all_labels = torch.tensor([f.label for f in features], dtype=torch.float)
            dataset = TensorDataset(all_guid, all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
        elif self._encoder_type == 1:
            all_guid = torch.tensor([f.guid for f in features], dtype=torch.int)
            # all_embed_1 = torch.tensor([f.embedding_1 for f in features], dtype=torch.float)
            # all_embed_2 = torch.tensor([f.embedding_2 for f in features], dtype=torch.float)
            all_embed_1 = torch.tensor([f.embedding_1 for f in features], dtype=torch.long)
            all_embed_2 = torch.tensor([f.embedding_2 for f in features], dtype=torch.long)
            if output_mode == "classification":
                all_labels = torch.tensor([int(f.label) for f in features], dtype=torch.long)
            else:
                all_labels = torch.tensor([float(f.label) for f in features], dtype=torch.float)
            dataset = TensorDataset(all_guid, all_embed_1, all_embed_2, all_labels)
        else:
            raise RuntimeError

        sampler = RandomSampler(dataset) if random else SequentialSampler(dataset)
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)

        return dataloader

    def _buld_data(self, task: str, task_idx: int, random: bool, logger: logging, train_batch_size: int,
                   eval_batch_size: int, test_batch_size: int = 8) -> tuple:
        ''' load train/dev/test data and create torch datasets with sampler

        :return: three DataLoaderLoader
        '''
        logger.info("--> begin loading and encapsulating task %s <--" % task)
        if task == "MNLI":
            task += "Matched"
        data_path = os.path.join(TASK_INFO['PATH_TO_DATA'], task)
        model_name = "lstm" if self._encoder_type == 1 else "bert"
        train_path = os.path.join(data_path, "cached_{}_train.pkl".format(model_name))
        dev_path = os.path.join(data_path, "cached_{}_dev.pkl".format(model_name))
        test_path = os.path.join(data_path, "cached_{}_test.pkl".format(model_name))

        train_features, dev_features, test_features = None, None, None
        if os.path.isfile(train_path):
            # train_features = torch.load(train_path)
            with open(train_path, "rb") as f:
                train_features = pickle.load(f)
            logger.info("train_features loaded from %s" % train_path)
        else:
            # create features
            logger.info("train file not found, begin pre-processing, this can be time consuming...")
            label_list = self._processor.get_labels(task_idx)
            train_data = self._processor.get_examples(data_path, "train_format", task_idx)
            train_features = self._convert_features(train_data, label_list, task_idx)
            # torch.save(train_features, train_path)
            with open(train_path, "wb") as f:
                pickle.dump(train_features, f)
            logger.info("train_features have been cached to %s" % train_path)

        if task_idx == 0:
            if os.path.isfile(dev_path):
                # dev_features = torch.load(dev_path)
                with open(dev_path, "rb") as f:
                    dev_features = pickle.load(f)
                logger.info("dev_features loaded from %s" % dev_path)
            else:
                # create features
                logger.info("dev file not found, begin pre-processing, this can be time consuming...")
                label_list = self._processor.get_labels(task_idx)
                dev_data = self._processor.get_examples(data_path, "dev_format", task_idx)
                dev_features = self._convert_features(dev_data, label_list, task_idx)
                # torch.save(dev_features, dev_path)
                with open(dev_path, "wb") as f:
                    pickle.dump(dev_features, f)
                logger.info("dev_features have been cached to %s" % dev_path)
            if os.path.isfile(test_path):
                # test_features = torch.load(test_path)
                with open(test_path, "rb") as f:
                    test_features = pickle.load(f)
                logger.info("test_features loaded from %s" % test_path)
            else:
                # create features
                logger.info("test file not found, begin pre-processing, this can be time consuming...")
                label_list = self._processor.get_labels(task_idx)
                test_data = self._processor.get_examples(data_path, "test_format", task_idx)
                test_features = self._convert_features(test_data, label_list, task_idx)
                # torch.save(test_features, test_path)
                with open(test_path, "wb") as f:
                    pickle.dump(test_features, f)
                logger.info("test_features have been cached to %s" % test_path)

        output_mode = 'regression' if self._task_name[task_idx] in TASK_INFO["REGRESS"] else 'classification'
        # build train data loader
        train_data_loader = self._build_loader(train_features, output_mode, train_batch_size, random)
        # build dev data loader
        eval_data_loader = self._build_loader(dev_features, output_mode, eval_batch_size,
                                              False) if task_idx == 0 else None
        # build test data loader
        test_data_loader = self._build_loader(test_features, output_mode, test_batch_size,
                                              False) if task_idx == 0 else None

        return train_data_loader, eval_data_loader, test_data_loader

    def _save(self, save_path):
        self._model._save(save_path)

    def _load(self, save_path):
        flag = self._model._load(save_path)
        return flag

    def get_step(self):
        return self._global_step

    def get_history(self):
        return self._eval_history

    def append_history(self, score):
        self._eval_history.append(score)


class AutoTaskSelectMoudle(MultiTaskBaseMoudle):
    def __init__(self, super_args: dict, prior_alpha=1, prior_beta=1, decay_rate=0.0):
        super(AutoTaskSelectMoudle, self).__init__(**super_args)
        self._task_selector = BernoulliBanditTS(len(self._task_name), prior_alpha=prior_alpha, prior_beta=prior_beta,
                                                decay_rate=decay_rate)

    def _update(self, score: float):
        ''' use the eval score to update MAB and self._task_idx

        :param score: evaluation score
        :return: (next_task, sampled_means, sample_history) -> for observation
        '''
        # apply action and get reward
        self._task_selector.update(score, self._eval_history, self._task_idx)

        # sample next action
        next_action, sampled_means = self._task_selector.sample()
        self._update_task(next_action)

        return self._task_name[next_action], sampled_means, self._task_selector.get_sample_hist()

    def get_utility(self) -> dict:
        ''' get each task's utility (E of beta distribution)

        :return: task_utility -> dict
        '''
        utility = self._task_selector.draw_expection()
        assert len(utility) == len(self._task_name)
        return dict(zip(self._task_name, utility))


def random_argmax(vector):
    """Helper function to select argmax at random... not just first one."""
    index = np.random.choice(np.where(vector == vector.max())[0])
    return index


class BernoulliBanditTS(object):

    def __init__(self,
                 num_actions,
                 prior_alpha=1,
                 prior_beta=1,
                 decay_rate=0.0):
        """
        Args:
            prior_alpha:
                Prior parameters for Beta Distribution
            prior_beta:
                Prior parameters for Beta Distribution
            decay_rate:
                How quickly uncertainty is injected. Set to
                non-zero values will effectly create a non-stationary TS bandit

        """
        super(BernoulliBanditTS, self).__init__()

        self._num_actions = num_actions
        self._prior_alpha = prior_alpha
        self._prior_beta = prior_beta
        self._decay_rate = decay_rate
        self._parameters = [self._get_param(prior_alpha, prior_beta) for _ in range(num_actions)]

        self._sample_histories = [0]

    def _get_param(self, alpha, beta):
        return {"Alpha": alpha, "Beta": beta}

    def _reward_shaping_fn(self, score: float, history: list):
        if len(history) == 0:
            return 0.0
        else:
            return float(score >= history[-1])

    def get_sample_hist(self):
        return self._sample_histories

    @property
    def alphas(self):
        return [p['Alpha'] for p in self._parameters]

    @property
    def betas(self):
        return [p['Beta'] for p in self._parameters]

    def draw_expection(self):
        return [p['Alpha'] / (p['Alpha'] + p['Beta']) for p in self._parameters]

    def sample(self):
        ''' Thompson sampling '''
        sampled_means = np.random.beta(self.alphas, self.betas)  # sample utility from each task's distribution
        chosen_arm = random_argmax(sampled_means)  # chosen task <- argmax beta mean

        assert chosen_arm in range(self._num_actions), "unknown error!"
        self._sample_histories.append(chosen_arm)

        return chosen_arm, sampled_means

    def update(self, score, history, chosen_arm):
        reward = self._reward_shaping_fn(score, history)

        if reward not in [0.0, 1.0]:
            raise ValueError("`shaped_reward` should be a Bernoulli variable")

        # All values decay slightly, observation updated
        for arm in range(self._num_actions):
            self._parameters[arm]['Alpha'] = (1 - self._decay_rate) * self._parameters[arm][
                'Alpha'] + self._decay_rate * self._prior_alpha
            self._parameters[arm]['Beta'] = (1 - self._decay_rate) * self._parameters[arm][
                'Beta'] + self._decay_rate * self._prior_beta

        self._parameters[chosen_arm]['Alpha'] += reward
        self._parameters[chosen_arm]['Beta'] += 1 - reward
