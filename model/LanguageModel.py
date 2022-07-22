#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: RenzeLou
# This file implement AutoSem model structure
# while set two alternative neural encoder (rnn-lstm & bert)
# which can apply to MTL

import abc
import os
import json
import copy
from overrides import overrides
from nltk.tokenize import WordPunctTokenizer

import torch
from torch.nn import LSTM, Linear, ModuleList, Dropout, CrossEntropyLoss, MSELoss
from torch.optim import Adam
import torch.nn.functional as F
from transformers import AutoConfig, AutoTokenizer, AutoModel, AdamW, get_linear_schedule_with_warmup

from allennlp.modules.elmo import Elmo, batch_to_ids
from multitask.task_constant import TASK_INFO


def get_task_label_num(task_name: list):
    ''' read task data_info.json to get task label num and idx2lb(used to get prediction on test set) '''
    idx2labels = []
    for name in task_name:
        if name == "MNLI":
            name += "Matched"
        data_path = os.path.join(TASK_INFO['PATH_TO_DATA'], name)
        with open(os.path.join(data_path, "data_info.json"), "r", encoding="utf-8") as f:
            data_info = json.load(f)
            idx2lb = data_info["labels"]  # dict
            idx2labels.append(idx2lb)
    return idx2labels


class NeuralEncoder(metaclass=abc.ABCMeta):
    ''' used for sentence pair forward (similar to siamese net) '''

    def __init__(self, task_name, hidden):
        self._task_name = task_name
        self._project_layer = None  # ModuleList
        # label(index value) -> true label
        self._idx2labels = get_task_label_num(task_name)
        self._encoder = None  # nn.Module
        self._optimizer = []
        self._dropout = None
        self._criterion = []  # create task criterion, 2 categories namely CLS and REG
        self._hidden = hidden

    def _trainable(self, task_idx: int):
        ''' make the encoder and project layers trainable '''
        self._encoder.train()
        self._project_layer[task_idx].train()

    def _testable(self, task_idx: int):
        ''' make the encoder and project layers testable '''
        self._encoder.eval()
        self._project_layer[task_idx].eval()

    def _transfer(self, labels: list, task_idx: int):
        ''' transfer label index to original label '''
        appromix = lambda x: round(x, 3)
        if self._task_name[task_idx] in TASK_INFO["REGRESS"]:
            return list(map(appromix, labels))
        idx2labels = self._idx2labels[task_idx]
        ori_labels = []
        for label in labels:
            ori_label = idx2labels[str(label)]
            ori_labels.append(ori_label)
        return ori_labels

    @abc.abstractmethod
    def _build(self):
        pass

    @abc.abstractmethod
    def _train(self, batch_data, task_idx) -> float:
        pass

    @abc.abstractmethod
    def _eval(self, batch_data, task_idx):
        pass

    @abc.abstractmethod
    def _save(self, save_path):
        pass

    @abc.abstractmethod
    def _load(self, save_path) -> bool:
        pass


class RnnLstmEncoder(NeuralEncoder):
    ''' two layer BI-LSTM with ELMO embedding, as mentioned in original paper '''

    def __init__(self, embedding_dim=1024, num_units=512, num_layers=2, dropout_rate=0.5, learning_rate=0.001,
                 task_name=[], hidden=False, full_model_name="bert-base-cased", weight_decay=0.0):
        super(RnnLstmEncoder, self).__init__(task_name, hidden)
        # pre-defined data path
        self.options_file = os.path.join(TASK_INFO['PATH_TO_ELMO'], "option.json")
        self.weight_file = os.path.join(TASK_INFO['PATH_TO_ELMO'], "elmo_original_weights.hdf5")
        # model hy-param
        self._embedding_dim = embedding_dim  # the elmo representation dim which is taken as a hy-param of LSTM model
        self._num_units = num_units
        self._num_layers = num_layers
        self._dropout_rate = dropout_rate
        self._learning_rate = learning_rate
        self._weight_decay = weight_decay

    @overrides
    def _build(self):
        # -----------------------------------------------------------------
        # elmo
        self._elmo = Elmo(self.options_file, self.weight_file, 1, requires_grad=False,
                          dropout=0)  # chose calculate method 1, sum 3 layers' outputs
        # self._dropout = Dropout(self._dropout_rate)
        # --------------------------------------------------------------------
        # Bi-LSTM
        args = {'input_size': self._embedding_dim, 'hidden_size': self._num_units,
                'num_layers': self._num_layers, 'batch_first': True,
                'dropout': self._dropout_rate, 'bidirectional': True}
        self._encoder = LSTM(**args)
        # -----------------------------------------------------------------
        # project (fully connected) layer
        project_layer = []
        for i, name in enumerate(self._task_name):
            in_dim = self._num_units if name not in TASK_INFO["DUAL"] else self._num_units * 4  # cat[u,v,u*v,|u-v|]
            in_dim = in_dim * 2 if not self._hidden else in_dim
            out_dim = max(1, len(self._idx2labels[i]))  # label num
            dense = Linear(in_dim, out_dim)
            project_layer.append(dense)
        self._project_layer = ModuleList(project_layer)
        # ----------------------------------------------------------------
        # create optimizers ans criterion
        for idx, dense_layer in enumerate(self._project_layer):
            if self._task_name[idx] in TASK_INFO["REGRESS"]:
                self._criterion.append(MSELoss())
            else:
                self._criterion.append(CrossEntropyLoss())
            params = []
            for name, param in self._encoder.named_parameters():
                if param.requires_grad == True:
                    if "weight" in name:
                        params += [{"params": param, "lr": self._learning_rate, "weight_decay": self._weight_decay}]
                    elif "bias" in name:
                        params += [{"params": param, "lr": self._learning_rate}]
            params += [{"params": list(dense_layer.parameters())[0], "lr": self._learning_rate,
                        "weight_decay": self._weight_decay}]
            params += [{"params": list(dense_layer.parameters())[1], "lr": self._learning_rate}]
            # each task has it's own optimizer
            optimizer = Adam(params)
            self._optimizer.append(optimizer)

        # move encoder/dense/criterion to cuda
        self._elmo.cuda()
        for param in self._elmo.parameters():
            param.requires_grad = False
        self._encoder.cuda()
        self._project_layer.cuda()
        for i in range(len(self._task_name)):
            self._criterion[i] = self._criterion[i].cuda()

    @overrides
    def _train(self, batch_data: dict, task_idx: int) -> float:
        ''' use given batch_data to train lstm

        :return: loss -> float
        '''
        # ---------------------------------------------------
        # one batch training
        # self._project_layer[task_idx].cuda()
        self._trainable(task_idx)
        assert len(batch_data.keys()) == 4
        # guid, embed_1, embed_2, labels = tuple([t.cuda() if i != 0 else t for i, t in enumerate(batch_data)])
        # guid, embed_1, embed_2, labels = ([v for v in batch_data.values()])
        # get batch_data
        batch_text_a = [sen.split() for sen in batch_data["text_a"]]
        batch_text_b = [sen.split() for sen in batch_data["text_b"]]

        # characterize
        embed_1 = batch_to_ids(batch_text_a)
        embed_2 = batch_to_ids(batch_text_b) if batch_text_b[0][0] != "-1" else None

        embed_1 = embed_1.cuda()
        regression = self._task_name[task_idx] in TASK_INFO["REGRESS"]
        # labels = torch.tensor(batch_data["label"], dtype=torch.float) if regression else torch.tensor(
        #     batch_data["label"], dtype=torch.long)
        labels = batch_data["label"]
        labels = labels.cuda()
        embed_1 = self._elmo(embed_1)["elmo_representations"][0]
        state_1, (_, _) = self._encoder(embed_1)
        pooled_1 = torch.max(state_1, dim=1)[0]  # [batch_size,512]
        if embed_2 is not None:
            embed_2 = embed_2.cuda()
            embed_2 = self._elmo(embed_2)["elmo_representations"][0]
            state_2, (_, _) = self._encoder(embed_2)
            pooled_2 = torch.max(state_2, dim=1)[0]  # [batch_size,512]
            # concat
            multiply = torch.mul(pooled_1, pooled_2)
            substract = torch.abs(pooled_1 - pooled_2)
            concat = torch.cat((pooled_1, pooled_2, multiply, substract), dim=1)  # [batch_size,4 * 512]
            out = self._project_layer[task_idx](concat)  # [batch_size,num_class]
        else:
            out = self._project_layer[task_idx](pooled_1)  # [batch_size,num_class]

        assert len(out.shape) == 2, "unknown error"
        if regression:
            labels = labels.unsqueeze(-1).float()
            pred = out.detach().cpu()
        else:
            labels = labels.long()
            pred = F.softmax(out, dim=1)  # note that torch cross entropy has already included log_softmax
            pred = torch.argmax(pred.detach().cpu(), dim=1, keepdim=True)

        loss = self._criterion[task_idx](out, labels)

        self._optimizer[task_idx].zero_grad()
        loss.backward()
        self._optimizer[task_idx].step()
        # ---------------------------------------------------
        # transfer label
        # prediction = self._transfer(pred.squeeze().tolist())

        # self._project_layer[task_idx].cpu()  # save GPU memory, we only move project layer to GPU when necessary

        return loss.item()

    @overrides
    def _eval(self, batch_data: dict, task_idx: int):
        '''  pass given batch_data to model, get eval result

        :return: pred_flatten -> list, model predictions
                 label_flatten -> list, target label
                 prediction -> list, model predictions which had been convert to original style
                 index -> list, instance serial number
        '''
        # ---------------------------------------------------
        # one batch evaluation
        self._testable(task_idx)
        with torch.no_grad():
            # self._project_layer[task_idx].cuda()
            assert len(batch_data) == 4
            size = len(batch_data["text_a"])
            # get batch_data
            batch_text_a = [sen.split() for sen in batch_data["text_a"]]
            batch_text_b = [sen.split() for sen in batch_data["text_b"]]

            # characterize
            embed_1 = batch_to_ids(batch_text_a)
            embed_2 = batch_to_ids(batch_text_b) if batch_text_b[0][0] != "-1" else None
            regression = self._task_name[task_idx] in TASK_INFO["REGRESS"]

            labels = batch_data["label"].float() if regression else batch_data["label"].long()
            label_flatten = labels.squeeze().tolist()
            index = batch_data["guid"].squeeze().tolist()
            # if size == 1:
            #     label_flatten = batch_data["label"].squeeze().unsqueeze(0).tolist()
            #     index = batch_data["guid"].squeeze().unsqueeze(0).tolist()

            embed_1 = embed_1.cuda()
            embed_1 = self._elmo(embed_1)["elmo_representations"][0]
            state_1, (_, _) = self._encoder(embed_1)
            pooled_1 = torch.max(state_1, dim=1)[0]  # [batch_size,512]
            if embed_2 is not None:
                embed_2 = embed_2.cuda()
                embed_2 = self._elmo(embed_2)["elmo_representations"][0]
                state_2, (_, _) = self._encoder(embed_2)
                pooled_2 = torch.max(state_2, dim=1)[0]  # [batch_size,512]
                # concat
                multiply = torch.mul(pooled_1, pooled_2)
                substract = torch.abs(pooled_1 - pooled_2)
                concat = torch.cat((pooled_1, pooled_2, multiply, substract), dim=1)  # [batch_size,4 * 512]
                out = self._project_layer[task_idx](concat)  # [batch_size,num_class]
            else:
                out = self._project_layer[task_idx](pooled_1)  # [batch_size,num_class]

            # get prediction
            assert len(out.shape) == 2, "unknown error"
            if regression:
                pred = out.detach().cpu().squeeze().tolist()
            else:
                pred = F.softmax(out, dim=1)  # note that torch cross entropy has already included log_softmax
                pred = torch.argmax(pred.detach().cpu(), dim=1, keepdim=False)
                pred = pred.squeeze().tolist()

            if size == 1:
                pred = [pred]
                label_flatten = [label_flatten]
            # ---------------------------------------------------
            # transfer label
            prediction = self._transfer(pred, task_idx)

            # self._project_layer[task_idx].cpu()  # save GPU memory, we only move project layer to GPU when necessary

        return pred, label_flatten, prediction, index

    @overrides
    def _save(self, save_path):
        encoder_path = os.path.join(save_path, "encoder")
        dense_path = os.path.join(save_path, "project_layers")
        if not os.path.exists(encoder_path):
            os.mkdir(encoder_path)
        if not os.path.exists(dense_path):
            os.mkdir(dense_path)
        # save LSTM
        torch.save({'state_dict': self._encoder.state_dict()}, os.path.join(encoder_path, "Bi-LSTM.pth.tar"))
        # save project_layers
        torch.save({'state_dict': self._project_layer.state_dict()}, os.path.join(dense_path, "dense.pth.tar"))

    @overrides
    def _load(self, save_path) -> bool:
        encoder_path = os.path.join(save_path, "encoder")
        dense_path = os.path.join(save_path, "project_layers")
        if not os.path.exists(encoder_path):
            return False
        if not os.path.exists(dense_path):
            return False
        try:
            # load LSTM
            lstm_checkpoint = torch.load(os.path.join(encoder_path, "Bi-LSTM.pth.tar"))
            self._encoder.load_state_dict(lstm_checkpoint['state_dict'])
            # load project_layers
            dense_checkpoint = torch.load(os.path.join(dense_path, "dense.pth.tar"))
            self._project_layer.load_state_dict(dense_checkpoint['state_dict'])
        except FileNotFoundError:
            print("model checkpoints file missing: %s" % save_path)
            return False
        return True

    def get_elmo_param(self):
        return self._elmo

    def fetch_elmo(self, character_ids, gpu=True):
        ''' fetch elmo embedding '''
        if character_ids is None:
            return (None, None)
        else:
            if gpu:
                self._elmo.cuda()
                character_ids = character_ids.cuda()
            result = self._elmo(character_ids)
            representation = result["elmo_representations"][0]
            mask = result["mask"]

            return (representation, mask)

    def _fetch_lstm_hidden(self, embed_1, embed_2=None, dropped=False):
        ''' fetch lstm hidden states '''
        if dropped:
            self._dropout.cuda()
            embed_1 = self._dropout(embed_1)
            embed_2 = self._dropout(embed_2) if embed_2 is not None else None
        _, (state_1, _) = self._encoder(embed_1)
        _, (state_2, _) = self._encoder(embed_2) if embed_2 is not None else (None, (None, None))
        return (state_1, state_2)

    def _fetch_lstm_output(self, embed_1, embed_2=None, dropped=False):
        ''' fetch lstm output states '''
        if dropped:
            self._dropout.cuda()
            embed_1 = self._dropout(embed_1)
            embed_2 = self._dropout(embed_2) if embed_2 is not None else None
        out_1, (_, _) = self._encoder(embed_1)
        out_2, (_, _) = self._encoder(embed_2) if embed_2 is not None else (None, (None, None))
        return (out_1, out_2)

    def seq_to_ids(self, data):
        if isinstance(data, torch.Tensor):
            return None
        elif isinstance(data, list):
            # use nltk to tokenize sentence
            tokenizer = WordPunctTokenizer()
            new_data = []
            for sentence in data:
                new_data.append(tokenizer.tokenize(sentence))
            character_ids = batch_to_ids(new_data)
            return character_ids
        else:
            raise RuntimeError("the input sequence break!")

    def tokenization(self, data):
        ''' use nltk to tokenize a dict list '''
        if data is None:
            return None
        tokenized_data = []
        tokenizer = WordPunctTokenizer()
        for item in data:
            assert type(item['seq1']) == str
            item["seq1"] = tokenizer.tokenize(item['seq1'])
            item["seq2"] = tokenizer.tokenize(item['seq2']) if item['seq2'] is not None else None
            tokenized_data.append(item)
        return tokenized_data


class BertBaseEncoder(NeuralEncoder):
    def __init__(self, embedding_dim=1024, num_units=512, num_layers=2, dropout_rate=0.5, learning_rate=0.001,
                 task_name=[], hidden=False, full_model_name="bert-base-cased", weight_decay=2e-5, warm_up=0):
        super(BertBaseEncoder, self).__init__(task_name, hidden)
        if hidden == True:
            raise NotImplementedError("use the last output representation of BERT ")
        self._drop_rate = dropout_rate
        self._learning_rate = learning_rate
        self._full_name = full_model_name
        self._model_path = os.path.join(TASK_INFO["PATH_TO_BERT"], full_model_name)
        self._tokenizer = None  # bert tokenizer
        self._config = None  # bert config
        self._hidden_size = 768 if "base" in self._full_name else 1024
        self._weight_decay = weight_decay
        self._adam_eps = 1e-8
        self._warm_up = warm_up

    @overrides
    def _build(self):
        # load pre-trained bert model and it's tokenizer
        lower_case = "uncased" in self._full_name
        self._config = AutoConfig.from_pretrained(self._model_path)
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_path, do_lower_case=lower_case)
        self._encoder = AutoModel.from_pretrained(self._model_path, config=self._config)

        # project (fully connected) layer
        project_layer = []
        for i, name in enumerate(self._task_name):
            # in_dim = self._hidden_size if name not in TASK_INFO["DUAL"] else self._hidden_size * 4  # cat[u,v,u*v,|u-v|]
            in_dim = self._hidden_size
            out_dim = max(1, len(self._idx2labels[i]))  # label num
            dense = Linear(in_dim, out_dim)
            project_layer.append(dense)
        self._project_layer = ModuleList(project_layer)

        # create optimizers ans criterion
        no_decay = ["bias", "LayerNorm.weight"]
        for idx, dense_layer in enumerate(self._project_layer):
            if self._task_name[idx] in TASK_INFO["REGRESS"]:
                self._criterion.append(MSELoss())
            else:
                self._criterion.append(CrossEntropyLoss())
            params = []
            params += [
                {"params": [p for n, p in self._encoder.named_parameters() if not any([nd in n for nd in no_decay])],
                 "weight_decay": self._weight_decay, "lr": self._learning_rate},
                {"params": [p for n, p in self._encoder.named_parameters() if any([nd in n for nd in no_decay])],
                 "weight_decay": 0, "lr": self._learning_rate}
            ]
            params += [{"params": list(dense_layer.parameters())[0], "lr": self._learning_rate,
                        "weight_decay": self._weight_decay}]
            params += [{"params": list(dense_layer.parameters())[1], "lr": self._learning_rate}]
            # each task has it's own optimizer
            optimizer = AdamW(params, eps=self._adam_eps)
            self._optimizer.append(optimizer)

        # move encoder/dense/criterion to cuda
        self._encoder.cuda()
        self._project_layer.cuda()
        for i in range(len(self._task_name)):
            self._criterion[i] = self._criterion[i].cuda()

    @overrides
    def _train(self, batch_data, task_idx: int) -> float:
        ''' use given batch_data to train lstm

        :return: loss -> float
        '''
        # ---------------------------------------------------
        # one batch training
        # self._project_layer[task_idx].cuda()
        regression = self._task_name[task_idx] in TASK_INFO["REGRESS"]
        self._trainable(task_idx)
        assert len(batch_data) == 5
        # get batch_data
        index, input_ids, attention_mask, token_type_ids, labels = tuple(batch_data)

        labels = labels.cuda()
        input_ids = input_ids.cuda()
        attention_mask = attention_mask.cuda()
        token_type_ids = token_type_ids.cuda()
        feed_bert = {"input_ids": input_ids, "attention_mask": attention_mask, "token_type_ids": token_type_ids}
        last_hidden_state = self._encoder(**feed_bert)[0]  # [batch_size, sequence_length, hidden_size]
        pooled = torch.max(last_hidden_state, dim=1)[0]  # [batch_size,hidden_size]
        out = self._project_layer[task_idx](pooled)  # [batch_size,num_class]

        assert len(out.shape) == 2, "unknown error"
        if regression:
            labels = labels.unsqueeze(-1).float()
        else:
            labels = labels.long()

        loss = self._criterion[task_idx](out, labels)

        self._optimizer[task_idx].zero_grad()
        loss.backward()
        self._optimizer[task_idx].step()

        # self._project_layer[task_idx].cpu()  # save GPU memory, we only move project layer to GPU when necessary

        return loss.item()

    @overrides
    def _eval(self, batch_data, task_idx: int):
        '''  pass given batch_data to model, get eval result

        :return: pred_flatten -> list, model predictions
                 label_flatten -> list, target label
                 prediction -> list, model predictions which had been convert to original style
                 index -> list, instance serial number
        '''
        # ---------------------------------------------------
        # one batch evaluation
        self._testable(task_idx)
        with torch.no_grad():
            # self._project_layer[task_idx].cuda()
            assert len(batch_data) == 5
            # get batch_data
            index, input_ids, attention_mask, token_type_ids, labels = tuple(batch_data)
            regression = self._task_name[task_idx] in TASK_INFO["REGRESS"]

            size = index.shape[0]
            index = index.squeeze().tolist()
            label_flatten = labels.squeeze().tolist()
            input_ids = input_ids.cuda()
            attention_mask = attention_mask.cuda()
            token_type_ids = token_type_ids.cuda()
            feed_bert = {"input_ids": input_ids, "attention_mask": attention_mask, "token_type_ids": token_type_ids}
            last_hidden_state = self._encoder(**feed_bert)[0]  # [batch_size, sequence_length, hidden_size]
            pooled = torch.max(last_hidden_state, dim=1)[0]  # [batch_size,hidden_size]
            out = self._project_layer[task_idx](pooled)  # [batch_size,num_class]

            assert len(out.shape) == 2, "unknown error"
            if regression:
                pred = out.detach().cpu().squeeze().tolist()
            else:
                pred = F.softmax(out, dim=1)  # note that torch cross entropy has already included log_softmax
                pred = torch.argmax(pred.detach().cpu(), dim=1, keepdim=False)
                pred = pred.squeeze().tolist()

            if size == 1:
                index = [index]
                pred = [pred]
                label_flatten = [label_flatten]
            # ---------------------------------------------------
            # transfer label
            prediction = self._transfer(pred, task_idx)

            # self._project_layer[task_idx].cpu()  # save GPU memory, we only move project layer to GPU when necessary

        return pred, label_flatten, prediction, index

    @overrides
    def _save(self, save_path):
        encoder_path = os.path.join(save_path, "encoder")
        dense_path = os.path.join(save_path, "project_layers")
        if not os.path.exists(encoder_path):
            os.mkdir(encoder_path)
        if not os.path.exists(dense_path):
            os.mkdir(dense_path)
        # save BERT weight,config and tokenizer
        model_to_save = (self._encoder.module if hasattr(self._encoder, "module") else self._encoder)
        model_to_save.save_pretrained(encoder_path)
        self._tokenizer.save_pretrained(encoder_path)
        # save project_layers
        torch.save({'state_dict': self._project_layer.state_dict()}, os.path.join(dense_path, "dense.pth.tar"))

    @overrides
    def _load(self, save_path) -> bool:
        encoder_path = os.path.join(save_path, "encoder")
        dense_path = os.path.join(save_path, "project_layers")
        if not os.path.exists(encoder_path):
            return False
        if not os.path.exists(dense_path):
            return False
        try:
            # load BERT weight,config and tokenizer
            self._encoder = AutoModel.from_pretrained(encoder_path)
            self._tokenizer = AutoTokenizer.from_pretrained(encoder_path)
            self._encoder.cuda()
            # load project_layers
            dense_checkpoint = torch.load(os.path.join(dense_path, "dense.pth.tar"))
            self._project_layer.load_state_dict(dense_checkpoint['state_dict'])
        except FileNotFoundError:
            print("model checkpoints file missing: %s" % save_path)
            return False
        return True


if __name__ == '__main__':
    # pass
    options_file = "../cache/elmo/option.json"
    weight_file = "../cache/elmo/elmo_original_weights.hdf5"

    elmo = Elmo(options_file, weight_file, 1, dropout=0)

    # use batch_to_ids to convert sentences to character ids
    sentence_lists = ["I have a dog", "How are you , today is Monday", "I am fine thanks"]
    character_ids = batch_to_ids(sentence_lists)
    # [batch_size, max_length, 50]

    result = elmo(character_ids)
    representation = result["elmo_representations"][0]
    # [batch_size, max_length, 1024]
    # i.e. [3,29,1024]
    mask = result["mask"]
