#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: RenzeLou
# a implementation of AutoSem (multi task select model)
# Multi task learning with multi arms bandits

import warnings
from tqdm import tqdm

import torch
from torch.utils.data import RandomSampler, SequentialSampler, DataLoader
from utils.dataloader_iter import DataLoaderIter


class myDataset(torch.utils.data.Dataset):
    ''' create my own torch Dataset
    implement torch.utils.data.Dataset
    '''

    def __init__(self, dataSource):
        '''
        :param dataSource: iterable object like list
        datasource means a list contain all data sample
        e.g. [obj1,obj2...]
        '''
        self.dataSource = dataSource

    def __getitem__(self, index):
        element = self.dataSource[index]
        return element

    def __len__(self):
        return len(self.dataSource)


class myDataIter(object):
    '''  a torch data_iterator  '''

    def __init__(self, data, batch_size, random=True):
        self.data = data
        self.random = random
        self._data_iter = None
        self._batch_size = batch_size
        self._iteration = 0
        self._reset = False  # one epoch

    def get_iteration(self):
        return self._iteration

    def if_reset(self):
        return self._reset

    def _build(self):
        ''' create a new DataLoaderIter object '''
        # dataset = myDataset(self.data)
        # sampler = RandomSampler(dataset) if self.random else SequentialSampler(dataset)
        # data_loader = DataLoader(dataset=dataset, batch_size=self._batch_size, sampler=sampler)
        data_loader = simple_data_loader(self.data, self._batch_size, self.random)
        self._data_iter = DataLoaderIter(data_loader)

    def next(self):
        ''' get next batch data
        if the data has been taken out then initialize a new data_iterator
        :return: a batch of data
        '''
        if self._data_iter is None:
            warnings.warn("create data_loader_iter firstly")
            self._build()
        try:
            batch = self._data_iter.next()
            self._iteration += 1

            return batch
        except StopIteration:
            self._build()
            self._iteration = 1  # reset and return the 1st batch
            self._reset = True

            batch = self._data_iter.next()

            return batch


def simple_data_loader(data: list, batch_size: int, random: bool):
    ''' create a naive data loader '''
    dataset = myDataset(data)
    sampler = RandomSampler(dataset) if random else SequentialSampler(dataset)
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, sampler=sampler)

    return data_loader


if __name__ == '__main__':
    # template = [{"1": "so elegant", "2": torch.tensor([2,2]).long(), "3": torch.tensor([[2,4],[3,8]]).long()},
    #             {"1": "yyy", "2": torch.randn((2)).long(), "3": torch.rand((2,2)).long()},
    #             {"1": "www", "2": 2, "3": 3},
    #             {"1": "xxx", "2": 2, "3": 3},
    #             {"1": "zzz", "2": 2, "3": 3}]
    template = [{"1": "so elegant", "2": ["asdfa","sadk"], "3": torch.tensor([[2, 4], [3, 8]]).long()},
                {"1": "yyy", "2": ["asdfa","sadk","dasdas"], "3": torch.rand((2, 2)).long()},
                {"1": "www", "2": ["asdfa","sadk","111"], "3": torch.rand((2, 2)).long()},
                {"1": "xxx", "2": ["asdfa","sadk","4523","2222"], "3": torch.rand((2, 2)).long()},
                {"1": "zzz", "2": ["asdfa","sadk"],
                 "3": torch.rand((2, 2)).long()}]  # torch.randn((2)).long() torch.tensor([2, 2]).long()
    # get a dict which has the same key as original template element
    # if the original value is a string, then get a list contain a batch of string
    # else if original value is int, then get a tensor with shape [batch_size,]

    # dataset = myDataset(template)
    # sampler = RandomSampler(dataset)
    # train_loader = DataLoader(dataset=dataset, batch_size=1, sampler=sampler)
    #
    # for t in train_loader:
    #     print("first:", t)
    #     break
    #
    # for t in train_loader:
    #     print("second:", t)
    #     break
    #
    # dataIter = DataLoaderIter(train_loader)
    # print(dataIter.next())
    # print(dataIter.next())
    data_iter = myDataIter(template, 3, random=False)
    data_loader = simple_data_loader(template,3,random=False)
    for batch in data_loader:
        print(batch)
    # first = data_iter.next()
    # print(first["3"].shape)
    # print(first["1"], type(first["1"]))
    # print(data_iter.next()["2"].shape)
    # print(data_iter.next()["2"].shape)
    # print(data_iter.next()["2"].shape)
    # print(data_iter.next()["2"].shape)
    # print(data_iter.next()["2"].shape)

    wait = True
