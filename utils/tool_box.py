#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: RenzeLou

import torch
import torch.nn as nn
import torch.nn.functional as F
import tensorflow as tf
import os
import re
import numpy as np
import pynvml
import random
import csv
import logging as log


def seed_torch(seed=42):
    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False


def seed_tensorflow(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)


def search_free_cuda():
    pynvml.nvmlInit()
    id = 2
    for i in range(4):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        if mem_info.used == 0:
            id = i
            break
    os.environ["CUDA_VISIBLE_DEVICES"] = str(id)


def cudafy(module):
    if torch.cuda.is_available():
        return module.cuda(), True
    else:
        return module.cpu(), False


def cuda_avaliable():
    if torch.cuda.is_available():
        return True, torch.device("cuda")
    else:
        return False, torch.device("cpu")


def show_parameters(model: nn.Module, if_show_parameters=False):
    ''' show all named parameters
    i.e. "name:xxx ; size: 1000 "
    :param model: target model
    :param if_show_parameters: whether print param (tensor)
    :return: NULL
    '''
    for name, parameters in model.named_parameters():
        if parameters.requires_grad == False:
            continue
        print("name:{} ; size:{} ".format(name, parameters.shape))
        if if_show_parameters:
            print("parameters:", parameters)


def count_parameters(model: nn.Module):
    ''' count all parameters '''
    return sum([p.numel() for p in model.parameters() if p.requires_grad])


def shuffle2list(a: list, b: list):
    # shuffle two list with same rule, you can also use sklearn.utils.shuffle package
    c = list(zip(a, b))
    random.shuffle(c)
    a[:], b[:] = zip(*c)
    return a, b


def gather(param, ids):
    # Take the line corresponding to IDS subscript from param and form a new tensor
    if param.is_cuda:
        mask = F.one_hot(ids, num_classes=param.shape[0]).float().cuda()
    else:
        mask = F.one_hot(ids, num_classes=param.shape[0]).float()
    ans = torch.mm(mask, param)
    return ans


def pairwise_distance(embeddings, squared=False):
    pairwise_distances_squared = torch.sum(embeddings ** 2, dim=1, keepdim=True) + \
                                 torch.sum(embeddings.t() ** 2, dim=0, keepdim=True) - \
                                 2.0 * torch.matmul(embeddings, embeddings.t())

    error_mask = pairwise_distances_squared <= 0.0
    if squared:
        pairwise_distances = pairwise_distances_squared.clamp(min=0)
    else:
        pairwise_distances = pairwise_distances_squared.clamp(min=1e-16).sqrt()

    pairwise_distances = torch.mul(pairwise_distances, ~error_mask)

    num_data = embeddings.shape[0]
    # Explicitly set diagonals to zero.
    if pairwise_distances.is_cuda:
        mask_offdiagonals = torch.ones_like(pairwise_distances) - torch.diag(cudafy(torch.ones([num_data]))[0])
    else:
        mask_offdiagonals = torch.ones_like(pairwise_distances) - torch.diag(torch.ones([num_data]))

    pairwise_distances = torch.mul(pairwise_distances, mask_offdiagonals)

    return pairwise_distances


def generate_logger(log_dir=".", log_name="LOG.log", level=1):
    ''' generate a python logger, write training information into terminal and a file

    :param: log_dir -> path to log file
    :param: log_name -> name of log file
    :param: level -> showing level, range from 1 to 5 with the importance rising

    :return: a logger class which can both write info to terminal and log_file
    '''
    log_levels = {1: log.DEBUG,
                  2: log.INFO,
                  3: log.WARNING,
                  4: log.ERROR,
                  5: log.CRITICAL}
    log_level = log_levels[level]

    log.basicConfig(format='%(asctime)s: %(message)s', level=log.INFO, datefmt='%m/%d %I:%M:%S %p')
    log_file = os.path.join(log_dir, log_name)
    file_handler = log.FileHandler(log_file)
    file_handler.setLevel(log_level)
    log.getLogger().addHandler(file_handler)

    return log


def FindAllSuffix(path: str, suffix: str, verbose: bool = False) -> list:
    ''' find all files have specific suffix under the path

    :param path: target path
    :param suffix: file suffix. e.g. ".json"/"json"
    :param verbose: whether print the found path
    :return: a list contain all corresponding file path (relative path)
    '''
    result = []
    if not suffix.startswith("."):
        suffix = "." + suffix
    for root, dirs, files in os.walk(path, topdown=False):
        # print(root, dirs, files)
        for file in files:
            if suffix in file:
                file_path = os.path.join(root, file)
                result.append(file_path)
                if verbose:
                    print(file_path)

    return result


def clean_tokenize(data, lower=False):
    ''' used to clean token, split all token with space and lower all tokens
    this function usually use in some language models which don't require strict pre-tokenization
    such as LSTM(with glove vector) or ELMO(already has tokenizer)
    :param data: string
    :return: list, contain all cleaned tokens from original input
    '''
    # split all tokens with a space
    data = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", data)
    data = re.sub(r"\'s", " \'s", data)
    data = re.sub(r"\'ve", " \'ve", data)
    data = re.sub(r"n\'t", " n\'t", data)
    data = re.sub(r"\'re", " \'re", data)
    data = re.sub(r"\'d", " \'d", data)
    data = re.sub(r"\'ll", " \'ll", data)
    data = re.sub(r",", " , ", data)
    data = re.sub(r"!", " ! ", data)
    data = re.sub(r"\(", " ( ", data)
    data = re.sub(r"\)", " ) ", data)
    data = re.sub(r"\?", " ? ", data)
    data = re.sub(r"\s{2,}", " ", data)
    data = data.lower() if lower else data

    # split all tokens, form a list
    return [x.strip() for x in re.split('(\W+)?', data) if x.strip()]


# the flowing two tsv method are break maybe
# ================================================================================
def write_to_tsv(output_path: str, file_columns: list, data: list):
    csv.register_dialect('tsv_dialect', delimiter='\t', quoting=csv.QUOTE_ALL)
    with open(output_path, "w", newline="") as wf:
        writer = csv.DictWriter(wf, fieldnames=file_columns, dialect='tsv_dialect')
        writer.writerows(data)
    csv.unregister_dialect('tsv_dialect')


def read_from_tsv(file_path: str, column_names: list) -> list:
    csv.register_dialect('tsv_dialect', delimiter='\t', quoting=csv.QUOTE_ALL)
    with open(file_path, "r") as wf:
        reader = csv.DictReader(wf, fieldnames=column_names, dialect='tsv_dialect')
        datas = []
        for row in reader:
            data = dict(row)
            datas.append(data)
    csv.unregister_dialect('tsv_dialect')
    return datas
# =======================================================================================


if __name__ == "__main__":
    # p = torch.rand(3, 3)
    # ids = torch.from_numpy(np.arange(3))
    # ans = gather(p, ids)
    # print("p:", p)
    # print("ans", ans)
    # centroid_ids = torch.tensor([0, 2, 3])
    # pd = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 0], [2, 3, 7, 6, 4], [2, 9, 7, 0, 4], [1, 8, 2, 3, 0]])
    # for i in range(5):
    #     pd[i][i] = 0
    # ans = gather(pd, centroid_ids)
    # print("ans:", ans)
    # w = torch.where(ans == 0)
    # print(w)

    # logger = generate_logger()
    # logger.info("hello")

    # ori = ["i love you,but it seems. like you love me too!ok? ",".yes,be quickly!"]
    # print(clean_tokenize("i love you,but it seems. like you love me too!ok? "))


    write_to_tsv("./test.csv",["index","pred"],[{"index": 1, "pred": 2},{"index": 1, "pred": 2}])