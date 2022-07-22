# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
''' use this file to load json and process the data type'''

import logging
import os
import json
import copy
import pickle


class InputExample(object):
    """
    A single training/test example for simple sequence classification.

    Args:
        guid: Unique id for the example.
        text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
        text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
        label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """

    def __init__(self, guid, text_a, text_b=None, label=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class InputFeatures_ELMO(object):
    """
    This class is similar to transformers.data.processors.utils.InputFeatures,
    use this class for elmo embedding and unify with BERT
    """

    def __init__(self, guid, embedding_1, embedding_2=-1, label=-1):
        self.guid = guid
        self.embedding_1 = embedding_1
        # if the property is -1, means None
        self.embedding_2 = embedding_2
        self.label = label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class InputFeatures_BERT(object):
    """
    This class is similar to transformers.data.processors.utils.InputFeatures,
    use this class for elmo embedding and unify with BERT
    """

    def __init__(self, guid, input_ids, attention_mask, token_type_ids=None, label=None):
        self.guid = guid
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label = label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class Processor(object):
    def __init__(self, idx2labels: dict, model_type: int):
        self._idx2labels = idx2labels
        self._model_type = model_type
        # 1:lstm 2:bert

    def _read_json(self, input_file):
        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data

    def get_examples(self, data_path, set_type, task_idx):
        if self._model_type == 1:
            return self._get_elmo_examples(data_path, set_type, task_idx)
        else:
            return self._get_bert_examples(data_path, set_type, task_idx)

    def _get_elmo_examples(self, data_path, set_type, task_idx):
        data_info = self._idx2labels[task_idx]
        lines = self._read_json(os.path.join(data_path, "{}.json".format(set_type)))
        index, seq1, seq2, lb = [], [], [], []
        for (i, line) in enumerate(lines):
            guid = str(line["index"])
            text_a = line["seq1"]
            assert type(text_a) == str, "unknown error"
            text_b = str(line["seq2"])
            label = str(line["label"])
            index.append(guid)
            seq1.append(text_a)
            seq2.append(text_b)
            lb.append(label)
        examples = InputExample(guid=index, text_a=seq1, text_b=seq2, label=lb)

        return examples

    def _get_bert_examples(self, data_path, set_type, task_idx):
        data_info = self._idx2labels[task_idx]
        lines = self._read_json(os.path.join(data_path, "{}.json".format(set_type)))
        examples = []
        for (i, line) in enumerate(lines):
            guid = str(line["index"])
            text_a = line["seq1"]
            assert type(text_a) == str, "unknown error"
            text_b = line["seq2"] if type(line["seq2"]) == str else None
            if "test" in set_type:
                label = None
            else:
                label = str(data_info[str(line["label"])]) if len(self._idx2labels) != 0 else str(
                    line["label"])  # original label
            assert isinstance(text_a, str)
            if label is not  None:
                assert isinstance(label, str)
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))

        return examples

    def get_labels(self, task_idx: int):
        """use for bert, return original label_list"""
        labels = list(self._idx2labels[task_idx].values())
        return labels


if __name__ == '__main__':
    t = []
    t.append(InputFeatures_ELMO(1, [[1, 2, 3], [3, 2, 1]], -1, -1))
    t.append(InputFeatures_ELMO(1, [[1, 2, 3], [3, 2, 1]], [[1, 2, 3], [5, 6, 7]], '-1'))
    with open("./test.pkl", "wb") as f:
        pickle.dump(t, f)
    with open("./test.pkl", "rb") as f:
        tt = pickle.load(f)
    print(t)
    print(tt)
