''' use this script to format .json file to be compatible with the training later '''
import json


def format_json(data: list):
    '''
    index: int -> str
    seq2: None -> -1
    label: None -> -1
    '''
    new_data = []
    for item in data:
        item["index"] = str(item["index"])
        item["seq2"] = -1 if item["seq2"] is None else item["seq2"]
        item["label"] = -1 if item["label"] is None else item["label"]
        new_data.append(item)

    return new_data


with open("train.json", "r", encoding="utf-8") as train, open("dev.json", "r", encoding="utf-8") as dev, open(
        "test.json", "r", encoding="utf-8") as test:
    train_data = json.load(train)
    dev_data = json.load(dev)
    test_data = json.load(test)

    new_train = format_json(train_data)
    new_dev = format_json(dev_data)
    new_test = format_json(test_data)

with open("train_format.json", "w", encoding="utf-8") as train, open(
        "dev_format.json", "w", encoding="utf-8") as dev, open(
    "test_format.json", "w", encoding="utf-8") as test:
    json.dump(new_train, train)
    json.dump(new_dev, dev)
    json.dump(new_test, test)
    print("format end!")
