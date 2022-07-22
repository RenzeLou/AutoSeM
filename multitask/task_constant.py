''' contain all task specific information and param '''

TASK_INFO = {
    'ALL_TASK': ["CoLA", "MNLI", "MNLIMatched", "MNLIMisMatched", "MRPC", "QNLI", "QQP", "RTE", "SST-2", "STS-B",
                 "WNLI"],
    'DATA_FOLDER': {"CoLA", "MNLI", "MNLIMatched", "MNLIMisMatched", "MRPC", "QNLI", "QQP", "RTE", "SST-2", "STS-B",
                    "WNLI"},
    'PATH_TO_DATA': "./data/GLUE",
    'PATH_TO_ELMO': "./cache/elmo",
    'PATH_TO_BERT': "./cache",
    'BERT_MAX_LEN': 512,
    'PATH_TO_MODEL_PARAM': "./checkpoints",
    'REGRESS': ['STS-B'],
    "DUAL": ["MNLI", "MNLIMatched", "MNLIMisMatched", "MRPC", "QNLI", "QQP", "RTE", "STS-B", "WNLI"],
    "CACHED_NAME": ["CoLA", "STS-B"]}

# used data pre-process
# DATA_CSV_FIELD = [TaskSpecInfo(fold_name)for fold_name in DATA_FOLDER]
# DATA_CSV_FIELD[0].csv_field += []

#
# class TaskSpecInfo(object):
#     def __init__(self, task_name, csv_field):
#         self.task_name = task_name
#         self.csv_field = []
