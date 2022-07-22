import warnings

# ======================================================================================================================
main_task = ["CoLA", "MNLIMatched", "MNLIMisMatched", "MRPC", "QNLI", "QQP", "RTE", "SST-2", "WNLI"]


# when main task is MNLI, you should run both MNLIMatched and MNLIMisMatched as main task to get two eval results


def chose_task(task_idx):
    ''' the first idx is main task

    :param task_idx -> can be int or list
    :return: a task name string split by '_'
    '''
    if type(task_idx) == list:
        # setting 1 & 2
        if 1 in task_idx and 2 in task_idx:
            raise ValueError("MNLIMatched and MNLIMisMatched can't appear in an experiment")
        task = []
        for idx in task_idx:
            task.append(main_task[idx])
        return "_".join(task)
    elif type(task_idx) == int:
        # namely setting 1
        auxilary = list(range(len(main_task)))
        auxilary.remove(task_idx)
        if task_idx not in [1, 2]:
            auxilary.remove(2)
            return chose_task([task_idx] + auxilary)
        else:
            warnings.warn("note you are running MNLI as main task, please remember to run both mm and m")
            auxilary.remove(3 - task_idx)
            return chose_task([task_idx] + auxilary)
    else:
        raise NotImplementedError


# ======================================================================================================================
cuda = "1"
max_steps = 200
train_batch_size = 16
train_batch_num = 10
eval_batch_size = 8
steps_per_eval = 8
encoder_type = 1
learning_rate = 0.001
stage = 1
setting = 0
seed = 42

# tasks = chose_task(2)

random = False
hidden = False

shell_name = "auto_task_selection.sh"
# ======================================================================================================================
# template
# template = "python main.py "
# template += "--cuda {} -- max_steps {} --train_batch_size {} --train_batch_num {} ".format(cuda, max_steps,
#                                                                                            train_batch_size,
#                                                                                            train_batch_num)
# template += "--eval_batch_size {} --steps_per_eval {} ".format(eval_batch_size, steps_per_eval)
# template += "--encoder_type {} --learning_rate {} --stage {} ".format(encoder_type, learning_rate, stage)
# template += "--random_seed {} ".format(seed)
# template += "--tasks {} ".format(tasks)
# template += "--random_sampler " if random else ""
# template += "--use_hidden_state " if hidden else ""
# template += "\n"
# ======================================================================================================================
template = ""
for i in range(len(main_task)):
    tasks = chose_task(i)
    template += "python main.py "
    template += "--tasks {} ".format(tasks)
    template += "--random_seed {} --setting {} ".format(seed, setting)
    template += "--encoder_type {} --learning_rate {} --stage {} ".format(encoder_type, learning_rate, stage)
    template += "--cuda {} --max_steps {} --train_batch_size {} --train_batch_num {} ".format(cuda, max_steps,
                                                                                               train_batch_size,
                                                                                               train_batch_num)
    template += "--eval_batch_size {} --steps_per_eval {} ".format(eval_batch_size, steps_per_eval)
    template += "--random_sampler " if random else ""
    template += "--use_hidden_state " if hidden else ""

    template += "\n"

with open(shell_name, "w") as f:
    f.write(template)
# ======================================================================================================================
