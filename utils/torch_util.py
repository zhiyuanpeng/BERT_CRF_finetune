# coding: utf-8

import numpy as np
import torch
import random


def set_random_seed(seed):
    """ set random seed for numpy and torch, more information here:
        https://pytorch.org/docs/stable/notes/randomness.html
    Args:
        seed: the random seed to set
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(name='auto'):
    """ choose device

    Returns:
        the device specified by name, if name is None, proper device will be returned

    """
    if name == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(name)


def calc_f1(tp, fp, fn, print_result=True):
    """ calculating f1

    Args:
        tp: true positive
        fp: false positive
        fn: false negative
        print_result: whether to print result

    Returns:
        precision, recall, f1

    """
    precision = 0 if tp + fp == 0 else tp / (tp + fp)
    recall = 0 if tp + fn == 0 else tp / (tp + fn)
    f1 = 0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    if print_result:
        print(" precision = %f, recall = %f, micro_f1 = %f\n" % (precision, recall, f1))
    return precision, recall, f1


def f1_score(y_true, y_pred, max_id, id_filter=0):
    total_num = 0
    t_precision = 0
    t_recall = 0
    t_f1 = 0
    for i in range(id_filter, max_id):
        precision, recall, f1, support_num = binary_f1_score(y_true == i, y_pred == i, i)
        t_precision += precision
        t_recall += recall
        t_f1 += f1
        total_num += support_num
    return t_precision/total_num, t_recall/total_num, t_f1/total_num


def binary_f1_score(y_true, y_pred, i):
    LABEL_LIST = ["O", "B-ORG", "I-ORG", "B-PER", "I-PER", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]
    num_proposed = y_pred.sum()
    num_correct = np.logical_and(y_true, y_pred).sum()
    num_gold = y_true.sum()
    precision = 0 if num_proposed == 0 else num_correct / num_proposed
    recall = 0 if num_gold == 0 else num_correct / num_gold
    f1 = 0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
    print("support num for label %s is %d" % (LABEL_LIST[i], num_gold))
    return precision*num_gold, recall*num_gold, f1*num_gold, num_gold


def main():
    y_true = np.array([0,0,0,1,1,1,2,2,2,3,3,3])
    y_pred = np.array([0,0,1,0,1,2,3,2,2,1,3,3])
    pre, re, f1 = f1_score(y_true, y_pred)
    pass


if __name__ == '__main__':
    main()
