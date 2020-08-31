# coding: utf-8
import torch
import numpy as np
from torch.utils.data import DataLoader
from dataset import NERDataset
from utils.path_util import from_project_root
from utils.torch_util import f1_score
import transformers as ppb
# from sklearn import metrics


def evaluate(model, data_url, label_list, bert_model_name, dataset_name, log_or_not):
    """
    evaluating end2end model on data_url
    Args:
        model: trained end2end model
        data_url: url to test dataset for evaluating
        label_list: the label list of the dataset
        bert_model_name: name or path of bert model
        dataset_name: name of the dataset
        log_or_not: true, log the result
    Returns:
        ret: dict of precision, recall, and f1
    """
    print("\nevaluating model on:", data_url, "\n")
    dataset = NERDataset(data_url, device=next(model.parameters()).device, bert_model=bert_model_name,
                         label_list=label_list)
    loader = DataLoader(dataset, batch_size=256, collate_fn=dataset.collate_func)
    # switch to eval mode
    model.eval()
    with torch.no_grad():
        sentence_true_list, sentence_pred_list = list(), list()
        for data, sentence_labels in loader:
            try:
                pred_sentence_labels = model.forward(*data, train=False)
                # pred_sentence_output (batch_size, n_classes, lengths[0])
            except RuntimeError:
                print("all 0 tags, no evaluating this epoch")
                continue
            sentences_str = data[6]
            sentence_lengths = data[1]
            write_predict_result(sentences_str, sentence_lengths, sentence_labels, pred_sentence_labels, label_list,
                                 dataset_name)
            for length, true_labels, pred_labels in zip(sentence_lengths, sentence_labels, pred_sentence_labels):
                for i in range(length):
                    true_labels_numpy = true_labels.cpu().numpy()
                    sentence_true_list.append(true_labels_numpy[i])
                    sentence_pred_list.append(pred_labels[i])
        precision, recall, f1 = f1_score(np.array(sentence_true_list), np.array(sentence_pred_list), label_list,
                                         log_or_not, id_filter=0, max_id=len(label_list))
        print("Precision is %.4f, Recall is %.4f, F1 is %.4f" % (precision, recall, f1))
    return f1


def write_predict_result(sentences_str, sentence_lengths, sentence_labels, pred_sentence_labels, label_list,
                         dataset_name):
    for sentence, length, real, pred in zip(sentences_str, sentence_lengths, sentence_labels.cpu().numpy(),
                                            pred_sentence_labels):
        real_list = list(real)[:length]
        pred_list = pred[:length]
        with open("./data/{}/final_result.txt".format(dataset_name), "a+") as f:
            for i in range(length):
                result = sentence[i] + "\t" + label_list[real_list[i]] + "\t" + label_list[pred_list[i]] + "\n"
                f.write(result)
            f.write("\n")


def main():
    label_list = ["O", "B-ORG", "I-ORG", "B-PER", "I-PER", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]
    model_url = from_project_root("data/model/CoNLL_lr0.000020_stop5_epoch35_0.942166.pt")
    test_url = from_project_root("data/CoNLL2003/conll2003_test.bio")
    bert_model_name = "bert-base-uncased"
    dataset_name = "CoNLL2003"
    model = torch.load(model_url)
    evaluate(model, test_url, label_list, bert_model_name, dataset_name)
    pass


if __name__ == '__main__':
    main()
