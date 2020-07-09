# coding: utf-8
import torch
import numpy as np
from torch.utils.data import DataLoader
from dataset import NERDataset
from utils.path_util import from_project_root
from utils.torch_util import f1_score
from sklearn import metrics


def evaluate(model, data_url, label_list):
    """
    evaluating end2end model on dataurl
    Args:
        model: trained end2end model
        data_url: url to test dataset for evaluating
        label_list: the label list of the dataset
    Returns:
        ret: dict of precision, recall, and f1
    """
    print("\nevaluating model on:", data_url, "\n")
    dataset = NERDataset(data_url, device=next(model.parameters()).device, bert_model="bert-base-uncased", label_list=label_list)
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
            sentence_lengths = data[1]
            for length, true_labels, pred_labels in zip(sentence_lengths, sentence_labels, pred_sentence_labels):
                for i in range(length):
                    true_labels_numpy = true_labels.cpu().numpy()
                    sentence_true_list.append(true_labels_numpy[i])
                    sentence_pred_list.append(pred_labels[i])
        precision, recall, f1 = f1_score(np.array(sentence_true_list), np.array(sentence_pred_list), max_id=len(label_list), id_filter=0)
        # f1 = metrics.f1_score(sentence_true_list, sentence_pred_list, average="weighted")
        print("Precision is %.4f, Recall is %.4f, F1 is %.4f" % (precision, recall, f1))
        # print("Precision is %.4f, Recall is %.4f, F1 is %.4f" % (0, 0, f1))
    return f1


def main():
    label_list = ["O", "B-ORG", "I-ORG", "B-PER", "I-PER", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]
    model_url = from_project_root("data/model/lr0.001000_stop10_epoch67_0.944072.pt")
    test_url = from_project_root("data/CoNLL2003/conll2003_test.bio")
    model = torch.load(model_url)
    evaluate(model, test_url, label_list)
    pass


if __name__ == '__main__':
    main()
