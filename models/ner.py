# coding: utf-8
import torch
import torch.nn as nn
import numpy as np
from torchcrf import CRF
from utils.torch_util import get_device
from models.bert_char_embedding import BertCharEmbedding


class CallNoteNER(nn.Module):
    def __init__(self, labels_num, lstm_hidden_dim, device, bert_model, char_dim=50, lstm_layers=1):
        super().__init__()

        self.embedding = BertCharEmbedding(labels_num, lstm_hidden_dim, char_dim, lstm_layers, bert_model, device)
        # define the CRF model
        self.crf = CRF(labels_num, batch_first=True)
        self.softmax = nn.Softmax(dim=1)
        self.device = device

    def forward(self, sentences, sentence_lengths, sentence_words, sentence_words_lengths,
                sentence_words_indices, masks, sentences_list, train=True):
        """
        :param sentences:
                    [sentence1, sentence2, ...]
                    sentence1 = [word1, word2, ...] sentence1 is the longest sent in batch, other sents are padded
        :param sentence_lengths: [len(sentence1), len(sentence2), ...]
        :param sentence_words:
                    [sentence1, sentence2,...]
                    sentence1 = [[c1,c2,...](word1), [c1, c2,...](word2), ...]
                    each word is padded to have same length as the max len(word) in the current sent, so for
                    different sentence the padding length is different
        :param sentence_words_lengths:
                    [sentence1, sentence2, ...]
                    sentence1 = [len(word1), len(word2)]
        :param sentence_words_indices: the original loc of each word in each sentence
        :param masks: mask the padding
        :param train:
        :return:
        """
        sentence_outputs = self.embedding(sentences, sentence_lengths, sentence_words, sentence_words_lengths,
                                          sentence_words_indices, masks)
        # shape of sentence_outputs: (batch_size, n_classes, lengths[0])
        if train:
            return sentence_outputs
        else:
            crf_mask = np.array([[True for i in range(sentence_lengths[0])] for j in range(len(sentence_lengths))])
            for sent_index in range(len(sentence_lengths)):
                if sentence_lengths[sent_index] < sentence_lengths[0]:
                    crf_mask[sent_index, sentence_lengths[sent_index]:] = False
            crf_mask = torch.from_numpy(crf_mask)
            sentence_labels = self.crf.decode(sentence_outputs.permute(0, 2, 1),
                                              mask=crf_mask.to(self.device))
            return sentence_labels


def main():
    pass


if __name__ == '__main__':
    main()

