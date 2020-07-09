import os
import torch
import joblib
import numpy as np
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from collections import defaultdict
import transformers as ppb

import utils.json_util as ju
from utils.path_util import from_project_root, dirname


class NERDataset(Dataset):
    def __init__(self, data_url, device, bert_model, label_list):
        super().__init__()
        self.data_url = data_url
        self.bert_model = bert_model
        self.label_list = label_list
        self.sentences, self.labels = load_raw_data(data_url)
        self.device = device

    def __getitem__(self, index):
        return self.sentences[index], self.labels[index]

    def __len__(self):
        return len(self.sentences)

    def collate_func(self, data_list):
        """
        for xx in collate_func(batch_size, sentences, labels)
        :param data_list:
        [([sentence1], [label1]),
        ([sentence2], [label2]),
        ...total batch size #]
        ([sentence], [label]) format
        (['Bayern', 'München', 'ist', 'wieder', 'alleiniger'],
        [B-ORG, I-ORG, O, O,....O])
        :return:
        """
        # sort the data_list according to the len of the sentence
        data_list = sorted(data_list, key=lambda tup: len(tup[0]), reverse=True)
        sentences_list, labels_list = zip(*data_list)  # un zip
        max_len = len(sentences_list[0])
        """
        sentence_tensors is tup(sentences, sentence_lengths, sentence_words, sentence_word_lengths, sentence_word_indices)
        sentences: [sentence1, sentence2, ...]
                   sentence1 = [word1, word2, ...] sentence1 is the longest sent in batch, other sents are padded
        sentence_lengths: [len(sentence1), len(sentence2), ...]
                          sort the words in each sentence by len then:
        sentence_words: [sentence1, sentence2,...]
                        sentence1 = [[c1,c2,...](word1), [c1, c2,...](word2), ...]
                        each word is padded to have same length as the max len(word) in the current sent, so for
                        different sentence the padding length is different
        sentence_words_lengths: [sentence1, sentence2, ...]
                               sentence1 = [len(word1), len(word2)]
        sentence_words_indices: the original loc of each word in each sentence
        """
        sentence_tensors = gen_sentence_tensors(sentences_list, self.device, self.data_url, self.bert_model)
        # (sentences, sentence_lengths, sentence_words, sentence_word_lengths, sentence_word_indices)
        sentence_labels = list()
        for labels in labels_list:
            # labels is a list of label
            digit_label = []
            for i in range(max_len):
                if i < len(labels):
                    digit_label.append(self.label_list.index(labels[i]))
                else:
                    digit_label.append(0)
            sentence_labels.append(digit_label)
        sentence_labels = torch.LongTensor(sentence_labels).to(self.device)
        return sentence_tensors, sentence_labels


def gen_sentence_tensors(sentence_list, device, data_url, bert_model):
    """
    generate input tensors from sentence list
    :param sentence_list: [[w1, w2,...,wn](sent1), [w1, w2,...](sent2), ...] len(sentence_list) = batch_size
    :param device: torch device
    :param data_url: used to find the vocab file
    :return:
    sentences: [sentence1, sentence2, ...]
               sentence1 = [word1, word2, ...] sentence1 is the longest sent in batch, other sents are padded
    sentence_lengths: [len(sentence1), len(sentence2), ...]
    sort the words in each sentence by len then:
    sentence_words: [sentence1, sentence2,...]
                    sentence1 = [[c1,c2,...](word1), [c1, c2,...](word2), ...]
                    each word is padded to have same length as the max len(word) in the current sent, so for
                    different sentence the padding length is different. All the words are sorted by length.
                    The original loc of word is stored in sentence_word_indices
    sentence_words_lengths: [sentence1, sentence2, ...]
                            sentence1 = [len(word1), len(word2)]
    sentence_words_indices: the original loc of each word in each sentence
    """
    char_vocab = ju.load(dirname(data_url) + '/char_vocab.json')

    sentences = list()
    sentence_words = list()
    sentence_words_lengths = list()
    sentence_words_indices = list()
    masks = list()

    unk_idx = 1
    # initialize the tokenizer
    bert_tokenizer = ppb.BertTokenizer.from_pretrained(bert_model)
    for sent in sentence_list:
        # word to word id
        sentence = torch.LongTensor(bert_tokenizer.encode(sent, add_special_tokens=False)).to(device)
        # add mask
        mask = [1 for i in range(list(sentence.shape)[0])]
        mask = torch.LongTensor(mask).to(device)
        # char of word to char id
        # words is a 2D list, stores the char ID list of each word
        words = list()
        for word in sent:
            words.append([char_vocab[ch] if ch in char_vocab else unk_idx
                          for ch in word])
        # save word lengths
        word_lengths = torch.LongTensor([len(word) for word in words]).to(device)
        # sorting lengths according to length
        # word_indices is the original location of word in sentence
        word_lengths, word_indices = torch.sort(word_lengths, descending=True)
        # sorting word according word length
        words = np.array(words)[word_indices.cpu().numpy()]
        word_indices = word_indices.to(device)
        words = [torch.LongTensor(word).to(device) for word in words]
        # padding char tensor of words
        words = pad_sequence(words, batch_first=True).to(device)
        # (max_word_len, sent_len)
        # need to check the dim of words

        sentences.append(sentence)
        sentence_words.append(words)
        sentence_words_lengths.append(word_lengths)
        sentence_words_indices.append(word_indices)
        masks.append(mask)

    # record sentence length and padding sentences
    sentence_lengths = [len(sentence) for sentence in sentences]
    # (batch_size)
    sentences = pad_sequence(sentences, batch_first=True).to(device)
    masks = pad_sequence(masks, batch_first=True).to(device)
    # (batch_size, max_sent_len)
    # return the sentences_list for write the pred result to the file
    return sentences, sentence_lengths, sentence_words, sentence_words_lengths, sentence_words_indices, masks, sentence_list


def load_raw_data(data_url, update=False):
    """
    load data into sentences and labels
    :param data_url: url of data file
    :param update: whether force to update
    :return: sentences, labels
    """
    # load from pickle
    save_url = data_url.replace('.bio', '.raw.pkl').replace('.iob2', '.raw.pkl')
    if not update and os.path.exists(save_url):
        return joblib.load(save_url)

    sentences = list()
    labels = list()
    with open(data_url, 'r', encoding='utf-8') as iob_file:
        first_line = iob_file.readline()
        n_columns = first_line.count('\t') + 1
        columns = [[x] for x in (first_line.strip("\n")).split("\t")]
        for line in iob_file:
            if line != '\n':
                line_values = (line.strip("\n")).split("\t")
                for i in range(n_columns):
                    # columns = [[token1, token2,...], [l1, l2,...]]
                    columns[i].append(line_values[i])
            else:  # end of a sentence
                # sentence = [token1, token2,...]
                sentence = columns[0]
                sentences.append(sentence)
                # if we have more than one label column, then columns[1:] = [[label list 1],[label list 2],...]
                label = columns[1]
                labels.append(label)
                columns = [list() for i in range(n_columns)]
    joblib.dump((sentences, labels), save_url)
    return sentences, labels


class Word2Vector:
    """
    load the pre-trained embedding into a dict
    """
    def __init__(self, pretrained_url):
        self.pretrained_url = pretrained_url
        self.embeddings_dictionary = dict()
        self.load()
        self.vector_size = len(self.embeddings_dictionary["."])

    def load(self):
        with open(self.pretrained_url, encoding="utf8") as f:
            word2vector = f.readlines()
            for line in word2vector:
                records = line.split()
                word = records[0]
                vector_dimensions = np.asarray(records[1:], dtype='float32')
                self.embeddings_dictionary[word] = vector_dimensions


def gen_vocab_from_data(data_urls, pretrained_url=None, update=True, min_count=2):
    """
    generate vocabulary and embeddings from data file, generated vocab files will be saved in data dir
    :param data_urls: a list of data path for preparing vocab
    :param pretrained_url: pre-trained embedding file
    :param update: force to update if true
    :param min_count: if the word occurrence min_count times, then add it to the vocab
    :return: generated word embedding url
    """
    # get the dir name of the data file
    data_dir = os.path.dirname(data_urls[0])
    # creat the vocab.json file
    vocab_url = os.path.join(data_dir, "vocab.json")
    # creat the char_vocab.json file
    char_vocab_url = os.path.join(data_dir, "char_vocab.json")
    embedding_url = os.path.join(data_dir, "embeddings.npy") if pretrained_url else None

    if (not update) and os.path.exists(char_vocab_url):
        print("char vocab file already exists")

    vocab = set()
    char_vocab = set()
    word_counts = defaultdict(int)
    print("generating vocab from", data_urls)
    for data_url in data_urls:
        with open(data_url, 'r', encoding='utf-8') as data_file:
            for row in data_file:
                if row == '\n':
                    continue
                # in each data file:
                # token\tBIO_flag
                token = row.split()[0]
                word_counts[token] += 1
                if word_counts[token] >= min_count:
                    vocab.add(token)
                # union the chars in token
                char_vocab = char_vocab.union(token)

    # sorting vocab according alphabet order
    vocab = sorted(vocab)
    char_vocab = sorted(char_vocab)

    # generate word embeddings for vocab, if word is in pre-trained embedding, then extract the embedding for the word,
    # else, random generate am embedding for that word (here, we can't use the same embedding for all the words that are
    # not in the pre-trained embedding file)
    if pretrained_url is not None:
        print("generating pre-trained embedding from", pretrained_url)
        kvs = Word2Vector(pretrained_url)
        embeddings = list()
        for word in vocab:
            if word in kvs.embeddings_dictionary:
                embeddings.append(kvs.embeddings_dictionary[word])
            else:
                embeddings.append(np.random.uniform(-0.25, 0.25, kvs.vector_size))
        embeddings = np.vstack([np.zeros(kvs.vector_size),  # for <pad>
                                np.random.uniform(-0.25, 0.25, kvs.vector_size),  # for <unk>
                                embeddings])
        np.save(embedding_url, embeddings)
        vocab = ['<pad>', '<unk>'] + vocab
        ju.dump(ju.list_to_dict(vocab), vocab_url)
    char_vocab = ['<pad>', '<unk>'] + char_vocab
    ju.dump(ju.list_to_dict(char_vocab), char_vocab_url)


def main():
    data_urls = [from_project_root("data/CoNLL2003/conll2003_train.bio"),
                 from_project_root("data/CoNLL2003/conll2003_dev.bio"),
                 from_project_root("data/CoNLL2003/conll2003_test.bio")]
    gen_vocab_from_data(data_urls)
    pass


if __name__ == '__main__':
    main()
