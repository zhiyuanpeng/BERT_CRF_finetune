# general embedding layer consisting of bert and char-lstm
import torch
import torch.nn as nn
import transformers as ppb
from utils.torch_util import get_device


class BertCharEmbedding(nn.Module):
    """
    embedding layer consists of bert + char-lstm
    """

    def __init__(self, labels_num, lstm_hidden_dim, char_dim, lstm_layers, bert_model, device):
        super().__init__()
        # bert output 768 for each word
        self.bert_dim = 768
        # output dim of self.char_repr
        self.char_dim = char_dim
        # the input dim of self.lstm
        self.word_repr_dim = self.bert_dim + self.char_dim

        self.char_repr = CharLSTM(
            n_chars=1000,
            embedding_size=self.char_dim // 2,
            hidden_size=self.char_dim // 2
        )

        self.dropout = nn.Dropout(p=0.5)
        # this LSTM takes the word rep(embedding+char_feat_dim) as input
        # the output_dim of self.lstm is (1 + bidirectional) * lstm_hidden_dim = lstm_hidden_dim*2
        self.lstm = nn.LSTM(
            input_size=self.word_repr_dim,
            hidden_size=lstm_hidden_dim,
            bidirectional=True,
            num_layers=lstm_layers,
            batch_first=True
        )
        # bert embedding layer
        self.bert = ppb.BertModel.from_pretrained(bert_model)
        self.device = device
        # convert to the dim=# of labels
        self.ht_labeler = nn.Sequential(
            nn.ReLU(),
            # for Linear layer, initialization only need to denote input_dim, output_dim
            # the input data (N, *, input_dim) * is any other dimension we need, output (N, *, output_dim)
            # input_dim = lstm_hidden_dim*2 is the output_dim of self.lstm
            nn.Linear(lstm_hidden_dim*2, labels_num)
        )

    def forward(self, sentences, sentence_lengths, sentence_words, sentence_word_lengths, sentence_word_indices, masks):
        # sentences (batch_size, max_sent_len)
        # sentence_length (batch_size)
        # with torch.no_grad():
        word_repr, _ = self.bert(sentences, attention_mask=masks)
        # word_feat shape: (batch_size, max_sent_len, self.bert_dim=768)
        # add character level feature
        # sentence_words (batch_size, *sent_len, max_word_len)
        # sentence_word_lengths (batch_size, *sent_len)
        # sentence_word_indices (batch_size, *sent_len, max_word_len)
        # char level feature
        char_feat = self.char_repr(sentence_words, sentence_word_lengths, sentence_word_indices)
        # char_feat shape: (batch_size, max_sent_len, char_feat_dim)

        # concatenate char level representation and word level one
        word_repr = torch.cat([word_repr, char_feat], dim=-1)
        # word_repr shape: (batch_size, max_sent_len, word_repr_dim)

        # drop out
        word_repr = self.dropout(word_repr)

        packed = nn.utils.rnn.pack_padded_sequence(word_repr, sentence_lengths, batch_first=True)
        out, (hn, _) = self.lstm(packed)
        # out packed_sequence(batch_size, max_sent_len, num_directions*hidden_size)
        # hn (n_layers * n_directions, batch_size, hidden_size)

        max_sent_len = sentences.shape[1]
        unpacked, _ = nn.utils.rnn.pad_packed_sequence(out, total_length=max_sent_len, batch_first=True)
        # unpacked (batch_size, max_sent_len, num_directions*hidden_size)

        sent_first = unpacked.transpose(0, 1)
        # sent_first (max_sent_len, batch_size, n_hidden)
        # token (batch_size, n_hidden)
        sentence_outputs = torch.stack([self.ht_labeler(token) for token in sent_first], dim=-1)
        # shape of each ht_labeler output: (batch_size, n_classes)
        # shape of sentence_outputs: (batch_size, n_classes, lengths[0])
        # lengths[0] is the max length in the batch
        # add a CRF layer on top of sentence_outputs
        return sentence_outputs


class CharLSTM(nn.Module):
    # n_chars=1000,
    # embedding_size=char_feat_dim // 2 = 25
    # hidden_size=char_feat_dim // 2 = 25
    def __init__(self, n_chars, embedding_size, hidden_size, lstm_layers=1, bidirectional=True):
        super().__init__()
        self.n_chars = n_chars
        self.embedding_size = embedding_size
        self.n_hidden = hidden_size * (1 + bidirectional)
        self.embedding = nn.Embedding(n_chars, embedding_size, padding_idx=0)

        self.lstm = nn.LSTM(
            input_size=embedding_size,
            hidden_size=hidden_size,
            bidirectional=bidirectional,
            num_layers=lstm_layers,
            batch_first=True,
        )

    def sent_forward(self, words, lengths, indices):
        """
        :param words: [[c1,c2,...](word1), [c1, c2,...](word2), ...]
        :param lengths: [len(word1), len(word2),...]
        :param indices: the original loc of each word in each sentence
        :return:
        """
        sent_len = words.shape[0]
        # words shape: (sent_len, max_word_len)
        embedded = self.embedding(words)
        # in_data shape: (sent_len, max_word_len, embedding_dim)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths, batch_first=True)
        _, (hn, _) = self.lstm(packed)
        # shape of hn:  (n_layers * n_directions, sent_len, hidden_size) num_directions should be 2, else it should be 1
        hn = hn.permute(1, 0, 2).contiguous().view(sent_len, -1)
        # shape of hn:  (sent_len, n_layers * n_directions * hidden_size) = (sent_len, 2*hidden_size)
        # hn.permute(1, 0, 2) = (sent_len, n_layers*n_directions, hidden_size)
        # view(sent_len, -1), -1 means that the dimension is inferred
        # for each word, the vector is n_layers*n_directions*hidden_size(1*2*25=50)

        # shape of indices: (sent_len, max_word_len)
        hn[indices] = hn  # unsort hn
        # unsorted = hn.new_empty(hn.size())
        # unsorted.scatter_(dim=0, index=indices.unsqueeze(-1).expand_as(hn), src=hn)
        return hn

    def forward(self, sentence_words, sentence_word_lengths, sentence_word_indices):
        """
        :param sentence_words:
        [sentence1, sentence2,...]
        sentence1 = [[c1,c2,...](word1), [c1, c2,...](word2), ...]
        each word is padded to have same length as the max len(word) in the current sent, so for
        different sentence the padding length is different. All the words are sorted by length.
        The original loc of word is stored in sentence_word_indices
        :param sentence_word_lengths:
        [sentence1, sentence2, ...]
        sentence1 = [len(word1), len(word2)]
        :param sentence_word_indices:
        the original loc of each word in each sentence
        :return: a word vector learned by the chars inside the word, the first word vector represents the real first
        word in the sentence, we use the sentence_word_indices to recover the order of the words in the sentence
        """
        batch_size = len(sentence_words)
        # sentence_words[i] is: sentencei = [[c1,c2,...](word1), [c1, c2,...](word2), ...]
        # sentence_word_lengths[i] is: [len(word1), len(word2),...]
        # sentence_word_indices[i] is: the original loc of each word in each sentence
        batch_char_feat = torch.nn.utils.rnn.pad_sequence(
            [self.sent_forward(sentence_words[i], sentence_word_lengths[i], sentence_word_indices[i])
             for i in range(batch_size)], batch_first=True)
        return batch_char_feat
        # (batch_size, sent_len, 2 * hidden_size)


def main():
    pass


if __name__ == '__main__':
    main()

