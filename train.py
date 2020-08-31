import os
import torch
import argparse
import numpy as np
from torch.utils.data import DataLoader
from datetime import datetime
from utils.path_util import from_project_root, exists
from utils.torch_util import get_device
from dataset import NERDataset, gen_vocab_from_data
from models.ner import CallNoteNER
from models.config import Config
from eval import evaluate
from utils.torch_util import set_random_seed
RANDOM_SEED = 233
set_random_seed(RANDOM_SEED)


def train_end2end(config):
    dataset_name = config.dataset_name
    device = config.device
    save_only_best = config.save_only_best
    n_epochs = config.n_epochs
    bert_model_name = config.bert_model_name
    early_stop = config.early_stop
    learning_rate = config.learning_rate
    batch_size = config.batch_size
    clip_norm = config.clip_norm
    n_tags = config.n_tags
    log_per_batch = config.log_per_batch
    train_url = config.train_url
    dev_url = config.dev_url
    test_url = config.test_url
    c2v = config.c2v
    w2v = config.w2v
    label_list = config.label_list
    #
    if c2v and not exists(c2v):
        gen_vocab_from_data([train_url, dev_url, test_url])
    start_time = datetime.now()
    device = get_device(device)
    train_set = NERDataset(train_url, device=device, bert_model=bert_model_name, label_list=label_list)
    train_loader = DataLoader(train_set, batch_size=batch_size, drop_last=False,
                              collate_fn=train_set.collate_func)
    # N_TAGS labels and the hidden_size on top of embedding is 200
    model = CallNoteNER(n_tags, 200, device, bert_model_name)

    if device.type == 'cuda':
        print("using gpu,", torch.cuda.device_count(), "gpu(s) available!\n")
    else:
        print("using cpu\n")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    cnt = 0
    max_f1, max_f1_epoch = 0, 0
    best_model_url = None
    for epoch in range(n_epochs):
        # switch to train mode
        model.train()
        batch_id = 0
        for data, sentence_labels in train_loader:
            optimizer.zero_grad()
            pred_sentence_labels = model.forward(*data)
            # change pred_sentence_labels to (batch_size, seq_length, num_tags)
            sentence_len = data[1]
            crf_mask = np.array([[True for i in range(sentence_len[0])] for j in range(len(sentence_len))])
            for sent_index in range(len(sentence_len)):
                if sentence_len[sent_index] < sentence_len[0]:
                    crf_mask[sent_index, sentence_len[sent_index]:] = False
            crf_mask = torch.from_numpy(crf_mask)
            loss = - model.crf(pred_sentence_labels.permute(0, 2, 1), sentence_labels,
                               mask=crf_mask.to(device), reduction='mean')
            loss.backward()

            # gradient clipping
            if clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
            optimizer.step()
            if batch_id % log_per_batch == 0:
                print("epoch #%d, batch #%d, loss: %.12f, %s" %
                      (epoch, batch_id, loss.item(), datetime.now().strftime("%X")))
            batch_id += 1

        cnt += 1
        # evaluating model use development dataset or and additional test dataset
        f1 = evaluate(model, dev_url, label_list, bert_model_name, dataset_name, False)
        if f1 > max_f1:
            max_f1, max_f1_epoch = f1, epoch
            if save_only_best and best_model_url:
                os.remove(best_model_url)
            best_model_url = from_project_root("data/model/%s_lr%f_stop%d_epoch%d_%f.pt" % (dataset_name, learning_rate, early_stop, epoch, f1))
            torch.save(model, best_model_url)
            cnt = 0

        print("maximum of f1 value: %.6f, in epoch #%d" % (max_f1, max_f1_epoch))
        print("training time:", str(datetime.now() - start_time).split('.')[0])
        print(datetime.now().strftime("%c\n"))

        if cnt >= early_stop > 0:
            break

    if test_url:
        best_model = torch.load(best_model_url)
        print("best model url:", best_model_url)
        print("evaluating on test dataset:", test_url)
        evaluate(best_model, test_url, label_list, bert_model_name, dataset_name, True)


def main():

    param = argparse.ArgumentParser()
    param.add_argument("--dataset_name", type=str, help="dataset name")
    param.add_argument("--device", type=str, help="device")
    param.add_argument("--save_only_best", type=bool, help="only save the best model or not")
    param.add_argument("--n_epochs", type=int, help="number of epochs")
    param.add_argument("--bert_model_name", type=str, help="bert model name")
    param.add_argument("--early_stop", type=int, help="early stop")
    param.add_argument("--learning_rate", type=float, help="learning rate")
    param.add_argument("--batch_size", type=int, help="batch size")
    param.add_argument("--clip_norm", type=int, help="clip norm")
    param.add_argument("--n_tags", type=int, help="number of tags")
    param.add_argument("--log_per_batch", type=int, help="log per batch")
    param.add_argument("--train_url", type=str, help="train url")
    param.add_argument("--dev_url", type=str, help="dev url")
    param.add_argument("--test_url", type=str, help="test url")
    param.add_argument("--c2v", type=str, help="path of c2v file")
    param.add_argument("--w2v", type=str, help="path of w2v file")
    param.add_argument("--label_list", help="label list")
    args = param.parse_args()

    config = Config(args)
    train_end2end(config)


if __name__ == '__main__':
    main()
