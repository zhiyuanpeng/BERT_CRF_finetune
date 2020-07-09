import os
import torch
import json
import numpy as np
from torch.utils.data import DataLoader
from datetime import datetime
from utils.path_util import from_project_root, exists
from utils.torch_util import get_device
from dataset import NERDataset, gen_vocab_from_data
from models.ner import CallNoteNER
from eval import evaluate
from utils.torch_util import set_random_seed
RANDOM_SEED = 233
set_random_seed(RANDOM_SEED)


EARLY_STOP = 5
LR = 2e-5
BATCH_SIZE = 32
MAX_GRAD_NORM = 5
N_TAGS = 9
LOG_PER_BATCH = 10

# this is for CoNLL 2003
TRAIN_URL = from_project_root("data/CoNLL2003/conll2003_train.bio")
DEV_URL = from_project_root("data/CoNLL2003/conll2003_dev.bio")
TEST_URL = from_project_root("data/CoNLL2003/conll2003_test.bio")
CHAE_URL = from_project_root("data/CoNLL2003/char_vocab.json")
LABEL_LIST = ["O", "B-ORG", "I-ORG", "B-PER", "I-PER", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]
# this is for Ali
# LABEL_LIST = ["O", "B-Brand Name", "I-Brand Name", "B-Material", "I-Material", "B-Color", "I-Color", "B-Category",
#              "I-Category"]
# TRAIN_URL = from_project_root("data/Ali/Ali-train.iob2")
# DEV_URL = from_project_root("data/Ali/Ali-dev.iob2")
# TEST_URL = from_project_root("data/Ali/Ali-test.iob2")
# CHAE_URL = from_project_root("data/Ali/char_vocab.json")


def train_end2end(n_epochs=3000,
                  dataset_name="CoNLL",
                  train_url=TRAIN_URL,
                  test_url=TEST_URL,
                  dev_url=DEV_URL,
                  learning_rate=LR,
                  batch_size=BATCH_SIZE,
                  early_stop=EARLY_STOP,
                  clip_norm=MAX_GRAD_NORM,
                  device='cuda',
                  save_only_best=True
                  ):
    """
    Train deep exhaustive model, trained best model will be saved at 'data/model/'
    Args:
        n_epochs: number of epochs
        dataset_name: the name of the dataset
        train_url: url to train data
        test_url: url to test data
        dev_url: url to dev data
        learning_rate: learning rate
        batch_size: batch_size
        early_stop: early stop for training
        clip_norm: whether to perform norm clipping, set to 0 if not need
        device: device for torch
        save_only_best: only save model of best performance
    """
    # print arguments
    arguments = json.dumps(vars(), indent=2)
    print("arguments", arguments)
    start_time = datetime.now()

    device = get_device(device)
    train_set = NERDataset(train_url, device=device, bert_model='bert-base-uncased', label_list=LABEL_LIST)
    train_loader = DataLoader(train_set, batch_size=batch_size, drop_last=False,
                              collate_fn=train_set.collate_func)
    # N_TAGS labels and the hidden_size on top of embedding is 200
    model = CallNoteNER(N_TAGS, 200)

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
            if batch_id % LOG_PER_BATCH == 0:
                print("epoch #%d, batch #%d, loss: %.12f, %s" %
                      (epoch, batch_id, loss.item(), datetime.now().strftime("%X")))
            batch_id += 1

        cnt += 1
        # evaluating model use development dataset or and additional test dataset
        f1 = evaluate(model, dev_url, LABEL_LIST)
        if f1 > max_f1:
            max_f1, max_f1_epoch = f1, epoch
            if save_only_best and best_model_url:
                os.remove(best_model_url)
            best_model_url = from_project_root("data/model/%s_lr%f_stop%d_epoch%d_%f.pt" % (dataset_name, LR, EARLY_STOP, epoch, f1))
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
        evaluate(best_model, test_url, LABEL_LIST)

    print(arguments)


def main():
    start_time = datetime.now()
    if CHAE_URL and not exists(CHAE_URL):
        gen_vocab_from_data([TRAIN_URL, DEV_URL, TEST_URL])
    train_end2end(test_url=TEST_URL)
    print("finished in:", datetime.now() - start_time)
    pass


if __name__ == '__main__':
    main()
