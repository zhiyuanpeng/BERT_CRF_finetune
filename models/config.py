import json
from datetime import datetime
from utils.path_util import from_project_root
from utils.torch_util import get_logger

early_stop_default = 5
learning_rate_default = 2e-5
batch_size_default = 32
clip_norm_default = 5
n_tags_default = 9
log_per_batch_default = 10
n_epochs_default = 3000
# this can also be replaced with path of bert model
bert_model_name_default = "bert-base-uncased"
device_default = "CUDA"
save_only_best_default = True
dataset_name_default = "CoNLL2003"

# this is for CoNLL 2003
train_url_default = from_project_root("data/CoNLL2003/conll2003_train.bio")
dev_url_default = from_project_root("data/CoNLL2003/conll2003_dev.bio")
test_url_default = from_project_root("data/CoNLL2003/conll2003_test.bio")
c2v_default = from_project_root("data/CoNLL2003/char_vocab.json")
w2v_default = from_project_root("data/CoNLL2003/char_vocab.json")
label_list_default = ["O", "B-ORG", "I-ORG", "B-PER", "I-PER", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]


class Config:
    def __init__(self, args,
                 dataset_name=None,
                 device=None,
                 save_only_best=None,
                 n_epochs=None,
                 bert_model_name=None,
                 early_stop=None,
                 learning_rate=None,
                 batch_size=None,
                 clip_norm=None,
                 n_tags=None,
                 log_per_batch=None,
                 train_url=None,
                 dev_url=None,
                 test_url=None,
                 c2v=None,
                 w2v=None,
                 label_list=None):
        super().__init__()
        if args is not None:
            if args.dataset_name is not None:
                dataset_name = args.dataset_name

            if args.device is not None:
                device = args.device

            if args.save_only_best is not None:
                save_only_best = args.save_only_best

            if args.n_epochs is not None:
                n_epochs = args.n_epochs

            if args.bert_model_name is not None:
                bert_model_name = args.bert_model_name

            if args.early_stop is not None:
                early_stop = args.early_stop

            if args.learning_rate is not None:
                learning_rate = args.learning_rate

            if args.batch_size is not None:
                batch_size = args.batch_size

            if args.clip_norm is not None:
                clip_norm = args.clip_norm

            if args.n_tags is not None:
                n_tags = args.n_tags

            if args.log_per_batch is not None:
                log_per_batch = args.log_per_batch

            if args.train_url is not None:
                train_url = args.train_url

            if args.dev_url is not None:
                dev_url = args.dev_url

            if args.test_url is not None:
                test_url = args.test_url

            if args.c2v is not None:
                c2v = args.c2v

            if args.w2v is not None:
                w2v = args.w2v

            if args.label_list is not None:
                label_list = args.label_list

        if dataset_name is None:
            dataset_name = dataset_name_default

        if device is None:
            device = device_default

        if save_only_best is None:
            save_only_best = save_only_best_default

        if n_epochs is None:
            n_epochs = n_epochs_default

        if bert_model_name is None:
            bert_model_name = bert_model_name_default

        if early_stop is None:
            early_stop = early_stop_default

        if learning_rate is None:
            learning_rate = learning_rate_default

        if batch_size is None:
            batch_size = batch_size_default

        if clip_norm is None:
            clip_norm = clip_norm_default

        if n_tags is None:
            n_tags = n_tags_default

        if log_per_batch is None:
            log_per_batch = log_per_batch_default

        if train_url is None:
            train_url = train_url_default

        if dev_url is None:
            dev_url = dev_url_default

        if test_url is None:
            test_url = test_url_default

        if c2v is None:
            c2v = c2v_default

        if w2v is None:
            w2v = w2v_default

        if label_list is None:
            label_list = label_list_default

        # print arguments
        arguments = json.dumps(vars(), indent=2)
        print("arguments", arguments)

        self.logger = get_logger("../log.txt")
        self.logger.info("This is the begin of the experiment")
        self.logger.info("dataset_name is: %s", dataset_name)
        self.logger.info("device is: %s", device)
        self.logger.info("save_only_best is: %s", str(save_only_best))
        self.logger.info("n_epochs is: %d", n_epochs)
        self.logger.info("bert_model_name is: %d", bert_model_name)
        self.logger.info("early_stop is: %d", early_stop)
        self.logger.info("learning_rate is: %f", learning_rate)
        self.logger.info("batch_size is: %d", batch_size)
        self.logger.info("clip_norm is: %d", clip_norm)
        self.logger.info("n_tags is: %f", n_tags)
        self.logger.info("log_per_batch is: %d", log_per_batch)
        self.logger.info("train_url is: %s", train_url)
        self.logger.info("dev_url is: %s", dev_url)
        self.logger.info("test_url is: %s", test_url)
        self.logger.info("c2v is: %s", c2v)
        self.logger.info("w2v is: %s", w2v)
        self.logger.info("label list is: %s", " ".join(label_list))
        self.logger.info("This is the end of the experiment")
