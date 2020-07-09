1)6/26/2020
finish the bert + crf model and test on the CoNLL2003 dataset, gets F1 score 0.9313, while the start of art model gets F1 0.9318. I haven't fine tune the model. Fine tunning the model
will improve the F1 score

2)6/29/2020
train on Ali data and test
add a parameter dataset_name, to add the name in the model name
add a parameter id_filter, to only calculate the F1 score bigger than id_filter. Ali's dataset don't consider the O but CoNLL2003 will consider the O

3)7/9/2020
new version BERT_CRF_finetune
add mask to mask padding
remove the zero_grad to fine tune the bert
train on CoNLL2003
modify eval.py to output sentence, true label, predict label
