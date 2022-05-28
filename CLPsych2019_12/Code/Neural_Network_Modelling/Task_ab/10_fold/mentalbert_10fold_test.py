#!/usr/bin/env python
# coding: utf-8

from __future__ import absolute_import, division, print_function

import argparse
import sys
import csv
import logging
import os
import random
import sys

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from transformers import AutoTokenizer as tokenizer
from transformers import AutoModelForSequenceClassification, AutoConfig

import random
output_dir = "/home/yy452/rds/rds-gvdd-Yuap0gjVpKM/yy452/CLPsych2019_12/output/task_a_emoji/mentalbert_10fold"

csv.field_size_limit(sys.maxsize)
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO,
                    filename = os.path.join(output_dir, "logging.txt"))
logger = logging.getLogger(__name__)



class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label



class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines


def get_tp_fp_fn(logits, labels):
    assert labels.shape[1] == 1
    labels = labels.squeeze()
    predictions = np.argmax(logits, axis=1)
    labels, predictions = labels.astype(int), predictions.astype(int)
    tp = np.sum(np.logical_and(predictions == 1, labels == 1))
    fp = np.sum(np.logical_and(predictions == 1, labels == 0))
    fn = np.sum(np.logical_and(predictions == 0, labels == 1))
    return tp, fp, fn

def compute_metrics(tp, fp, fn):
  precision = tp / (tp + fp + np.finfo(float).eps)
  recall = tp / (tp + fn + np.finfo(float).eps)
  f1 = 2 * precision * recall / (precision + recall + np.finfo(float).eps)
  return precision, recall, f1

class CLPsychProcessor(DataProcessor):
    """Processor for the CLPsych data set."""
    def get_train_examples(self, data_dir, k):
        """See base class."""
        file_name = "User_Posts_Processed_Train_Final_{}.tsv".format(k)
        return self._create_examples(
            # self._read_tsv(os.path.join(data_dir, "User_Posts_Processed_Train_Final.tsv")), "train")
            self._read_tsv(os.path.join(data_dir, file_name)), "train")

    def get_dev_examples(self, data_dir,k):
        """See base class."""
        file_name = "User_Posts_Processed_Dev_Final_{}.tsv".format(k)
        return self._create_examples(
            # self._read_tsv(os.path.join(data_dir, "User_Posts_Processed_Dev_Final.tsv")), "dev")
            self._read_tsv(os.path.join(data_dir, file_name)), "dev")


    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples_test(
            # self._read_tsv(os.path.join(data_dir, "User_Posts_Processed_Test_Final.tsv")), "test")
            self._read_tsv(os.path.join(data_dir, "Full_Test_Data.tsv")), "test")


    def get_labels(self):
        """See base class."""
        return ["a", "b", "c", "d"]

    def _create_examples_test(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        random.seed(9001)
        req = list()
        for i in range(0, len(lines)):
            req.append(i)
        req_final = random.sample(req, len(lines))
        for i in req_final:
            if i == 0:
                continue
            guid = lines[i][0]
            text_a = lines[i][1]
            text_b = None
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b))
        return examples

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        random.seed(9001)
        req = list()
        for i in range(0, len(lines)):
            req.append(i)
        req_final = random.sample(req, len(lines))	
        for i in req_final:
            if i == 0:
                continue
            guid = lines[i][0]
            text_a = lines[i][1]
            text_b = None
            label = lines[i][2]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label : i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = None
        if(example.label is not None):
            label_id = label_map[example.label]
        #if ex_index < 5:
        #    logger.info("*** Example ***")
        #    logger.info("guid: %s" % (example.guid))
        #    logger.info("tokens: %s" % " ".join(
        #            [str(x) for x in tokens]))
        #    logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        #    logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        #    logger.info(
        #            "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        #    logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_id))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    pred.extend(outputs)
    true.extend(labels)
    return np.sum(outputs == labels)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
    device, n_gpu, False, False))

tokenizer = tokenizer.from_pretrained('mental/mental-bert-base-uncased', do_lower_case=True)
processor = CLPsychProcessor()
global_step = 0
nb_tr_steps = 0
tr_loss = 0
label_list = processor.get_labels()

batch_size = 8
epochs = 75
max_seq_length = 384
data_dir = '/home/yy452/rds/rds-gvdd-Yuap0gjVpKM/yy452/CLPsych2019_12/Dataset/add_expert/emoticons_add'
complete_logits = None


eval_examples = processor.get_test_examples(data_dir)
eval_features = convert_examples_to_features(
    eval_examples, label_list, max_seq_length, tokenizer)
complete_user_ids = list()

for example in eval_examples:
    complete_user_ids.append(example.guid)	
logger.info("***** Running evaluation *****")
logger.info("  Num examples = %d", len(eval_examples))
logger.info("  Batch size = %d", batch_size)
all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
all_label_ids = torch.tensor([0 for f in eval_features], dtype=torch.long)

eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
# Run prediction for full data
eval_sampler = SequentialSampler(eval_data)
eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=batch_size)
models = []

k=10
for i in range(k):
    # output_dir = "/home/yy452/rds/rds-gvdd-Yuap0gjVpKM/yy452/CLPsych2019_12/output/01022022_expert_1000"
    bert_model = "mentalbert_{}.bin".format(i+1)
    output_model_file = os.path.join(output_dir, bert_model)
    output_config_file = os.path.join(output_dir, "config.json")
    config = AutoConfig.from_pretrained(output_config_file)
    model_state_dict = torch.load(output_model_file)
    model = AutoModelForSequenceClassification.from_pretrained("mental/mental-bert-base-uncased", state_dict=model_state_dict, num_labels=4)
    model.to(device)
    model.eval()
    models.append(model)


# model avarage logits
pred = list()
true = list()   
eval_loss, eval_accuracy = 0, 0
nb_eval_steps, nb_eval_examples = 0, 0
complete_label_ids = list()
complete_outputs = list()
complete_logits = []
for input_ids, input_mask, segment_ids, label_ids in eval_dataloader:
    input_ids = input_ids.to(device)
    input_mask = input_mask.to(device)
    segment_ids = segment_ids.to(device)
    label_ids = label_ids.to(device)
    outputs_models = []
    logits = None
    with torch.no_grad():
        for i in range(k):
            outputs_models.append(models[i](input_ids, 
                            token_type_ids=None, 
                            attention_mask=input_mask))

    # Get the "logits" output by the model. The "logits" are the output
    # values prior to applying an activation function like the softmax.
    for i, outputs in enumerate(outputs_models):
        logits_new = outputs[0]

        # Move logits and labels to CPU
        logits_new = logits_new.detach().cpu().numpy()
        if i == 0:
            logits = logits_new
        else:
            logits = (logits * i + logits_new)/(i+1)
        
    outputs = np.argmax(logits, axis=1)
    complete_logits.append(logits)
    complete_outputs.extend(outputs)
    label_ids = label_ids.to('cpu').numpy()
    complete_label_ids.extend(label_ids)
    tmp_eval_accuracy = accuracy(logits, label_ids)

    eval_accuracy += tmp_eval_accuracy

    nb_eval_examples += input_ids.size(0)
    nb_eval_steps += 1


output_file = os.path.join(output_dir, "results.csv")
outcsv = open(output_file,'w', encoding = 'utf8', newline='')
writer = csv.writer(outcsv,quotechar = '"')
writer.writerow(["User","results"])
for user,pred in zip(complete_user_ids, complete_outputs):
    writer.writerow([user,pred])
outcsv.close()
    