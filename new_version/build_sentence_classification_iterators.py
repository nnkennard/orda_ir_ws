import csv
import torch.nn as nn
import pickle
import sys
import glob
import json
import torch
import collections
import classification_lib
from torchtext.legacy import data
from transformers import BertTokenizer
import torch.optim as optim


class LabelSet(object):
  Coarse = "coarse"
  Fine = "fine"
  Polarity = "pol"
  Aspect = "asp"
  Relation = "relation"

  ALL = [Coarse, Fine, Polarity, Aspect, Relation]  # Add no label labels??
  REVIEW_LABELS = [Coarse, Fine, Polarity, Aspect]


class ClassificationFormat(object):
  Sentence = "sentence"
  Sequence = "sequence"
  ALL = [
      Sentence  #,Sequence
  ]


class Subset(object):
  train = "train"
  dev = "dev"
  test = "test"
  ALL = [train, dev, test]


REVIEW, REBUTTAL, REVIEW_LABELS = "review rebuttal reviewlabels".split()


def create_review_sentence_examples(example_obj, offset):
  examples = []
  for i, sentence in enumerate(example_obj[REVIEW]):
    builder = {
        key: example_obj[REVIEW_LABELS][i]["labels"][key]
        for key in LabelSet.REVIEW_LABELS
    }
    builder["id"] = offset + i
    builder["sentence"] = sentence["sentence"]
    examples.append(builder)
  return examples


def create_rebuttal_sentence_examples(example_obj):
  return create_sentence_examples(REBUTTAL, example_obj)


ReviewExample = collections.namedtuple("ReviewExample",
                                       ["sentence"] + LabelSet.REVIEW_LABELS)


class ExampleType(object):
  ReviewSentence = "review-sentence"
  RebuttalSentence = "rebuttal-sentence"


def build_iterators(data_dir, batch_size):

  # Form csv files
  offset = 0
  for subset in Subset.ALL:
    glob_path = "/".join([data_dir, "traindev_" + subset, "*.json"])
    review_sentence_examples = []
    rebuttal_sentence_examples = []
    FIELDS = "id sentence".split() + LabelSet.REVIEW_LABELS
    for filename in glob.glob(glob_path):
      with open(filename, 'r') as f:
        example_obj = json.load(f)
        review_sentence_examples += create_review_sentence_examples(
            example_obj, offset)
        offset += len(example_obj[REVIEW])
    with open("".join([ExampleType.ReviewSentence, "_", subset, ".csv"]),
              'w') as f:
      writer = csv.DictWriter(f, FIELDS)
      writer.writeheader()
      for example in review_sentence_examples:
        writer.writerow(example)

  tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  metadata = classification_lib.TokenizerMetadata(tokenizer)

  RAW = data.RawField()
  TEXT = data.Field(
      use_vocab=True,
      batch_first=True,
      tokenize=lambda x: classification_lib.tokenize_and_cut(
          tokenizer, metadata.max_input_length - 2, x),
      #preprocessing=tokenizer.convert_tokens_to_ids,
      init_token=metadata.init_token_idx,
      eos_token=metadata.eos_token_idx,
      pad_token=metadata.pad_token_idx,
      unk_token=metadata.unk_token_idx)
  label_fields = [(field_name,
                   data.LabelField(dtype=torch.int,
                                   use_vocab=True,
                                   sequential=False))
                  for field_name in LabelSet.REVIEW_LABELS]

  fields = [('id', RAW), ('text', TEXT)] + label_fields[:1]

  train_file_name = "".join([ExampleType.ReviewSentence, "_train.csv"])
  train_obj, valid_obj, test_obj = data.TabularDataset.splits(
      path="./",
      train=train_file_name,
      validation=train_file_name.replace("_train.", "_dev."),
      test=train_file_name.replace("_train.", "_test."),
      format='csv',
      skip_header=True,
      fields=fields)

  for name, field in fields:
    if name in ['id']:
      continue
    field.build_vocab(train_obj, valid_obj, test_obj)

  return (data.BucketIterator.splits((train_obj, valid_obj, test_obj),
                                     batch_size=batch_size,
                                     device=device,
                                     sort_key=lambda x: x.id,
                                     sort_within_batch=False),
          classification_lib.DatasetTools(tokenizer, device, metadata,
                                          dict(fields)))


LABEL_GETTERS = {
    "coarse": lambda x: x.coarse,
    "fine": lambda x: x.fine,
    "asp": lambda x: x.asp,
    "pol": lambda x: x.pol,
}

iterators, dataset_tools = build_iterators(sys.argv[1], 512)
train_iterator, valid_iterator, test_iterator = iterators

for label in LabelSet.REVIEW_LABELS:
  print(label)
  field = dataset_tools.field_map[label]
  output_dim = len(field.vocab.stoi)
  print(field.vocab.stoi)
  model = classification_lib.BERTGRUClassifier(dataset_tools.device, output_dim)
  model.to(dataset_tools.device)

  optimizer = optim.Adam(model.parameters())
  #criterion = nn.BCEWithLogitsLoss()
  criterion = nn.CrossEntropyLoss()
  #criterion = nn.BCELoss()

  for epoch in range(20):

    print(len(train_iterator))
    this_epoch_data = classification_lib.do_epoch(model, train_iterator,
                                                  criterion,
                                                  LABEL_GETTERS[label],
                                                  output_dim,
                                                  optimizer, valid_iterator)

    classification_lib.report_epoch(epoch, this_epoch_data)
