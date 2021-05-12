import collections
import csv
import glob
import json
import os
import sys
import torch


from torchtext.legacy import data
from transformers import BertTokenizer

import ws_lib

TOKEN_LABEL_FIELD = data.Field(use_vocab=False,
                    batch_first=True,
                    tokenize=lambda x: x.split(),
                    #preprocessing=tokenizer.convert_tokens_to_ids,
                    )

class LabelSet(object):
  Coarse = "coarse"
  Fine = "fine"
  Polarity = "pol"
  Aspect = "asp"
  Relation = "relation"

  ALL = [Coarse, Fine, Polarity, Aspect, Relation] # Add no label labels??
  REVIEW_LABELS = [Coarse, Fine, Polarity, Aspect]


class ClassificationFormat(object):
  Sentence = "sentence"
  Sequence = "sequence"
  ALL = [Sentence #,Sequence
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
    builder = {key:example_obj[REVIEW_LABELS][i]["labels"][key]
        for key in LabelSet.REVIEW_LABELS}
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
    with open("".join(
      [ExampleType.ReviewSentence, "_", subset, ".json"]), 'w') as f:
      writer = csv.DictWriter(f, FIELDS)
      writer.writeheader()
      for example in review_sentence_examples:
        writer.writerow(example)


  # Get dataset tools

  # Build iterators

  pass



def get_dataset_tools(data_dir):
  tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  metadata = ws_lib.TokenizerMetadata(tokenizer)
  RAW = data.RawField()
  SENTENCE = ws_lib.generate_text_field(tokenizer)
  LABEL = data.LabelField(dtype=torch.float)
  fields = [('id', RAW), ('sent', SENTENCE), ('label', LABEL)]

  # Create fake train obj

  temp_obj, = data.TabularDataset.splits(
      path=data_dir, train='overall_dummy_vocabber.csv',
      format='csv', fields=fields, skip_header=True)
  for name, field in fields:
    print("Creating vocab for ", name)
    if name in ['id']:
      continue
    field.build_vocab(temp_obj)


  return DatasetTools(tokenizer, device, metadata, fields)



def main():
  output_dir = "./"
  
  iterators = build_iterators(sys.argv[1],  16)


if __name__ == "__main__":
  main()
