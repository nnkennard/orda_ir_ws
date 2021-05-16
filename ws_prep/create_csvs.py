import argparse
import csv
import glob
import itertools
import json
import numpy as np
import os
import sys
import random
from tqdm import tqdm

import ws_lib

parser = argparse.ArgumentParser(description='prepare CSVs for ws training')
parser.add_argument('-d',
                    '--datadir',
                    default="data/data_0.1/ws/",
                    type=str,
                    help='path to data file containing score jsons')

random.seed(34)

POS = "1_more_relevant"
NEG = "2_more_relevant"

#FIELDS = "id label d1 d2 q".split()
FIELDS = "id label text".split()

SAMPLES_PER_SENTENCE = 5
PAIRS_PER_FILE = 20
TRAIN_FRAC = 0.8


def bernoulli(mean):
  assert 0.0 < mean < 1.0
  return random.random() < mean

def sample_things(data_obj):

  scores, review, rebuttal = (data_obj["scores"], data_obj["review"],
  data_obj["rebuttal"])

  scores = np.array(scores)
  samples = []

  for reb_i in range(len(rebuttal)):
    samples = []
    i = 0
    while True:
      i += 1
      if i > len(rebuttal):
        break

      rev_i, rev_j = random.choices(range(len(review)), k=2)
      score_1 = scores[reb_i][rev_i]
      score_2 = scores[reb_i][rev_j]
      if score_1 == score_2:
        continue
      else:
        sample_starter = {
            "text":
                " [SEP] ".join([
                    data_obj["review"][rev_i],
                    data_obj["review"][rev_j],
                    data_obj["rebuttal"][reb_i],
                ]).replace("\0", ''),
            "id":
                len(samples)
        }
        if score_1 > score_2:
          sample_starter["label"] = POS
        else:
          sample_starter["label"] = NEG
        samples.append(sample_starter)
      if len(samples) == SAMPLES_PER_SENTENCE:
        break

  return samples

def build_overall_vocab(sentences):
  vocab = set()
  for sent in sentences:
    vocab.update(sent.split())
  fake_sentences = []
  vocab = list(sorted(vocab))
  for start_index in range(0, len(vocab), 100):
    fake_sentences.append(" ".join(vocab[start_index:start_index +
      100]).replace('\0', ""))
  return fake_sentences
  
def main():

  args = parser.parse_args()

  assert args.datadir.endswith('/ws/')

  overall_builders = {"d": [], "q": []}

  all_filenames = glob.glob(args.datadir + "/*.json")

  for batch_i, input_file_start_index in enumerate(tqdm(range(0, len(all_filenames),
    PAIRS_PER_FILE))):
    this_file_filenames = all_filenames[
        input_file_start_index:input_file_start_index + PAIRS_PER_FILE]

    sample_builder = {"train":[], "dev":[]}
    for filename in this_file_filenames:
      subset = "train" if bernoulli(TRAIN_FRAC) else "dev"
      with open(filename, 'r') as f:
        data_obj = json.load(f)

      overall_builders["d"] += data_obj["review"]
      overall_builders["q"] += data_obj["rebuttal"]
      sample_builder[subset] += sample_things(data_obj)

    for subset, examples in sample_builder.items():
      output_file = "".join([
        args.datadir, "batch_", str(batch_i), "_", subset,  ".csv"])
      with open(output_file, 'w') as g:
        writer = csv.DictWriter(g, FIELDS)
        writer.writeheader()
        for i, example in enumerate(examples):
          writer.writerow(example)


  fake_sentences = {}
  for builder_type in ["d", "q"]:
    fake_sentences[builder_type] = build_overall_vocab(
      overall_builders[builder_type])


  stop_len = max([len(fake_sentences["q"]), len(fake_sentences["d"])])
  with open(args.datadir + "/overall_dummy_vocabber.csv", 'w') as h:
    writer = csv.DictWriter(h, FIELDS)
    writer.writeheader()
    for i, (d1, d2, q, label) in enumerate(
        zip(*[
            itertools.cycle(x) for x in [
                fake_sentences["d"], fake_sentences["d"],
                fake_sentences["q"], [POS, NEG]
            ]
        ])):
      if i == stop_len:
        break
      writer.writerow(
          {"id": i,
            "text": " [SEP] ".join([d1, d2, q]),
            "label": label})



if __name__ == "__main__":
  main()
