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


def bernoulli(mean):
  assert 0.0 < mean < 1.0
  return random.random() < mean


def assign_categories(num_items, mean):
  return ["train" if bernoulli(mean) else "dev" for _ in range(num_items)]


def main():

  args = parser.parse_args()

  assert args.datadir.endswith('/ws/')

  overall_builders = {"d": [], "q": []}

  for input_file in tqdm(glob.glob(args.datadir + "/*.json")):

    with open(input_file, 'r') as f:
      data_obj = json.load(f)

    overall_builders["d"] += data_obj["review"]
    overall_builders["q"] += data_obj["rebuttal"]

    num_samples = 5
    # Split queries into train and dev
    subset_map = assign_categories(len(data_obj["rebuttal"]), 0.8)

    scores = np.array(data_obj["scores"])

    sample_builder = {"train": [], "dev": []}

    for reb_i in range(len(data_obj["rebuttal"])):
      samples = []
      i = 0
      while True:
        i += 1
        if i > len(data_obj["rebuttal"]):
          break

        rev_i, rev_j = random.choices(range(len(data_obj["review"])), k=2)
        score_1 = scores[reb_i][rev_i]
        score_2 = scores[reb_i][rev_j]
        if score_1 == score_2:
          continue
        else:
          sample_starter = {
              #"d1": data_obj["review"][rev_i],
              #"d2": data_obj["review"][rev_j],
              #"q": data_obj["rebuttal"][reb_i],
              "text":
                  " [SEP] ".join([
                      data_obj["review"][rev_i],
                      data_obj["review"][rev_j],
                      data_obj["rebuttal"][reb_i],
                  ]),
              "id":
                  len(samples)
          }
          if score_1 > score_2:
            sample_starter["label"] = POS
          else:
            sample_starter["label"] = NEG
          samples.append(sample_starter)

        if len(samples) == num_samples:
          break

      subset = subset_map[reb_i]
      sample_builder[subset] += samples

    assert input_file.endswith('.json')
    for subset, examples in sample_builder.items():
      output_file = input_file.replace(".json", "_" + subset + ".csv")
      with open(output_file, 'w') as g:
        writer = csv.DictWriter(g, FIELDS)
        writer.writeheader()
        for i, example in enumerate(examples):
          writer.writerow(example)

  stop_len = max([len(overall_builders["q"]), len(overall_builders["d"])])
  with open(args.datadir + "/overall_dummy_vocabber.csv", 'w') as h:
    writer = csv.DictWriter(h, FIELDS)
    for i, (d1, d2, q, label) in enumerate(
        zip(*[
            itertools.cycle(x) for x in [
                overall_builders["d"], overall_builders["d"],
                overall_builders["q"], [POS, NEG]
            ]
        ])):
      if i == stop_len:
        break
      writer.writerow({
          "id": i,
          # "d1": d1, "d2": d2, "q": q,
          "text": " [SEP] ".join([d1, d2, q]),
          "label": label
      })


if __name__ == "__main__":
  main()
