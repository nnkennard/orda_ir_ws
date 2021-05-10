import collections
import json
import numpy as np
import os
import pickle
import torch

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from rank_bm25 import BM25Okapi
from tqdm import tqdm

import random

random.seed(34)

DATASETS = "unstructured traindev_train traindev_dev traindev_test truetest".split()
STEMMER = PorterStemmer()
STOPWORDS = stopwords.words('english')


def preprocess(sentence_tokens):
  return [
      STEMMER.stem(word).lower() for word in sentence_tokens
      if word.lower() not in STOPWORDS
  ]


def dir_fix(dir_name):
  if not os.path.exists(dir_name):
    os.makedirs(dir_name)


def get_sentences_from_comment(comment_text):
  sentences = []
  for sentence in comment_text["sentences"]:
    sentences.append(comment_text["text"][sentence["start_index"]:sentence["end_index"]])
  return sentences


def main():

  data_dir = "data/data_0.1.3/"
  output_dir = data_dir + "/ws/"
  dir_fix(output_dir)

  corpus = []
  corpus_index_map = {}
  queries = collections.defaultdict(list)

  review_map = {}
  rebuttal_map = {}

  for dataset in DATASETS:
    if 'traindev' not in dataset:
      continue
    input_file = data_dir + dataset + ".json"
    with open(input_file, 'r') as f:
      obj = json.load(f)

    for pair in tqdm(obj["review_rebuttal_pairs"]):
      # Collect sentences
      key = (dataset, str(pair["index"]))
      review_map[key] = get_sentences_from_comment(pair["review_text"])
      rebuttal_map[key] = get_sentences_from_comment(pair["rebuttal_text"])
      
      # Add review sentences to corpus
      start_index = len(corpus)
      corpus += [preprocess(sentence) for sentence in review_map[key]]
      corpus_index_map[key] = (start_index, len(corpus))

      # Add rebuttal sentences to query sentences
      for j, query_sentence in enumerate(rebuttal_map[key]):
        queries[key].append(preprocess(query_sentence))

  model = BM25Okapi(corpus)

  for key, preprocessed_queries in tqdm(queries.items()):
    filename = "".join([output_dir, "/", "_".join(key), ".json"])
    if os.path.exists(filename):
      continue
    relevant_scores = []
    for j, preprocessed_query in enumerate(preprocessed_queries):
      scores = model.get_scores(preprocessed_query)
      start, exclusive_end = corpus_index_map[key]
      blerp = scores[start:exclusive_end].tolist()
      relevant_scores.append(blerp)
    with open(filename, "w") as f:
      json.dump({"review": review_map[key],
                 "rebuttal": rebuttal_map[key],
                 "scores": relevant_scores}, f)
    

if __name__ == "__main__":
  main()
