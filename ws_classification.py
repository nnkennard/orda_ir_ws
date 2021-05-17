import argparse
from transformers import BertTokenizer
from transformers import BertModel
import numpy as np
import glob
import time
from contextlib import nullcontext
import random
import torch
import torch.optim as optim
from torchtext.legacy import data
import torch.nn as nn
from tqdm import tqdm

import classification_lib

parser = argparse.ArgumentParser(description='prepare CSVs for ws training')
parser.add_argument('-d',
                    '--datadir',
                    default="data/data_0.1.3/ws/",
                    type=str,
                    help='path to data file containing score jsons')




# Setting random seeds
SEED = 43
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

logit_lookup = np.eye(2)

TEXT_FIELD_NAMES = ["text"]


def generate_text_field(tokenizer):
  metadata = classification_lib.TokenizerMetadata(tokenizer)
  return data.Field(use_vocab=False,
                    batch_first=True,
                    tokenize=lambda x: classification_lib.tokenize_and_cut(
                        tokenizer, metadata.max_input_length - 2, x),
                    preprocessing=tokenizer.convert_tokens_to_ids,
                    init_token=metadata.init_token_idx,
                    eos_token=metadata.eos_token_idx,
                    pad_token=metadata.pad_token_idx,
                    unk_token=metadata.unk_token_idx)




def get_dataset_tools(data_dir):
  print("Starting")
  tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  metadata = classification_lib.TokenizerMetadata(tokenizer)
  RAW = data.RawField()
  text_fields = [(field_name, generate_text_field(tokenizer))
               for field_name in TEXT_FIELD_NAMES]
  LABEL = data.LabelField(dtype=torch.long)

  fields = [('id', RAW), ('label', LABEL)] + text_fields 

  # Create fake train obj

  print("Creating temp object")
  temp_obj, = data.TabularDataset.splits(
      path=data_dir, train='overall_dummy_vocabber.csv',
      format='csv', fields=fields, skip_header=True)
  for name, field in fields:
    print("Creating vocab for ", name)
    if name in ['id']:
      continue
    field.build_vocab(temp_obj)


  return classification_lib.DatasetTools(tokenizer, device, metadata, fields)


def build_iterators(data_dir, train_file_name, dataset_tools, batch_size):
  train_obj, valid_obj = data.TabularDataset.splits(
      path=data_dir,
      train=train_file_name,
      validation=train_file_name.replace("_train_", "_dev_"),
      format='csv', skip_header=True,
      fields=dataset_tools.field_map)

  return data.BucketIterator.splits((train_obj, valid_obj),
                                    batch_size=batch_size,
                                    device=dataset_tools.device,
                                    sort_key=lambda x: x.id,
                                    sort_within_batch=False)

def main():

  args = parser.parse_args()

  dataset_tools = get_dataset_tools(args.datadir)

  model = classification_lib.BERTGRUClassifier(
    dataset_tools.device).to(dataset_tools.device)
  optimizer = optim.Adam(model.parameters())
  criterion = nn.CrossEntropyLoss()

  all_train_iterator, all_valid_iterator, = build_iterators(
      args.datadir, "all_train.csv", dataset_tools,
      classification_lib.Hyperparams.batch_size)

  model_save_name = "ws_ir_model.pt"
  best_valid_loss = float('inf')
  best_valid_epoch = None

  patience = 5

  for epoch in range(100):
    for train_file in tqdm(sorted(glob.glob(args.datadir+"/*train.csv"))[:10]):
      if 'all' in train_file:
        continue
      train_file_name = train_file.split('/')[-1]

      train_iterator, valid_iterator, = build_iterators(
          args.datadir, train_file_name, dataset_tools,
          classification_lib.Hyperparams.batch_size)

      this_epoch_data = classification_lib.do_epoch(
        model, train_iterator, criterion, lambda x:x.label, 2, optimizer,
          valid_iterator)

    this_epoch_data = classification_lib.do_epoch(
      model, all_train_iterator, criterion, lambda x:x.label, 2,
      optimizer, all_valid_iterator, eval_both=True)
    classification_lib.report_epoch(epoch, this_epoch_data)

    if this_epoch_data.val_loss < best_valid_loss:
      print("Best validation loss; saving model from epoch ", epoch)
      best_valid_loss = this_epoch_data.val_loss
      torch.save(model.state_dict(), model_save_name)
      best_valid_epoch = epoch

    if best_valid_epoch < (epoch - patience):
      break
  
  model.load_state_dict(torch.load('tut6-model.pt'))



if __name__ == "__main__":
  main()
