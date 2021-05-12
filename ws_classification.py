import argparse
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

import ws_lib


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


class Hyperparams(object):
  hidden_dim = 256
  output_dim = 2
  n_layers = 2
  bidirectional = True
  dropout = 0.25
  n_epochs = 20
  batch_size = 128


class BERTGRUSentiment(nn.Module):

  def __init__(self, device):

    super().__init__()

    self.bert = BertModel.from_pretrained('bert-base-uncased')
    self.device = device

    self.rnn = nn.GRU(
        self.bert.config.to_dict()['hidden_size'],
        Hyperparams.hidden_dim,
        num_layers=Hyperparams.n_layers,
        bidirectional=Hyperparams.bidirectional,
        batch_first=True,
        dropout=0 if Hyperparams.n_layers < 2 else Hyperparams.dropout)

    self.out = nn.Linear(
        Hyperparams.hidden_dim *
        2 if Hyperparams.bidirectional else Hyperparams.hidden_dim,
        Hyperparams.output_dim)
    self.dropout = nn.Dropout(Hyperparams.dropout)

    for name, param in self.named_parameters():
      if name.startswith('bert'):
        param.requires_grad = False

  def forward(self, text):
    #text = [batch size, sent len]

    with torch.no_grad():
      embedded = self.bert(text)[0]
      #embedded ~ [batch size, sent len, emb dim]

    _, hidden = self.rnn(embedded)
    #hidden ~ [n layers * n directions, batch size, emb dim]


    if self.rnn.bidirectional:
      hidden = self.dropout(
          torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
    else:
      hidden = self.dropout(hidden[-1, :, :])
      #hidden = [batch size, hid dim]

    output = self.out(hidden)
    #output = [batch size, out dim]

    return output


def count_parameters(model):
  return sum(p.numel() for p in model.parameters() if p.requires_grad)


def binary_accuracy(preds, y):
  """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
  correct = (torch.argmax(preds, 1) == y).float()
  return sum(correct)


def loss_acc_wrapper(batch, predictions, criterion):
  new_label = torch.tensor([logit_lookup[int(i)] for i in batch.label
                           ]).to(predictions.device)
  return (criterion(predictions,
                    new_label), binary_accuracy(predictions, batch.label))


def train_or_evaluate(model,
                      iterator,
                      criterion,
                      train_or_evaluate,
                      optimizer=None):
  assert train_or_evaluate in "train evaluate".split()
  is_train = train_or_evaluate == "train"

  epoch_loss = 0
  epoch_acc = 0
  example_counter = 0

  if is_train:
    model.train()
    context = nullcontext()
    assert optimizer is not None
  else:
    model.eval()
    context = torch.no_grad()

  with context:
    for batch in iterator:
      example_counter += len(batch)
      if is_train:
        optimizer.zero_grad()


      #predictions = model(batch.q, batch.d1, batch.d2).squeeze(1)
      predictions = model(batch.text).squeeze(1)
      loss, acc = loss_acc_wrapper(batch, predictions, criterion)

      if is_train:
        loss.backward()
        optimizer.step()

      epoch_loss += loss.item() * len(predictions)
      epoch_acc += acc.item()

  if not example_counter:
    return 999999.9, 999999.9
  return epoch_loss / example_counter, epoch_acc / example_counter


class EpochData(object):

  def __init__(self, start_time, end_time, train_loss, train_acc, val_loss,
               val_acc):
    self.train_loss = train_loss
    self.train_acc = train_acc
    self.val_loss = val_loss
    self.val_acc = val_acc

    elapsed_time = end_time - start_time
    self.elapsed_mins = int(elapsed_time / 60)
    self.elapsed_secs = int(elapsed_time - (self.elapsed_mins * 60))


def do_epoch(model, train_iterator, criterion, optimizer, valid_iterator):

  start_time = time.time()
  train_loss, train_acc = train_or_evaluate(model, train_iterator, criterion,
                                            "train", optimizer)
  valid_loss, valid_acc = train_or_evaluate(model, valid_iterator, criterion,
                                            "evaluate")
  end_time = time.time()
  return EpochData(start_time, end_time, train_loss, train_acc, valid_loss,
                   valid_acc)

def report_epoch(epoch, epoch_data):
  print((
  f'Epoch: {epoch+1:02} | Epoch Time: {epoch_data.elapsed_mins}m '
  f'{epoch_data.elapsed_secs}s\n'
  f'\tTrain Loss: {epoch_data.train_loss:.3f} | Train Acc: '
  f'{epoch_data.train_acc*100:.2f}%'
  f'\t Val. Loss: {epoch_data.val_loss:.3f} |  Val. Acc: '
  f'{epoch_data.val_acc*100:.2f}%'))


def main():

  args = parser.parse_args()

  dataset_tools = ws_lib.get_dataset_tools(args.datadir)

  model = BERTGRUSentiment(dataset_tools.device).to(dataset_tools.device)
  optimizer = optim.Adam(model.parameters())
  criterion = nn.BCEWithLogitsLoss()

  for epoch in range(Hyperparams.n_epochs):
    for train_file in tqdm(sorted(glob.glob(args.datadir+"/*train.csv"))):
      print(train_file)
      train_file_name = train_file.split('/')[-1]

      train_iterator, valid_iterator, = ws_lib.build_iterators(
          args.datadir, train_file_name, dataset_tools, Hyperparams.batch_size)

      this_epoch_data = do_epoch(model, train_iterator, criterion, optimizer,
          valid_iterator)

      report_epoch(epoch, this_epoch_data)

      #if this_epoch_data.val_loss < best_valid_loss:
      #  best_valid_loss = this_epoch_data.val_loss
      #  torch.save(model.state_dict(), 'tut6-model.pt')

    #model.load_state_dict(torch.load('tut6-model.pt'))



if __name__ == "__main__":
  main()
