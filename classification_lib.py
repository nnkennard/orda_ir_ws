import torch
import time
import numpy as np

import torch.nn as nn
from torchtext.legacy import data
from transformers import BertTokenizer
from transformers import BertModel
from contextlib import nullcontext


class Hyperparams(object):
  hidden_dim = 512
  output_dim = 2
  n_layers = 2
  bidirectional = True
  dropout = 0.25
  n_epochs = 100
  batch_size = 128
  patience = 20

class TokenizerMetadata(object):

  def __init__(self, tokenizer):
    self.tokenizer = tokenizer
    self.init_token_idx = self.get_index(tokenizer.cls_token)
    self.eos_token_idx = self.get_index(tokenizer.sep_token)
    self.pad_token_idx = self.get_index(tokenizer.pad_token)
    self.unk_token_idx = self.get_index(tokenizer.unk_token)
    self.max_input_length = tokenizer.max_model_input_sizes['bert-base-uncased']

  def get_index(self, token):
    return self.tokenizer.convert_tokens_to_ids(token)


def tokenize_and_cut(tokenizer, cut_idx, sentence):
  tokens = tokenizer.tokenize(sentence)
  tokens = tokens[:cut_idx]
  return tokens


def generate_text_field(tokenizer):
  metadata = TokenizerMetadata(tokenizer)
  return data.Field(use_vocab=False,
                    batch_first=True,
                    tokenize=lambda x: tokenize_and_cut(
                        tokenizer, metadata.max_input_length - 2, x),
                    preprocessing=tokenizer.convert_tokens_to_ids,
                    init_token=metadata.init_token_idx,
                    eos_token=metadata.eos_token_idx,
                    pad_token=metadata.pad_token_idx,
                    unk_token=metadata.unk_token_idx)


class DatasetTools(object):

  def __init__(self, tokenizer, device, metadata, field_map):
    self.tokenizer = tokenizer
    self.device = device
    self.metadata = metadata
    self.field_map = field_map


class BERTGRUClassifier(nn.Module):

  def __init__(self, device, output_dim=None):

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

    if output_dim is None:
      output_dim = Hyperparams.output_dim
    self.out = nn.Linear(
        Hyperparams.hidden_dim *
        2 if Hyperparams.bidirectional else Hyperparams.hidden_dim, output_dim)
    self.dropout = nn.Dropout(Hyperparams.dropout)

    for name, param in self.named_parameters():
      if name.startswith('bert'):
        param.requires_grad = False

  def forward(self, text):
    #text ~ [batch size, sent len]

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
      #hidden ~ [batch size, hid dim]

    output = self.out(hidden)
    #output ~ [batch size, out dim]

    return output


def binary_accuracy(preds, y):
  correct = (torch.argmax(preds, 1) == y).float()
  return sum(correct)


def loss_acc_wrapper(predictions, labels, criterion):
  return criterion(predictions, labels), binary_accuracy(predictions, labels)

  
def train_or_evaluate(model,
                      iterator,
                      criterion,
                      label_getter,
                      output_dim,
                      mode,
                      optimizer=None):
  assert mode in "train evaluate".split()
  is_train = mode == "train"

  epoch_loss = 0.0
  epoch_acc = 0.0
  example_counter = 0

  if is_train:
    model.train()
    context = nullcontext()
    assert optimizer is not None
  else:
    model.eval()
    context = torch.no_grad()

  logit_lookup = torch.Tensor(np.eye(output_dim)).to(model.device)

  with context:
    for batch in iterator:
      example_counter += len(batch)
      if is_train:
        optimizer.zero_grad()

      predictions = model(batch.text).squeeze(1)
      labels = label_getter(batch)
      loss, acc = loss_acc_wrapper(predictions, labels, criterion)

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


def do_epoch(model, train_iterator, criterion, label_getter, output_dim, optimizer,
             valid_iterator):

  start_time = time.time()
  train_loss, train_acc = train_or_evaluate(model, train_iterator, criterion,
                                            label_getter, output_dim, "train", optimizer)
  valid_loss, valid_acc = train_or_evaluate(model, valid_iterator, criterion,
                                            label_getter, output_dim, "evaluate")
  end_time = time.time()
  return EpochData(start_time, end_time, train_loss, train_acc, valid_loss,
                   valid_acc)


def report_epoch(epoch, epoch_data):
  print((f'Epoch: {epoch+1:02} | Epoch Time: {epoch_data.elapsed_mins}m '
         f'{epoch_data.elapsed_secs}s\n'
         f'\tTrain Loss: {epoch_data.train_loss:.3f} | Train Acc: '
         f'{epoch_data.train_acc*100:.2f}%'
         f'\t Val. Loss: {epoch_data.val_loss:.3f} |  Val. Acc: '
         f'{epoch_data.val_acc*100:.2f}%'))


# Need iterator builders
