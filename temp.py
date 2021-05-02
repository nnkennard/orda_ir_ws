import numpy as np
import time
from contextlib import nullcontext
import random
import torch
import torch.optim as optim
from transformers import BertTokenizer
from transformers import BertTokenizer, BertModel
from torchtext.legacy import data
import torch.nn as nn

# Setting random seeds
SEED = 1234
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
  n_epochs = 1000
  batch_size = 128


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


def get_fields(tokenizer):
  metadata = TokenizerMetadata(tokenizer)
  TEXT = data.Field(batch_first=True,
                    use_vocab=False,
                    tokenize=lambda x: tokenize_and_cut(
                        tokenizer, metadata.max_input_length - 2, x),
                    preprocessing=tokenizer.convert_tokens_to_ids,
                    init_token=metadata.init_token_idx,
                    eos_token=metadata.eos_token_idx,
                    pad_token=metadata.pad_token_idx,
                    unk_token=metadata.unk_token_idx)
  LABEL = data.LabelField(dtype=torch.float)
  return TEXT, LABEL


RAW = data.RawField()
TEXT = data.Field(
    sequential=True,
    init_token='',  # start of sequence
    eos_token='',  # end of sequence
    lower=True,
    tokenize=data.utils.get_tokenizer("basic_english"),
)
LABEL = data.Field(sequential=False,
                   use_vocab=False,
                   unk_token=None,
                   is_target=True)


def my_dataset_stuff(device):
  (train_obj, valid_obj, test_obj) = \
    data.TabularDataset.splits(
    path="./data/",
    train='train_data.csv',
    validation='valid_data.csv',
    test='test_data.csv',
    format='csv',
    fields=[('id', RAW), ('review', TEXT),
      ('label', LABEL)])

  TEXT.build_vocab(train_obj)

  train_iter = data.BucketIterator(dataset=train_obj,
                                   batch_size=2,
                                   sort_key=lambda x: len(x.review),
                                   shuffle=True,
                                   device=device)

  return train_obj, valid_obj, test_obj, train_iter


class BERTGRUSentiment(nn.Module):

  def __init__(self, bert):

    super().__init__()

    self.bert = bert
    self.rnn = nn.GRU(
        bert.config.to_dict()['hidden_size'],
        Hyperparams.hidden_dim,
        num_layers=Hyperparams.n_layers,
        bidirectional=Hyperparams.bidirectional,
        batch_first=True,
        dropout=0 if Hyperparams.n_layers < 2 else Hyperparams.dropout)

    self.out = nn.Linear(
        Hyperparams.hidden_dim, Hyperparams.output_dim *
        2 if Hyperparams.bidirectional else Hyperparams.output_dim)
    self.dropout = nn.Dropout(Hyperparams.dropout)

  def forward(self, text):
    #text = [batch size, sent len]

    with torch.no_grad():
      embedded = self.bert(torch.transpose(text, 0, 1))[0]
      #embedded = [batch size, sent len, emb dim]

    _, hidden = self.rnn(embedded)
    #hidden = [n layers * n directions, batch size, emb dim]

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
  return (criterion(
      predictions,
      torch.tensor([logit_lookup[i] for i in batch.label
                   ]).to(predictions.device)),
          binary_accuracy(predictions, batch.label))


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
      print(train_or_evaluate, len(batch))
      example_counter += len(batch)
      if is_train:
        optimizer.zero_grad()

      predictions = model(batch.review).squeeze(1)
      loss, acc = loss_acc_wrapper(batch, predictions, criterion)

      if is_train:
        loss.backward()
        optimizer.step()

      epoch_loss += loss.item() * len(predictions)
      epoch_acc += acc.item()

  return epoch_loss / example_counter, epoch_acc / example_counter


def epoch_time(start_time, end_time):
  elapsed_time = end_time - start_time
  elapsed_mins = int(elapsed_time / 60)
  elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
  return elapsed_mins, elapsed_secs


def main():

  tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  train_obj, valid_obj, test_obj, train_iter = my_dataset_stuff(device)
  LABEL.build_vocab(train_obj)
  train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
      (train_obj, valid_obj, test_obj),
      batch_size=Hyperparams.batch_size,
      device=device,
      sort_key=lambda x: x.id,
      sort_within_batch=False)

  bert = BertModel.from_pretrained('bert-base-uncased')
  model = BERTGRUSentiment(bert)

  for name, param in model.named_parameters():
    if name.startswith('bert'):
      param.requires_grad = False

  optimizer = optim.Adam(model.parameters())
  criterion = nn.BCEWithLogitsLoss()

  model = model.to(device)

  best_valid_loss = float('inf')

  for epoch in range(Hyperparams.n_epochs):

    start_time = time.time()

    train_loss, train_acc = train_or_evaluate(model, train_iterator, criterion,
                                              "train", optimizer)
    valid_loss, valid_acc = train_or_evaluate(model, valid_iterator, criterion,
                                              "evaluate")

    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    if valid_loss < best_valid_loss:
      print("Best loss", train_loss, train_acc)
      best_valid_loss = valid_loss
      torch.save(model.state_dict(), 'tut6-model.pt')

    print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')

  model.load_state_dict(torch.load('tut6-model.pt'))

  val_loss, val_acc = train_or_evaluate(model, valid_iterator, criterion,
                                        "evaluate")

  print(f'\t Val. Loss: {val_loss:.3f} |  Val. Acc: {val_acc*100:.2f}%')


if __name__ == "__main__":
  main()
