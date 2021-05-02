import numpy as np
import random
import torch
import torch.optim as optim

logit_lookup = np.eye(2)

from transformers import BertTokenizer
from transformers import BertTokenizer, BertModel
from torchtext.legacy import data
#from torchtext.legacy import datasets
#from torchtext.experimental.datasets import IMDB

# Setting random seeds
SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

device = torch.device("cpu")


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


def my_dataset_stuff():
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


import torch.nn as nn


class BERTGRUSentiment(nn.Module):

  def __init__(self, bert, hidden_dim, output_dim, n_layers, bidirectional,
               dropout):

    super().__init__()


    self.bert = bert
    embedding_dim = bert.config.to_dict()['hidden_size']
    self.rnn = nn.GRU(embedding_dim,
                      hidden_dim,
                      num_layers=n_layers,
                      bidirectional=bidirectional,
                      batch_first=True,
                      dropout=0 if n_layers < 2 else dropout)

    self.out = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim,
                         output_dim)
    self.dropout = nn.Dropout(dropout)

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
    acc = sum(correct) / len(correct)
    return acc

def train(model, iterator, optimizer, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.train()
    
    for batch in iterator:
        
        optimizer.zero_grad()

        predictions = model(batch.review).squeeze(1)
        labels = sigmoidize_labels(batch, predictions)               
        loss = criterion(predictions, labels)
        
        acc = binary_accuracy(predictions, batch.label)
        
        loss.backward()
        
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def sigmoidize_labels(batch, predictions):
  return torch.tensor([logit_lookup[i] for i in
          batch.label]).to(predictions.device)

def evaluate(model, iterator, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.eval()
    
    with torch.no_grad():
    
        for batch in iterator:


            predictions = model(batch.review).squeeze(1)
            
            loss = criterion(predictions, 
                sigmoidize_labels(batch, predictions))
            
            acc = binary_accuracy(predictions, batch.label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


import time

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def main():

  tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
  print("\nCreating RAW, TEXT, LABEL Field objects ")

  train_obj, valid_obj, test_obj, train_iter = my_dataset_stuff()
  LABEL.build_vocab(train_obj)

  BATCH_SIZE = 128

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
      (train_obj, valid_obj, test_obj), batch_size=BATCH_SIZE, device=device,
      sort_key = lambda x:x.id, sort_within_batch=False)

  bert = BertModel.from_pretrained('bert-base-uncased')
  HIDDEN_DIM = 256
  OUTPUT_DIM = 2
  N_LAYERS = 2
  BIDIRECTIONAL = True
  DROPOUT = 0.25

  model = BERTGRUSentiment(bert, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS,
                           BIDIRECTIONAL, DROPOUT)

  for name, param in model.named_parameters():
    if name.startswith('bert'):
      param.requires_grad = False

  optimizer = optim.Adam(model.parameters())
  criterion = nn.BCEWithLogitsLoss()

  model = model.to(device)
  criterion = criterion.to(device)

  N_EPOCHS = 1000

  best_valid_loss = float('inf')

  for epoch in range(N_EPOCHS):
      
      start_time = time.time()
      
      train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
      valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)
          
      end_time = time.time()
          
      epoch_mins, epoch_secs = epoch_time(start_time, end_time)
          
      if valid_loss < best_valid_loss:
          best_valid_loss = valid_loss
          torch.save(model.state_dict(), 'tut6-model.pt')
      
      print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
      print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
      print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')

  model.load_state_dict(torch.load('tut6-model.pt'))

  test_loss, test_acc = evaluate(model, test_iterator, criterion)

  print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')



if __name__ == "__main__":
  main()
