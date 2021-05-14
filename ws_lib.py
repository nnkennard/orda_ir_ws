
#TEXT_FIELD_NAMES = "d1 d2 q".split()
TEXT_FIELD_NAMES = ["text"]


def get_dataset_tools(data_dir):
  tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  metadata = TokenizerMetadata(tokenizer)
  RAW = data.RawField()
  text_fields = [(field_name, generate_text_field(tokenizer))
               for field_name in TEXT_FIELD_NAMES]
  LABEL = data.LabelField(dtype=torch.float)

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


  return DatasetTools(tokenizer, device, metadata, fields)


def build_iterators(data_dir, train_file_name, dataset_tools, batch_size):
  train_obj, valid_obj = data.TabularDataset.splits(
      path=data_dir,
      train=train_file_name,
      validation=train_file_name.replace("_train_", "_dev_"),
      format='csv', skip_header=True,
      fields=dataset_tools.fields)

  return data.BucketIterator.splits((train_obj, valid_obj),
                                    batch_size=batch_size,
                                    device=dataset_tools.device,
                                    sort_key=lambda x: x.id,
                                    sort_within_batch=False)
