#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import random
import pandas as pd
import numpy as np
import csv
import tensorflow as tf
import torch
from sklearn.model_selection import train_test_split
import textwrap
import tqdm as tqdm
import progressbar
import keras
from keras.preprocessing.sequence import pad_sequences
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup
import time
import datetime
import json


# In[ ]:


df = # path to ILDC multi dataset


# In[ ]:


train_set = df.query(" split=='train' ")
test_set = df.query(" split=='test' ")
validation_set = df.query(" split=='dev' ")


# In[ ]:


len(train_set), len(validation_set), len(test_set)


# In[ ]:


from transformers import PreTrainedModel, PreTrainedTokenizer, PretrainedConfig
from transformers import BertForSequenceClassification, BertTokenizer, BertConfig
from transformers import RobertaForSequenceClassification, RobertaTokenizer, RobertaConfig
from transformers import XLNetForSequenceClassification, XLNetTokenizer, XLNetConfig
from transformers import XLMForSequenceClassification, XLMTokenizer, XLMConfig
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer, DistilBertConfig
from transformers import AutoTokenizer, AutoModel, AutoConfig

MODEL_CLASSES = {
    'InLegalBERT': (BertForSequenceClassification, AutoTokenizer, AutoConfig),
    'InCaseLawBERT': (BertForSequenceClassification, AutoTokenizer, AutoConfig),
    'xlnet': (XLNetForSequenceClassification, XLNetTokenizer, XLNetConfig),
    'LegalBERT': (RobertaForSequenceClassification, RobertaTokenizer, RobertaConfig)}

model_type = ###--> CHANGE WHAT MODEL YOU WANT HERE!!! <--###
model_class, tokenizer_class, config_class = MODEL_CLASSES[model_type]
model_name = ### hugingface library model path


# In[ ]:


def att_masking(input_ids):
  attention_masks = []
  for sent in input_ids:
    att_mask = [int(token_id > 0) for token_id in sent]
    attention_masks.append(att_mask)
  return attention_masks


# In[ ]:


def grouped_input_ids(all_toks):
    splitted_toks = []
    CLS = tokenizer.cls_token
    SEP = tokenizer.sep_token
    e_sents = []
    
    if(len(all_toks)>510):
        splitted_toks = all_toks[510:]
    splitted_toks = [CLS] + all_toks + [SEP]
#     print(splitted_toks)
    encoded_sent = tokenizer.convert_tokens_to_ids(splitted_toks)
    e_sents.append(encoded_sent)

    e_sents = pad_sequences(e_sents, maxlen=512, value=0, dtype="long", padding="post")
    att_masks = att_masking(e_sents)
    return e_sents, att_masks


# In[ ]:


def generate_np_files_for_training(dataf, tokenizer):
  all_input_ids, all_att_masks, all_labels = [], [], []
  for i in progressbar.progressbar(range(len(dataf['text']))):
    text = dataf['text'].iloc[i]
    toks = tokenizer.tokenize(text)
    if(len(toks) > 510):
      toks = toks[len(toks)-510:]

    splitted_input_ids, splitted_att_masks = grouped_input_ids(toks)
    doc_label = dataf['label'].iloc[i]
    for i in range(len(splitted_input_ids)):
      all_input_ids.append(splitted_input_ids[i])
      all_att_masks.append(splitted_att_masks[i])
      all_labels.append(doc_label)

  return all_input_ids, all_att_masks, all_labels


# In[ ]:


from transformers import *

tokenizer = AutoTokenizer.from_pretrained(model_name)


# In[ ]:


train_input_ids, train_att_masks, train_labels = generate_np_files_for_training(train_set, tokenizer)


# In[ ]:


def input_id_maker(dataf, tokenizer):
  input_ids = []
  lengths = []

  for i in progressbar.progressbar(range(len(dataf['text']))):
    sen = dataf['text'].iloc[i]
    sen = tokenizer.tokenize(sen)#, add_prefix_space=True)
    CLS = tokenizer.cls_token
    SEP = tokenizer.sep_token
    if(len(sen) > 510):
      sen = sen[len(sen)-510:]

    sen = [CLS] + sen + [SEP]
    encoded_sent = tokenizer.convert_tokens_to_ids(sen)
    input_ids.append(encoded_sent)
    lengths.append(len(encoded_sent))

  input_ids = pad_sequences(input_ids, maxlen=512, value=0, dtype="long", truncating="post", padding="post")
  return input_ids, lengths


# In[ ]:


validation_input_ids, validation_lengths = input_id_maker(validation_set, tokenizer)


# In[ ]:


token_len = []
for i in progressbar.progressbar(range(len(train_set['text']))):
    text = train_set['text'].iloc[i]
#    print(train_set.iloc[i], text)
    token_len.append(len(tokenizer.tokenize(text)))
#    break


# In[ ]:


validation_attention_masks = att_masking(validation_input_ids)
validation_labels = validation_set['label'].to_numpy().astype('int')


# In[ ]:


train_inputs = train_input_ids
validation_inputs = validation_input_ids
train_masks = train_att_masks
validation_masks = validation_attention_masks

train_inputs = torch.tensor(train_inputs)
train_labels = torch.tensor(train_labels)
train_masks = torch.tensor(train_masks)
validation_inputs = torch.tensor(validation_inputs)
validation_labels = torch.tensor(validation_labels)
validation_masks = torch.tensor(validation_masks)


# In[ ]:


batch_size = 16
train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size = batch_size)
validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
validation_sampler = RandomSampler(validation_data)
validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size = batch_size)


# In[ ]:



device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

model = model_class.from_pretrained(model_name, num_labels=2)

model.to(device)


# In[ ]:


lr = 2e-6
max_grad_norm = 1.0
epochs = 3
num_total_steps = len(train_dataloader)*epochs
num_warmup_steps = 1000
warmup_proportion = float(num_warmup_steps) / float(num_total_steps)  # 0.1
optimizer = AdamW(model.parameters(), lr=lr, correct_bias=True)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = num_warmup_steps, num_training_steps = num_total_steps)

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

seed_val = 21


np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)


# In[ ]:


loss_values = []
train_loss_values = []
train_accuracy = []
val_loss_values = []
val_accuracy = []

# For each epoch...
for epoch_i in range(0, epochs):
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')

    t0 = time.time()
    total_loss = 0

    model.train()

    for step, batch in enumerate(train_dataloader):
        if step % 40 == 0 and not step == 0:
#             print('  Batch {:>5,}  of  {:>5,}. '.format(step, len(train_dataloader)))
            print('  Batch {:>5,}  of  {:>5,}. : loss: {:} '.format(step, len(train_dataloader), total_loss/step))


        
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        model.zero_grad()        

        outputs = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
        
        loss = outputs[0]
        total_loss += loss.item()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()

    avg_train_loss = total_loss / len(train_dataloader)            
    loss_values.append(avg_train_loss)

    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
        
    print("")
    print("Running Validation...")

    t0 = time.time()

    model.eval()

    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0

    for batch in validation_dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        
        with torch.no_grad():        
          outputs = model(b_input_ids, attention_mask=b_input_mask)
    
        logits = outputs[0]

        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        
        tmp_eval_accuracy = flat_accuracy(logits, label_ids)
        eval_accuracy += tmp_eval_accuracy

        nb_eval_steps += 1

    # Report the final accuracy for this validation run.
    print("  Accuracy: {0:.2f}".format(eval_accuracy/nb_eval_steps))

print("")
print("Training complete!")


# In[ ]:


out_path = ###give outputpath
output_dir = out_path + ###mention file name

# Create output directory if needed
if not os.path.exists(output_dir):
   os.makedirs(output_dir)

print("Saving model to %s" % output_dir)

# Save a trained model, configuration and tokenizer using `save_pretrained()`.
# They can then be reloaded using `from_pretrained()`
model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
model_to_save.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

# Good practice: save your training arguments together with the trained model
# torch.save(args, os.path.join(output_dir, 'training_args.bin'))


# In[ ]:


labels = test_set.label.to_numpy().astype(int)

input_ids, input_lengths = input_id_maker(test_set, tokenizer)
attention_masks = att_masking(input_ids)

# Convert to tensors.
prediction_inputs = torch.tensor(input_ids)
prediction_masks = torch.tensor(attention_masks)
prediction_labels = torch.tensor(labels)

# Set the batch size.  
batch_size = 16

# Create the DataLoader.
prediction_data = TensorDataset(prediction_inputs, prediction_masks, prediction_labels)
prediction_sampler = SequentialSampler(prediction_data)
prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)


# In[ ]:


# Prediction on test set

print('Predicting labels for {:,} test sentences...'.format(len(prediction_inputs)))

# Put model in evaluation mode
model.eval()

# Tracking variables 
predictions , true_labels = [], []


# Predict 
for (step, batch) in enumerate(prediction_dataloader):
  # Add batch to GPU
  batch = tuple(t.to(device) for t in batch)
  
  # Unpack the inputs from our dataloader
  b_input_ids, b_input_mask, b_labels = batch
  

  
  # Telling the model not to compute or store gradients, saving memory and 
  # speeding up prediction
  with torch.no_grad():
      # Forward pass, calculate logit predictions
      outputs = model(b_input_ids, token_type_ids=None, 
                      attention_mask=b_input_mask)

  logits = outputs[0]

  # Move logits and labels to CPU
  logits = logits.detach().cpu().numpy()
  label_ids = b_labels.to('cpu').numpy()
  
  # Store predictions and true labels
  predictions.append(logits)
  true_labels.append(label_ids)

print('    DONE.')


# In[ ]:


predictions = np.concatenate(predictions, axis=0)
true_labels = np.concatenate(true_labels, axis=0)
pred_flat = np.argmax(predictions, axis=1).flatten()
labels_flat = true_labels.flatten()

flat_accuracy(predictions,true_labels)


# In[ ]:


macro_precision, macro_recall, macro_f1, micro_precision, micro_recall, micro_f1 = metrics_calculator(pred_flat, labels_flat)
print("macro_precision", "\t", "macro_recall", "\t\t", "macro_f1", "\t\t", "accuracy")
print(macro_precision, "\t", macro_recall, "\t", macro_f1, "\t", flat_accuracy(predictions,true_labels))


# In[ ]:


len(pred_flat)


# In[ ]:


test_set['prediction'] = pred_flat


# In[ ]:


test_set


# In[ ]:


test_set= test_set.drop(columns=['text', 'split'])


# In[ ]:


test_set


# In[ ]:


test_set.to_csv(file path name, index=False)


# In[ ]:




