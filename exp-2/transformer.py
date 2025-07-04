#from https://www.kaggle.com/code/changyeop/how-to-fine-tune-gpt-2-for-beginners

import torch
import math
from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import Trainer, TrainingArguments
from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel, GPT2TokenizerFast, GPT2Tokenizer
import pandas as pd
import evaluate
import numpy as np

model_name = 'gpt2'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

def load_dataset(file_path, tokenizer, block_size = 128):
    dataset = TextDataset(
        tokenizer = tokenizer,
        file_path = file_path,
        block_size = block_size,
    )
    return dataset

def load_data_collator(tokenizer, mlm = False):
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=mlm,
    )
    return data_collator

def train(train_file_path,
          eval_file_path,
          model_name,
          output_dir,
          overwrite_output_dir,
          per_device_train_batch_size,
          num_train_epochs,
          save_steps,
          order):
  tokenizer = GPT2Tokenizer.from_pretrained(model_name)
  train_dataset = load_dataset(train_file_path, tokenizer)
  eval_dataset = load_dataset(eval_file_path, tokenizer)
  data_collator = load_data_collator(tokenizer)
  tokenizer.save_pretrained(output_dir)
      
  model = GPT2LMHeadModel.from_pretrained(model_name)

  model.save_pretrained(output_dir)

  training_args = TrainingArguments(
          output_dir=output_dir,
          overwrite_output_dir=overwrite_output_dir,
          per_device_train_batch_size=per_device_train_batch_size,
          num_train_epochs=num_train_epochs,
          logging_dir = "./logs",
          logging_steps = 10,
          use_cpu = False,
          learning_rate =1e-2
      )

  trainer = Trainer(
          model=model,
          args=training_args,
          data_collator=data_collator,
          train_dataset=train_dataset,
          eval_dataset=eval_dataset,
  )
      
  trainer.train()
  trainer.save_model()
  #https://gist.github.com/gmihaila/3e730aebff70b3f3798175026ae516bd
  eval_dataset = "./data/deps_dev" + order + ".txt"
  eval_metrics = trainer.evaluate()
  df = pd.DataFrame(trainer.state.log_history)
  fn = "./train_logs/train_log_" + order + "_epochs_" + str(num_train_epochs) + ".csv"
  df.to_csv(fn)
  fn2 = "./eval_logs/eval_log_" + order + "_epochs_" + str(num_train_epochs) + ".csv"
  df2 = pd.DataFrame([eval_metrics])
  df2.to_csv(fn2)



orders = ["svo", "sov", "vso", "vos", "osv", "ovs"]

for order in orders:
    train_file_path = "./data/deps_train_" + order + ".txt"
    model_name = 'gpt2'
    overwrite_output_dir = True
    per_device_train_batch_size = 8
    num_train_epochs = 2.0
    save_steps = 500
    output_dir = "./output/output_" + order + "_" + "_epochs_" + str(num_train_epochs)
    eval_file_path = "./data/deps_dev_" + order + ".txt"

    train(
        train_file_path=train_file_path,
        eval_file_path=eval_file_path,
        model_name=model_name,
        output_dir=output_dir,
        overwrite_output_dir=overwrite_output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        num_train_epochs=num_train_epochs,
        save_steps=save_steps,
        order=order
    )
