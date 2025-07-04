# Similar to experiment 1 of  Kallini et al. paper
import math
import random
import argparse

import torch
import torch.nn as nn

import numpy as np
import pandas as pd

from utils import *

# From official PyTorch implementation
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x, sentence):
        return self.pe[:x.size(0), :]

# Seq2seq sentence transformer
class EncoderTransformer(nn.Module):
    def __init__(self, vocab_size, hidden_size, word2index):
        super(EncoderTransformer, self).__init__()

        self.n_layers = 4
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.n_head = 4
        self.dim_feedforward = 4*self.hidden_size
        self.word2index = word2index

        self.embedding = nn.Embedding(self.vocab_size, self.hidden_size)

        self.positional_encoder = PositionalEncoding(self.hidden_size)

        encoder_layer = nn.TransformerEncoderLayer(d_model=self.hidden_size, nhead=self.n_head, dim_feedforward=self.dim_feedforward, dropout=0)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.n_layers)

    def forward(self, sentence):
        words = sentence.split()

        input_seq = torch.LongTensor([[self.word2index[word] for word in words]]).transpose(0,1)
        emb = self.embedding(input_seq)

        emb = emb + self.positional_encoder(emb, sentence)

        memory = self.transformer_encoder(emb)

        return memory

class DecoderTransformer(nn.Module):
    def __init__(self, vocab_size, hidden_size, max_length=30, word2index=None, index2word=None):
        super(DecoderTransformer, self).__init__()

        self.n_layers = 4
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.n_head = 4
        self.dim_feedforward = 4*self.hidden_size
        self.max_length = max_length
        self.word2index = word2index
        self.index2word = index2word

        self.embedding = nn.Embedding(self.vocab_size, self.hidden_size)

        self.positional_encoder = PositionalEncoding(self.hidden_size)

        decoder_layer = nn.TransformerDecoderLayer(d_model=self.hidden_size, nhead=self.n_head, dim_feedforward=self.dim_feedforward)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=self.n_layers)

        self.out = nn.Linear(self.hidden_size, self.vocab_size)

    def forward(self, memory, target_output=None):
        if target_output is not None:
            input_words = ["SOS"] + target_output.split() 
            input_tensor = torch.tensor([[self.word2index[word] for word in input_words]]).transpose(0,1)

            emb = self.embedding(input_tensor)
            positional_emb = self.positional_encoder(emb, target_output)
            emb = emb + positional_emb

            tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(emb.shape[0])
            output = self.transformer_decoder(emb, memory, tgt_mask=tgt_mask)
            output = self.out(output).squeeze(1)

            words = []
            output_vectors = []
            for output_vector in output:
                topv, topi = torch.topk(output_vector, 1)
                word = self.index2word[topi.item()]
                words.append(word)
                output_vectors.append(output_vector.unsqueeze(0))

        else:
            input_words = ["SOS"]
            output_vectors = []
            done = False
            while not done:
                input_tensor = torch.tensor([[self.word2index[word] for word in input_words]]).transpose(0,1)

                emb = self.embedding(input_tensor)
                emb = emb + self.positional_encoder(emb, None)

                tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(emb.shape[0])
                output = self.transformer_decoder(emb, memory, tgt_mask=tgt_mask)
                output = self.out(output).squeeze(1)

                final_output = output[-1]
                topv, topi = torch.topk(final_output, 1)
                word = self.index2word[topi.item()]
                input_words.append(word)
                output_vectors.append(final_output.unsqueeze(0))

                if word == "EOS" or len(input_words) > self.max_length:
                    done = True

            words = input_words[1:]

        return output_vectors, " ".join(words)

class Seq2SeqTransformer(nn.Module):
    def __init__(self, vocab_size, hidden_size, max_length=30, word2index=None, index2word=None):
        super(Seq2SeqTransformer, self).__init__()
        self.encoder = EncoderTransformer(vocab_size, hidden_size, word2index=word2index)
        self.decoder = DecoderTransformer(vocab_size, hidden_size, max_length=max_length, word2index=word2index, index2word=index2word)

    def forward(self, input_sentence, target_sentence=None):
        encoding = self.encoder(input_sentence)
        output_vectors, output_sentence = self.decoder(encoding, target_output=target_sentence)
        return output_vectors, output_sentence

# Train a model on a training set, and save its weights
def train_model(model, training_set, validation_set, model_name, word2index, lr=None):
    
    tr_metrics = pd.DataFrame(columns=["time_step","acc","loss","perplexity"])
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_loss = math.inf
    for index, example in enumerate(training_set):
        if index % 500 == 0:
            print(f"> {index} <")
            loss, accuracy, perplexity = compute_loss_on_dataset(model, validation_set, word2index)
            print(index, "Acc:", accuracy, "Loss:", loss.item(), "Perplexity:", perplexity)
            
            tr_metrics.loc[len(tr_metrics)] = {"time_step":index,"acc":accuracy,"loss":loss.item(),"perplexity":perplexity}

            if loss < best_loss:
                torch.save(model.state_dict(), model_name + ".weights")
                best_loss = loss

        input_sentence = example[0]
        target_sentence = example[1]

        output_vectors, output_sentence = model(input_sentence, target_sentence=target_sentence)

        loss = compute_loss(output_vectors, target_sentence, word2index)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
    return tr_metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyzing synthetic languages learnability of transformers, experiment 1")

    # Required parameters
    parser.add_argument(
        "--construction",
        default=1,
        type=int,
        required=True,
    )

    parser.add_argument(
        "--negation-type",
        default="real",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--train-test",
        default="train",
        type=str,
        required=True,
    )

    args = parser.parse_args()

    try:
        test_data = pd.read_csv("output/test_data.csv",index_col=0)
    except:
        test_data = pd.DataFrame(columns=["model","acc","loss","perplexity"])

    if args.construction == 1:
        # Construction 1 vocab
        word2index1, index2word1 = read_vocab("vocab-1.txt")

        if args.negation_type == "real":
            # Construction 1 real
            model = Seq2SeqTransformer(len(word2index1), hidden_size=128, word2index=word2index1, index2word=index2word1)

            if args.train_test == "train":
                lr = 0.00005
                training_set = file_to_dataset("data/negation-1.train", 1)
                validation_set = file_to_dataset("data/negation-1.dev", 1)
                validation_set = validation_set[:500]
                metrics = train_model(model, training_set, validation_set, "weights/transformer-1-real", word2index1, lr=lr)
                metrics.to_csv("output/part-1-real-train.csv")
            
            if args.train_test == "test":
                dataset = file_to_dataset("data/negation-1.test", 1)[:500]
                model.load_state_dict(torch.load("weights/transformer-1-real.weights"))
                print_n_examples(model, dataset, 10)
                loss, accuracy, perplexity = compute_loss_on_dataset(model, dataset, word2index1, provide_target=False,verbose=True)
                test_data.loc[len(test_data)] = {"model":"1-R","acc":accuracy,"loss":loss.item(),"perplexity":perplexity}
                print("FULL-SENTENCE ACCURACY:", accuracy)
        
        if args.negation_type == "fake":
            # Construction 1 fake
            model = Seq2SeqTransformer(len(word2index1), hidden_size=128, word2index=word2index1, index2word=index2word1)

            if args.train_test == "train":
                lr = 0.00005
                training_set = file_to_dataset("data/negation-1.train", 2)
                validation_set = file_to_dataset("data/negation-1.dev", 2)
                validation_set = validation_set[:500]
                metrics = train_model(model, training_set, validation_set, "weights/transformer-1-fake", word2index1, lr=lr)
                metrics.to_csv("output/part-1-fake-train.csv")
            
            if args.train_test == "test":
                dataset = file_to_dataset("data/negation-1.test", 2)[:500]
                model.load_state_dict(torch.load("weights/transformer-1-fake.weights"))
                provide_target = False
                print_n_examples(model, dataset, 10)
                loss, accuracy, perplexity = compute_loss_on_dataset(model, dataset, word2index1, provide_target=False,verbose=True)
                test_data.loc[len(test_data)] = {"model":"1-F","acc":accuracy,"loss":loss.item(),"perplexity":perplexity}
                print("FULL-SENTENCE ACCURACY:", accuracy)

    if args.construction == 2:
        # Construction 2 vocab
        word2index2, index2word2 = read_vocab("vocab-2.txt")
        
        if args.negation_type == "real":
            # Construction 2 real
            model = Seq2SeqTransformer(len(word2index2), hidden_size=128, word2index=word2index2, index2word=index2word2)
            
            if args.train_test == "train":
                lr = 0.00005
                training_set = file_to_dataset("data/negation-2.train", 1)
                validation_set = file_to_dataset("data/negation-2.dev", 1)
                validation_set = validation_set[:500]
                metrics = train_model(model, training_set, validation_set, "weights/transformer-2-real", word2index2, lr=lr)
                metrics.to_csv("output/part-2-real-train.csv")
            
            if args.train_test == "test":
                dataset = file_to_dataset("data/negation-2.test", 1)[:500]
                model.load_state_dict(torch.load("weights/transformer-2-real.weights"))
                print_n_examples(model, dataset, 10)
                loss, accuracy, perplexity = compute_loss_on_dataset(model, dataset, word2index2, provide_target=False,verbose=True)
                test_data.loc[len(test_data)] = {"model":"2-R","acc":accuracy,"loss":loss.item(),"perplexity":perplexity}
                print("FULL-SENTENCE ACCURACY:", accuracy)
        
        if args.negation_type == "fake":
            # Construction 2 fake
            model = Seq2SeqTransformer(len(word2index2), hidden_size=128, word2index=word2index2, index2word=index2word2)
            
            if args.train_test == "train":
                lr = 0.00005
                training_set = file_to_dataset("data/negation-2.train", 2)
                validation_set = file_to_dataset("data/negation-2.dev", 2)
                validation_set = validation_set[:500]
                metrics = train_model(model, training_set, validation_set, "weights/transformer-2-fake", word2index2, lr=lr)
                metrics.to_csv("output/part-2-fake-train.csv")
            
            if args.train_test == "test":
                dataset = file_to_dataset("data/negation-2.test", 2)[:500]
                model.load_state_dict(torch.load("weights/transformer-2-fake.weights"))
                provide_target = False
                print_n_examples(model, dataset, 10)
                loss, accuracy, perplexity = compute_loss_on_dataset(model, dataset, word2index2, provide_target=False,verbose=True)
                test_data.loc[len(test_data)] = {"model":"2-F","acc":accuracy,"loss":loss.item(),"perplexity":perplexity}
                print("FULL-SENTENCE ACCURACY:", accuracy)
        
    test_data.to_csv("output/test_data.csv")