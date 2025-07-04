import torch
import torch.nn as nn

def file_to_dataset(file_name, output_index):
    dataset = []

    fi = open(file_name, "r")
    for line in fi:
        parts = line.strip().split("\t")
        input_sentence = parts[0]
        output_sentence = parts[output_index]

        dataset.append([input_sentence, output_sentence])

    return dataset

def read_vocab(file_name):
    word2index = {}
    index2word = {}

    with open(file_name, "r") as f:
        for index, line in enumerate(f):
            word = line.strip()
            word2index[word] = index
            index2word[index] = word
    
    return word2index, index2word

def remove_excess_spaces(sentence):
    if "  " in sentence:
        return remove_excess_spaces(sentence.replace("  ", " "))
    else:
        return sentence

def truncate_at_eos(sentence):
    words = sentence.split()
    new_words = []

    for word in words:
        if word == "EOS":
            break
        elif word == "SOS":
            continue
        else:
            new_words.append(word)

    return " ".join(new_words)

def standardize_sentence(sentence):
    sentence = remove_excess_spaces(sentence)
    sentence = truncate_at_eos(sentence)
    return sentence

def compute_loss(output_vectors, target_sentence, word2index):
    loss_function = nn.CrossEntropyLoss()

    target_words = standardize_sentence(target_sentence).split() + ["EOS"]

    total_loss = 0
    count_losses = 0
    for output_vector, target_word in zip(output_vectors, target_words):
        target_index = word2index[target_word]

        loss = loss_function(output_vector, torch.tensor([target_index]))

        total_loss += loss
        count_losses += 1

    return total_loss / count_losses

def compute_loss_on_dataset(model, dataset, word2index, provide_target=True, verbose = False):
    total_loss = 0
    total_correct = 0
    count_sentences = 0
    length = len(dataset)
    count = 0
    for example in dataset:
        count += 1
        input_sentence = example[0]
        target_sentence = example[1]
        # verbose output:
        if verbose:
            if count % 10 == 0:
                num = int(count/length * 25)
                print(f"Progress: [{'#'*num}{' '*(25-num)}] {count}/{length}",end="\r")
        if provide_target:
            output_vectors, output_sentence = model(input_sentence, target_sentence=target_sentence)
        else:
            output_vectors, output_sentence = model(input_sentence)

        loss = compute_loss(output_vectors, target_sentence, word2index)

        total_loss += loss

        if standardize_sentence(output_sentence) == standardize_sentence(target_sentence):
            total_correct += 1

        count_sentences += 1

    average_loss = total_loss / count_sentences
    accuracy = total_correct / count_sentences
    perplexity = torch.exp(average_loss).item()

    return average_loss, accuracy, perplexity

def print_n_examples(model, dataset, n, provide_target=True):
    # Prints n examples of the output predicted by the model, printed
    # next to the correct output
    print("EXAMPLE MODEL OUTPUTS")
    for example in dataset[:n]:
        input_sentence = example[0]
        target_sentence = example[1]

        if provide_target:
            output_vectors, output_sentence = model(input_sentence, target_sentence=target_sentence)
        else:
            output_vectors, output_sentence = model(input_sentence)

        print("CORRECT:", standardize_sentence(target_sentence))
        print("OUTPUT: ", standardize_sentence(output_sentence))
        print("")

def first_word_accuracy(model, dataset, provide_target=True):
    total_correct = 0
    count_examples = 0

    for example in dataset:
        input_sentence = example[0]
        target_sentence = example[1]

        if provide_target:
            output_vectors, output_sentence = model(input_sentence, target_sentence=target_sentence)
        else:
            output_vectors, output_sentence = model(input_sentence)

        correct_first_word = standardize_sentence(target_sentence).split()[0]
        predicted_first_word = standardize_sentence(output_sentence).split()[0]

        if predicted_first_word == correct_first_word:
            total_correct += 1
        count_examples += 1

    return total_correct / count_examples