def process(file_name, new_file_name):
    lines = []

    # Read in data
    with open(file_name) as f:
        for line in f:
            new_line = line.strip().split(",")
            new_line = new_line[1:]

            for sentence in new_line:
                if len(sentence) > 1:
                    for word in sentence.split():
                        vocab.add(word)
            
            lines.append(new_line)

    # Write to file
    with open(new_file_name, "w") as f:
        for line in lines:
            f.write("\t".join(line) + "\n")

# Construction 1
vocab = set()
vocab.add("SOS")
vocab.add("EOS")

process("data/neg-construction-1.train", "data/negation-1.train")
process("data/neg-construction-1.test", "data/negation-1.test")
process("data/neg-construction-1.valid", "data/negation-1.dev")

with open("vocab-1.txt", "w") as f:
    for word in vocab:
        f.write(word + "\n")

# Construction 2
vocab = set()
vocab.add("SOS")
vocab.add("EOS")

process("data/neg-construction-2.train", "data/negation-2.train")
process("data/neg-construction-2.test", "data/negation-2.test")
process("data/neg-construction-2.valid", "data/negation-2.dev")

with open("vocab-2.txt", "w") as f:
    for word in vocab:
        f.write(word + "\n")