import pandas as pd

part1train = pd.read_csv("neg-construction-1.train",names=['sentence','a','b'])
part1test = pd.read_csv("neg-construction-1.test",names=['sentence','a','b'])
part1valid = pd.read_csv("neg-construction-1.valid",names=['sentence','a','b'])

part2train = pd.read_csv("neg-construction-2.train",names=['sentence','a','b','c'])
part2test = pd.read_csv("neg-construction-2.test",names=['sentence','a','b','c'])
part2valid = pd.read_csv("neg-construction-2.valid",names=['sentence','a','b','c'])

def is_bad(row):
    if len(row["sentence"].split(' ')) > 30:
        return 1
    else:
        return 0

part1train["bad"] = part1train.apply(is_bad,axis=1)
part1test["bad"] = part1test.apply(is_bad,axis=1)
part1valid["bad"] = part1valid.apply(is_bad,axis=1)

part2train["bad"] = part2train.apply(is_bad,axis=1)
part2test["bad"] = part2test.apply(is_bad,axis=1)
part2valid["bad"] = part2valid.apply(is_bad,axis=1)

print("Part 1:")
print(f"Training: {part1train['bad'].sum()}")
print(f"Testing: {part1test['bad'].sum()}")
print(f"Validation: {part1valid['bad'].sum()}")
print()
print("Part 2:")
print(f"Training: {part2train['bad'].sum()}")
print(f"Testing: {part2test['bad'].sum()}")
print(f"Validation: {part2valid['bad'].sum()}")
print()
print(f"Total: {part1train['bad'].sum()+part1test['bad'].sum()+part1valid['bad'].sum()+part2train['bad'].sum()+part2test['bad'].sum()+part2valid['bad'].sum()}")
