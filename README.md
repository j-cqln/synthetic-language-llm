# Analyzing synthetic language learnability of transformers

[William](https://github.com/gaussianprime), [Jacqueline](https://github.com/j-cqln), [Toby](https://github.com/Mr-Sbev)

## Requirements
See `requirements.txt`.

## Experiment 1
Run `exp-1/exp-1.py` for testing transformers' ability to negate sentences in a 'real' and a 'fake' way for two different sentence constructions.

Construction 1 has one clause, while construction 2 can have multiple. Real negations resemble English, where the main verb is negated, while the fake negations are count-based, adding a fake negating word at the fourth position.

### Sample usage
```
# Train model for construction 1, fake negation
python exp-1.py --construction=1 --negation-type=fake --train-test=train

# Test model for construction 2, real negation
# Requires running the same command with --train-test=train first
python exp-1.py --construction=2 --negation-type=real --train-test=test
```

## Experiment 2
Download the datasets and place it in a `./data`  folder. The hyperparameters in `transformer.py` such as number of epochs and learning rate can be modified near the bottom of the code. Make folders called `./output`, `train_logs`, and `eval_logs`. Note that the expected size of output is 20 gigabytes for each run. Run `transformer.py`. Then, run `plots.py`.

### Warning
Training is a time and resource consuming process, and the `./output` folder grows to around 20 GB per training run.
