import pandas as pd
import matplotlib.pyplot as plt

test_data = pd.read_csv("test_data.csv")

plt.bar(test_data["model"],test_data["acc"])
plt.xlabel("Construction Rule")
plt.ylabel("Accuracy")
plt.title("Test Accuracy of Models")
plt.xticks()
plt.show()

plt.bar(test_data["model"],test_data["loss"])
plt.xlabel("Construction Rule")
plt.ylabel("Loss")
plt.title("Test Loss of Models")
plt.xticks()
plt.show()

plt.bar(test_data["model"],test_data["perplexity"])
plt.xlabel("Construction Rule")
plt.ylabel("Perplexity")
plt.title("Test Perplexity of Models")
plt.xticks()
plt.show()