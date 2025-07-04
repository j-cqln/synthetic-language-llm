import pandas as pd
import matplotlib.pyplot as plt

real_1_data = pd.read_csv("part-1-real-train.csv")
fake_1_data = pd.read_csv("part-1-fake-train.csv")
real_2_data = pd.read_csv("part-2-real-train.csv")
fake_2_data = pd.read_csv("part-2-fake-train.csv")

plt.plot(real_1_data["time_step"],real_1_data["acc"])
plt.plot(fake_1_data["time_step"],fake_1_data["acc"])
plt.plot(real_2_data["time_step"],real_2_data["acc"])
plt.plot(fake_2_data["time_step"],fake_2_data["acc"])
plt.xlabel("Num Examples")
plt.ylabel("Accuracy")
plt.title("Accuracy of Models")
plt.legend(["Real 1","Fake 1","Real 2","Fake 2"])
plt.xticks()
plt.show()

plt.plot(real_1_data["time_step"],real_1_data["loss"])
plt.plot(fake_1_data["time_step"],fake_1_data["loss"])
plt.plot(real_2_data["time_step"],real_2_data["loss"])
plt.plot(fake_2_data["time_step"],fake_2_data["loss"])
plt.xlabel("Num Examples")
plt.ylabel("Loss")
plt.title("Loss of Models")
plt.legend(["Real 1","Fake 1","Real 2","Fake 2"])
plt.xticks()
plt.show()