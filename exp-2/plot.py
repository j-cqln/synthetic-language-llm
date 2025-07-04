import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math

def plot_indiv(epochs):
    orders = ["svo", "sov", "vso", "vos", "osv", "ovs"]
    for order in orders:
        path = "./train_logs/train_logs_epochs_" + str(epochs) + "/train_log_" + order + "_epochs_" + str(epochs) + ".csv"
        df = pd.read_csv(path)
        plt.plot(range(0,len(df["loss"])), df["loss"])
        plt.ylabel("Loss")
        plt.xlabel("Epochs (0.01)")
        plt.title(order + " training")
        fig_path = "./plots/epochs_" + str(epochs) + "/loss_" + order + "_" + str(epochs) + ".png"
        plt.savefig(fig_path, transparent=False, dpi=80, bbox_inches="tight")
        plt.close()

        df.insert(0, "perplexity", [0]*len(df["loss"]))
        for i in range(0,len(df["loss"])):
            df["perplexity"][i] = math.exp(df["loss"][i])
        plt.plot(range(0,len(df["perplexity"])), df["perplexity"])
        plt.ylabel("Perplexity")
        plt.xlabel("Epochs (0.01)")
        plt.title(order + " training")
        fig_path = "./plots/epochs_" + str(epochs) + "/perplexity_" + order + "_" + str(epochs) + ".png"
        plt.savefig(fig_path, transparent=False, dpi=80, bbox_inches="tight")
        plt.close()

        plt.plot(range(0,len(df["loss"])), df["loss"])
        plt.yscale("log")
        plt.ylabel("Log Loss")
        plt.xlabel("Epochs (0.01)")
        plt.title(order + " training")
        fig_path = "./plots/epochs_" + str(epochs) + "/log_loss_" + order + "_" + str(epochs) + ".png"
        plt.savefig(fig_path, transparent=False, dpi=80, bbox_inches="tight")
        plt.close()

def plot_total(epochs):
    orders = ["svo", "sov", "vso", "vos", "osv", "ovs"]
    for order in orders:
        path = "./train_logs/train_logs_epochs_" + str(epochs) + "/train_log_" + order + "_epochs_" + str(epochs) + ".csv"
        df = pd.read_csv(path)
        plt.plot(range(0,len(df["loss"])), df["loss"], label = order)
        plt.ylabel("Loss")
        plt.xlabel("Epochs (0.01)")
    plt.legend()
    plt.title("Total training")
    total_fig_path = "./plots/epochs_" +str(epochs) + "/loss_total_" + str(epochs) + ".png"
    plt.savefig(total_fig_path, transparent=False, dpi=80, bbox_inches="tight")
    plt.close()
    
    for order in orders:
        path = "./train_logs/train_logs_epochs_" + str(epochs) + "/train_log_" + order + "_epochs_" + str(epochs) + ".csv"
        df = pd.read_csv(path)
        plt.plot(range(0,len(df["loss"])), df["loss"], label = order)
        plt.ylabel("Log Loss")
        plt.yscale("log")
        plt.xlabel("Epochs (0.01)")
    plt.legend()
    plt.title("Total training")
    total_fig_path = "./plots/epochs_" + str(epochs) + "/log_loss_total_" + str(epochs) + ".png"
    plt.savefig(total_fig_path, transparent=False, dpi=80, bbox_inches="tight")
    plt.close()

def plot_eval(epochs):
    orders = ["svo", "sov", "vso", "vos", "osv", "ovs"]
    heights = []
    for order in orders:
        path = "./eval_logs/eval_logs_epochs_" + str(epochs) + "/eval_log_" + order + "_epochs_" + str(epochs) + ".csv"
        df = pd.read_csv(path)
        heights.append(df["eval_loss"][0])
    plt.bar(x=orders,height=heights)
    plt.ylabel("Loss")
    fig_path = "./plots/epochs_" + str(epochs) + "/eval_loss_" + str(epochs) + ".png"
    plt.savefig(fig_path, transparent=False, dpi=80, bbox_inches="tight")
    plt.close()

epochs = [0.1, 1.0, 2.0]
for epoch in epochs:
    plot_indiv(epoch)
    plot_total(epoch)
    plot_eval(epoch)