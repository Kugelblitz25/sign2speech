import json
import matplotlib.pyplot as plt
import pandas as pd

models =['i3d','x3d']

for model in models:
    train_acc = []
    validation_acc = []
    freeze = []

    for k in range(1,6):
        data = {}
        with open(f"experiments/freeze_{model}_{k}.json","r") as file:
            data = json.load(file)
        freeze = list(data.keys())
    
        for i, key in enumerate(freeze, start=0):
            if k!=1:
                train_acc[i]+= data[key][0]
                validation_acc[i]+= data[key][1]
            else:
                train_acc.append(data[key][0])
                validation_acc.append(data[key][1])

    train_acc = [round(i/5,2) for i in train_acc]
    validation_acc = [round(i/5,2) for i in validation_acc]

    plt.title(f"{model} Accuracy Vs Layers frozen ")
    plt.plot(freeze, train_acc, label = "Train Accuracy",  marker='o', color="blue")
    plt.plot(freeze, validation_acc, label = "Validation Accuracy", marker='o', color="orange")
    plt.xlabel("Frozen Layers")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(f"experiments/freeze_{model}.png", dpi=300, bbox_inches='tight')
    plt.close()

    d = {"layers_frozen": freeze, "train_acc": train_acc, "validation_acc": validation_acc}
    df = pd.DataFrame(d)
    df.to_csv(f"experiments/{model}_freeze.csv", index = False)