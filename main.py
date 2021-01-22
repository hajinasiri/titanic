import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
train_data = pd.read_csv("train.csv")
#print(train_data)
test_data = pd.read_csv("test.csv")

y = train_data["Survived"]
#sns.heatmap(train_data.isnull(),yticklabels=False,cbar=False,cmap='viridis')
#plt.show()
#sns.countplot(x="Survived",data=train_data)
#plt.show()
class1 = 38
class2 = 30
class3 = 25

def age_correction(input):
    Age= input[0]
    Class = input[1]
    if pd.isnull(Age):
        if Class == 1:
            return 38
        elif Class == 2:
            return 30
        elif Class == 3:
            return 25
    else:
       return Age
train_data["Age"] = train_data[["Age", "Pclass"]].apply(age_correction, axis=1)
