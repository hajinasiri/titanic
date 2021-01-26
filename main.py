import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
def age_correction(input):
    #This function gets Age and Pclass as element of input and if Age is null
    # replaces them with the average
    #age of their class
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

def children(num):
    #this function determins if a passenger is a child
    Age = num[0]
    male = num[1]
    if int(Age)<16:
        return -1
    else:
        return int(male)
test_data = pd.read_csv("test.csv")
train_data = pd.read_csv("train.csv")
#sns.heatmap(train_data.isnull(),yticklabels=False,cbar=False,cmap='viridis')
#plt.show()
#sns.countplot(x="Survived",data=train_data)
#plt.show()


class1 = 38
class2 = 30
class3 = 25

train_data["Age"] = train_data[["Age", "Pclass"]].apply(age_correction, axis=1)
train_data.drop('Cabin',axis=1,inplace=True)
train_data.dropna(inplace=True)
#sns.heatmap(train_data.isnull(),yticklabels=False,cbar=False,cmap='viridis')
#plt.show()
Mclass = pd.get_dummies(train_data['Pclass'],drop_first=True)
sex = pd.get_dummies(train_data['Sex'],drop_first=True)
#####embark = pd.get_dummies(train_data['Embarked'],drop_first=True)
#train_data = pd.concat([train_data,sex,embark,Mclass],axis=1)
train_data = pd.concat([train_data,sex,Mclass],axis=1)
train_data.drop(['Sex','Embarked','Name','Ticket','Pclass'],axis=1,inplace=True)
train_data.drop(['PassengerId'],axis=1,inplace=True)
train_data['male'] = train_data[['Age','male']].apply(children,axis=1)
y = train_data["Survived"]
X = train_data.drop('Survived',axis=1)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(C=1,max_iter=1000)
lr.fit(X_train,y_train)
prediction = lr.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(y_test,prediction))
#from sklearn.ensemble import RandomForestClassifier
#random_forest = RandomForestClassifier(n_estimators=50)
#random_forest.fit(X_train, y_train)
#y_pred = random_forest.predict(X_test)
#print(classification_report(y_test,y_pred))

#from sklearn.model_selection import GridSearchCV
#grid={"C":np.logspace(-3,3,7), "penalty":["l2"]}# l1 lasso l2 ridge
#logreg_cv=GridSearchCV(lr,grid,cv=10)
#logreg_cv.fit(X_train,y_train)
#print("tuned hpyerparameters :(best parameters) ",logreg_cv.best_params_)
#print("accuracy :",logreg_cv.best_score_)


#training LR with the whole training data
lr = lr.fit(X,y)
test_data["Age"] = test_data[["Age", "Pclass"]].apply(age_correction, axis=1)
test_data.drop('Cabin',axis=1,inplace=True)
test_data = test_data.fillna(test_data.mean())
#sns.heatmap(train_data.isnull(),yticklabels=False,cbar=False,cmap='viridis')
#plt.show()
Mclass = pd.get_dummies(test_data['Pclass'],drop_first=True)
sex = pd.get_dummies(test_data['Sex'],drop_first=True)
#embark = pd.get_dummies(test_data['Embarked'],drop_first=True)
#test_data = pd.concat([test_data,sex,embark,Mclass],axis=1)
test_data = pd.concat([test_data,sex,Mclass],axis=1)
test_data.drop(['Sex','Embarked','Name','Ticket','Pclass'],axis=1,inplace=True)
test_data.drop(['PassengerId'],axis=1,inplace=True)
test_data['male'] = test_data[['Age','male']].apply(children,axis=1)
data = test_data
result = lr.predict(data)
result = pd.DataFrame(result,index=range(892, 1310),columns=['Survived'])
result.index.names = ['PassengerId']
result.to_csv('./result.csv')