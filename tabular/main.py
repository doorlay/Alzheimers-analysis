import csv
import numpy
import pandas
import warnings
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import utils

warnings.filterwarnings('ignore')

longitudinal_df = pandas.read_csv("oasis_longitudinal.csv")

def activation(net):
    return 1/(1+numpy.exp(-net))


def train(X,t,nepochs=200,n=0.5,test_size=0.7,val_size=0.7,seed=0):
    X_train, X_test, t_train, t_test = train_test_split(X, t, test_size=test_size,random_state=seed)
    X_train2, X_val, t_train2, t_val = train_test_split(X_train, t_train, test_size=val_size,random_state=seed)

    train_accuracy = []
    val_accuracy = []
    nfeatures = X.shape[1]
    numpy.random.seed(seed)
    w = 2*numpy.random.uniform(size=(nfeatures,)) - 1
    
    for epoch in range(nepochs):
        y_train2 = X_train2.apply(lambda x: activation(numpy.dot(w,x)),axis=1)
        y_val = X_val.apply(lambda x: activation(numpy.dot(w,x)),axis=1)

        train_accuracy.append(sum(t_train2 == numpy.round(y_train2))/len(t_train2))
        val_accuracy.append(sum(t_val == numpy.round(y_val))/len(t_val))
                
        for j in range(len(w)):
            w[j] -= n*numpy.dot((y_train2 - t_train2)*y_train2*(1-y_train2),X_train2.iloc[:,j])
            
    results = pandas.DataFrame({"epoch": numpy.arange(nepochs)+1, 'train_accuracy':train_accuracy,'val_accuracy':val_accuracy, "n":n,'test_size':test_size,'val_size':val_size,'seed':seed}).set_index(['n','test_size','val_size','seed'])
    return w,X_test,t_test,results


def predict(w,X,threshold=0.5):
    y = []
    for i in range(len(X)):
        x = X.iloc[i].values
        if activation(numpy.sum(x*w)) > threshold:
            y.append(1)
        else:
            y.append(0)
    return pandas.Series(y)

def calculate_accuracy(x_train, y_train, x_test, y_test):
    correct = 0
    for i in range(len(x_test)):
        if predict(x_train, y_train, x_test.iloc[i]) == y_test.iloc[i]:
            correct += 1
    print(correct / len(x_test))
    return correct / len(x_test)

def calculate_permutation_feature_importance(Xtrain,ytrain,Xtest,ytest, npermutations = 20):
    importances = {}
    for col in Xtrain.columns:
        importances[col] = 0
    original_accuracy = calculate_accuracy(Xtrain,ytrain,Xtest,ytest)
    for col in Xtrain.columns:
        permutation_accuracy = 0
        for perm in range(npermutations):
            Xtrain2 = Xtrain.copy()
            Xtrain2[col] = Xtrain[col].sample(frac=1, replace=False).values
            permutation_accuracy += calculate_accuracy(Xtrain2, ytrain, Xtest, ytest)
        importances[col] = original_accuracy - permutation_accuracy / npermutations
    return importances


def evaluation(cm,positive_class=1):
    stats = {}
    negative_class = abs(positive_class - 1)
    stats["specificity"] = cm.loc[negative_class,negative_class] / sum(cm.loc[negative_class])
    stats["sensitivity/recall"] = cm.loc[positive_class,positive_class] / sum(cm.loc[positive_class])
    stats["precision"] = cm.loc[positive_class,positive_class] / sum(cm[positive_class])
    stats["accuracy"] = (cm.loc[negative_class,negative_class] + cm.loc[positive_class,positive_class]) / cm.to_numpy().sum()
    stats["F1"] = 2 * (stats["precision"] * stats["sensitivity/recall"]) / (stats["precision"] + stats["sensitivity/recall"])
    return stats


# Preprocessing on the CSV data
longitudinal_df = pandas.get_dummies(longitudinal_df.drop(labels=["Hand", "Visit", "MR Delay", "ASF", "MRI ID", "Subject ID"], axis=1).dropna())
X = longitudinal_df.drop(labels=['Group_Nondemented', 'Group_Demented', 'Group_Converted'], axis=1)
y = longitudinal_df['Group_Demented']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Scale data
ss = StandardScaler()
X_train_scaled = ss.fit_transform(X_train)
X_test_scaled = ss.transform(X_test)

# Convert y_train values to categorical values
y_train = LabelEncoder().fit_transform(y_train)

# Perform Sklearn's Logistic Regression to obtain feature importance
model = LogisticRegression()
model.fit(X_train_scaled, y_train)
importances = pandas.DataFrame(data={
    'Attribute': X_train.columns,
    'Importance': model.coef_[0]
})
importances = importances.sort_values(by='Importance', ascending=False)

# Use score method to get accuracy of model
score = model.score(X_test, y_test)
print(score)



# Display sklearn feature importances in a bar graph
plt.bar(x=importances['Attribute'], height=importances['Importance'], color='#087E8B')
plt.title('Feature importances, sklearn version', size=20)
plt.xticks(rotation='vertical')
plt.show()

# Perform Logistic Regression from scratch to obtain feature importance
seeds = [1,2,3,4,5]
importances = pandas.Series(numpy.zeros((X.shape[1],)),index=X.columns)
for seed in seeds:
    w,X_test,t_test,results = train(X,y,seed=seed)
    print(results["train_accuracy"])
    max_weight = max(w, key=abs)
    for i in range(len(w)):
        importances.loc[importances.index[i]] += abs(w[i]) / abs(max_weight)
importances = importances / len(seeds)
importances = importances.sort_values(ascending=False)

# Display from scratch feature importances in a bar graph
plt.bar(height=importances, x=X_train.columns, color='#087E8B')
plt.title('Feature importances, from scratch version', size=20)
plt.xticks(rotation='vertical')
plt.show()