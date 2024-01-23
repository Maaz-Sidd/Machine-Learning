import pandas as pd
import numpy as np
import random 
from collections import Counter
import statistics
import re
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier


csv_file_path = 'ecoli.data'

def get_score (model, X_train, Y_train, X_test, Y_test):
    mymodel = model.fit(X_train, Y_train)
    return cross_val_score(mymodel, X_train, Y_train, cv=5)

def main():

    # Initialize lists for the 80% and 20% splits
    X_train1 = []
    X_test1 = []

    # Initialize lists to store the last column of each split
    Y_train1= []
    Y_test1 = []

    data = []

    pattern = re.compile(r'\s+')

    # Read the file and split the data into rows
    with open(csv_file_path, "r") as file:
        lines = file.readlines()

    # Shuffle the lines randomly
    random.shuffle(lines)

    # Calculate the split points for 80% and 20%
    split_point = int(0.9 * len(lines))

    # Iterate through the shuffled lines
    for i, line in enumerate(lines):
        # Strip the newline character and split the line into elements
        elements = re.split(pattern, line.strip())
        data.append(elements)
        # Append the line to the appropriate split list
    data = pd.DataFrame(data)
    category_mapping = {}
    for col in range(0,9,8):
        categories = data[col].unique() if col == 0 else sorted(data[col].unique())
        category_mapping[col] = {category: i for i, category in enumerate(categories)}
        data[col] = data[col].map(category_mapping[col])
    data = data.apply(pd.to_numeric, errors='coerce')



    for i in range(len(lines)):   
        if i < split_point:
            X_train1.append(data.iloc[i, 0:8])
            Y_train1.append(data.iloc[i, 8:9])
        else:
            X_test1.append(data.iloc[i, 0:8])
            Y_test1.append(data.iloc[i, 8:9])
    
    train_X = pd.DataFrame(X_train1)
    train_Y = pd.DataFrame(Y_train1)
    test_X = pd.DataFrame(X_test1)
    test_Y = pd.DataFrame(Y_test1)

    train_X = train_X.drop(train_X.columns[0], axis=1)
    test_X = test_X.drop(test_X.columns[0], axis=1)

    X_train_values = train_X.values
    X_test_values = test_X.values
    Y_train_values = train_Y.values.tolist()
    
    print(get_score(LogisticRegression(), X_train_values, Y_train_values, X_test_values, test_Y))
    print(get_score(tree.DecisionTreeClassifier(), X_train_values, Y_train_values, X_test_values, test_Y))
    print(get_score(KNeighborsClassifier(), X_train_values, Y_train_values, X_test_values, test_Y))
    print(get_score(GaussianNB(), X_train_values, Y_train_values, X_test_values, test_Y))
    print(get_score(MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1), X_train_values, Y_train_values, X_test_values, test_Y))

    train_X.to_csv('X_train.csv', index=False)
    train_Y.to_csv('Y_train.csv', index=False)
    test_X.to_csv('X_test.csv', index=False)
    test_Y.to_csv('Y_test.csv', index=False)

main()

