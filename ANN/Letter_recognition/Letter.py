import pandas as pd
import numpy as np
from collections import Counter
import csv
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier

csv_file_path = 'letter-recognition.data'
data = []

def get_score (model, X_train, Y_train, X_test, Y_test):
    mymodel = model.fit(X_train, Y_train)
    return cross_val_score(mymodel, X_train, Y_train, cv=5)

def main():
    with open(csv_file_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)

        for row in reader:
            data.append(row)

    df = pd.DataFrame(data)
    letter_mapping = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'J': 9, 'K':10, 
                      'L': 11, 'M': 12, 'N': 13, 'O': 14, 'P': 15, 'Q': 16, 'R': 17, 'S': 18, 'T': 19, 'U': 20, 'V':21, 
                      'W': 22, 'X': 23, 'Y': 24, 'Z': 25}
    df = df.applymap(lambda x: letter_mapping[x] if x in letter_mapping else x)
    df = df.apply(pd.to_numeric, errors='coerce')


    X_train, X_test = np.split(df.sample(frac=1), [int(0.8*len(df))])

    index_to_remove = 0  # Index of the column to remove

    Y_train = pd.DataFrame()
    Y_train[X_train.columns[index_to_remove]] = X_train.iloc[:, index_to_remove]

    # Remove the column from the first DataFrame
    X_train.drop(X_train.columns[index_to_remove], axis=1, inplace=True)
   
    Y_test = pd.DataFrame()
    Y_test[X_test.columns[index_to_remove]] = X_test.iloc[:, index_to_remove]

    # Remove the column from the first DataFrame
    X_test.drop(X_test.columns[index_to_remove], axis=1, inplace=True)

    X_train_values = X_train.values
    X_test_values = X_test.values
    Y_train_values = Y_train.values.tolist()
    Y_test_values = Y_test.values

    print(get_score(LogisticRegression(), X_train_values, Y_train_values, X_test_values, Y_test_values))
    print(get_score(tree.DecisionTreeClassifier(), X_train_values, Y_train_values, X_test_values, Y_test_values))
    print(get_score(KNeighborsClassifier(), X_train_values, Y_train_values, X_test_values, Y_test_values))
    print(get_score(GaussianNB(), X_train_values, Y_train_values, X_test_values, Y_test_values))
    print(get_score(MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1), X_train_values, Y_train_values, X_test_values, Y_test_values))
   
    X_train.to_csv('X_train.csv', index=False)
    X_test.to_csv('X_test.csv', index=False)
    Y_train.to_csv('Y_train.csv', index=False)
    Y_test.to_csv('Y_test.csv', index=False)

main()
