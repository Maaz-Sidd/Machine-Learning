import pandas as pd 
import numpy as np
import csv
from collections import Counter
import math
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier

csv_file_path = 'wdbc.data'


def get_score (model, X_train, Y_train, X_test, Y_test):
    mymodel = model.fit(X_train, Y_train)
    return cross_val_score(mymodel, X_train, Y_train, cv=5)



def main():
    #open file in read mode and read the data
    data = pd.read_csv(csv_file_path, header= None)

    index_to_remove = 1  # Index of the column to remove

    train_split = 0.8

    X_train, X_test = np.split(data.sample(frac=1), [int(train_split*len(data))])
    

    X_train = pd.DataFrame(X_train)
    
    X_test = pd.DataFrame(X_test)

    #remove target column from x data and add to y data frame
    column_to_remove = X_train.pop(X_train.columns[index_to_remove])
    Y_train = pd.DataFrame({X_train.columns[index_to_remove]: column_to_remove})

    column_to_remove = X_test.pop(X_test.columns[index_to_remove])
    Y_test = pd.DataFrame({X_test.columns[index_to_remove]: column_to_remove})

    X_test = X_test.apply(pd.to_numeric, errors='coerce')
    X_train = X_train.apply(pd.to_numeric, errors='coerce')

    del X_train[X_train.columns[0]]
    del X_test[X_test.columns[0]]

    X_train_values = X_train.values
    X_test_values = X_test.values
    mapping = {'M': 0, 'B': 1}

    # Map the letters in the DataFrame to numbers
    Y_train_mapped = Y_train.applymap(lambda x: mapping[x])
    Y_test_mapped = Y_test.applymap(lambda x: mapping[x])

    Y_train_values = Y_train_mapped.values.tolist()
    Y_test_values = Y_test_mapped.values

    test_Y = pd.DataFrame(Y_test_values)
    train_Y = pd.DataFrame(Y_train_values)

    
    print(get_score(LogisticRegression(), X_train_values, Y_train_values, X_test_values, Y_test_values))
    print(get_score(tree.DecisionTreeClassifier(), X_train_values, Y_train_values, X_test_values, Y_test_values))
    print(get_score(KNeighborsClassifier(), X_train_values, Y_train_values, X_test_values, Y_test_values))
    print(get_score(GaussianNB(), X_train_values, Y_train_values, X_test_values, Y_test_values))
    print(get_score(MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1), X_train_values, Y_train_values, X_test_values, Y_test_values))


    X_train.to_csv('X_train.csv', index=False)
    X_test.to_csv('X_test.csv', index=False)
    train_Y.to_csv('Y_train.csv', index=False)
    test_Y.to_csv('Y_test.csv', index=False)


main()
