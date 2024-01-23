import pandas as pd
import numpy as np
from collections import Counter
import random 
import csv
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier


csv_file_path = 'agaricus-lepiota.data'

def get_score (model, X_train, Y_train, X_test, Y_test):
    mymodel = model.fit(X_train, Y_train)
    return cross_val_score(mymodel, X_train, Y_train, cv=5)


def main():
    data = []

    X_train = []
    X_test = []

    # Initialize lists to store the last column of each split
    Y_train= []
    Y_test = []

    with open(csv_file_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)

        for row in reader:
            data.append(row)

    random.shuffle(data)

    df = pd.DataFrame(data)
    
    train_split = 0.8

    count = 0
    columns_to_map = ['p','x','s','n','t','p','f','c','n','k','e','e','s','s','w','w','p','w','o','p','k','s','u']
    category_mapping = {}
    for col in columns_to_map:
        categories = df[col].unique() if count == 0 else sorted(df[col].unique())
        category_mapping[col] = {category: i for i, category in enumerate(categories)}
        df[col] = df[col].map(category_mapping[col])
        count += 1

    X_train, X_test = np.split(df.sample(frac=1), [int(train_split*len(df))])
    
    index_to_remove = 0  # Index of the column to remove

    #remove target column from x data and add to y data frame
    column_to_remove = X_train.pop(X_train.columns[index_to_remove])
    Y_train = pd.DataFrame({X_train.columns[index_to_remove]: column_to_remove})

    column_to_remove = X_test.pop(X_test.columns[index_to_remove])
    Y_test = pd.DataFrame({X_test.columns[index_to_remove]: column_to_remove})

    X_train = pd.DataFrame(X_train)
    Y_train = pd.DataFrame(Y_train)
    X_test = pd.DataFrame(X_test)
    Y_test = pd.DataFrame(Y_test)

    X_test = X_test.apply(pd.to_numeric, errors='coerce')
    X_train = X_train.apply(pd.to_numeric, errors='coerce')
    Y_test = Y_test.apply(pd.to_numeric, errors='coerce')
    Y_train = Y_train.apply(pd.to_numeric, errors='coerce')

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
