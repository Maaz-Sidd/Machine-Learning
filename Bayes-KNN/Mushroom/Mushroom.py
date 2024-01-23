import pandas as pd
import numpy as np
from collections import Counter
import random 
import csv


csv_file_path = 'agaricus-lepiota.data'

class NaiveBayesClassifier:
    def __init__(self):
        self.class_probs = {}
        self.mean = {}
        self.var = {}
        
    def fit(self, X_train, y_train):
        data = pd.DataFrame(X_train)
        data['target'] = y_train
        
        self.class_probs = {val: len(data[data['target'] == val]) / len(data) for val in data['target'].unique()}
        
        for label in self.class_probs:
            label_data = data[data['target'] == label].drop('target', axis=1)
            self.mean[label] = np.array(label_data.mean())
            self.var[label] = np.array(label_data.var())
    
    def _pdf(self, x, mean, var):
        return (1 / np.sqrt(2 * np.pi * var)) * np.exp(-((x - mean) ** 2) / (2 * var))
    
    def _predict_single(self, sample):
        posteriors = {}
        
        for label in self.class_probs:
            class_prob = self.class_probs[label]
            likelihood = np.prod(self._pdf(sample, self.mean[label], self.var[label]))
            posteriors[label] = class_prob * likelihood
        
        return max(posteriors, key=posteriors.get)
    
    def predict(self, X_test):
        predictions = [self._predict_single(sample) for sample in X_test]
        return predictions



def KNN(X_train, Y_train, X_test, K):
    predictions = []
    dis_arr = []
    
    for i in range(len(X_test)):
        dis_arr.clear()
        for j in range(len(X_train)):
            distances = np.sqrt(np.sum((X_train[j]-X_test[i])**2))
            dis_arr.append([j, distances])
        neighbor_arr = sorted(dis_arr, key = lambda x : x[1])
        class_arr = [Y_train[i[0]] for i in neighbor_arr[0:K]]
        counter = Counter(tuple(sublist) for sublist in map(tuple, class_arr))
        most_common_sublist = max(counter, key=counter.get)
        predictions.append(most_common_sublist)
    return list(predictions)
    
    
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

    predictions = KNN(X_train_values, Y_train_values, X_test_values, 5)
    print(predictions)

    acc = np.sum(predictions == Y_test_values)/len(Y_test_values)
    print(acc)

    nb = NaiveBayesClassifier()
    nb.fit(np.array(X_train_values), np.array(Y_train_values))
    predictions_Bayes = nb.predict(np.array(X_test_values))

    print(predictions_Bayes)
  
    
    X_train.to_csv('X_train.csv', index=False)
    X_test.to_csv('X_test.csv', index=False)
    Y_train.to_csv('Y_train.csv', index=False)
    Y_test.to_csv('Y_test.csv', index=False)

main()
