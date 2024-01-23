import pandas as pd 
import numpy as np
import csv
from collections import Counter
import math

csv_file_path = 'wdbc.data'

import numpy as np
import pandas as pd

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
    #open file in read mode and read the data
    data = pd.read_csv(csv_file_path, header= None)

    train_split = 0.8

    X_train, X_test = np.split(data.sample(frac=1), [int(train_split*len(data))])
    
    index_to_remove = 1  # Index of the column to remove

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


    predictions = KNN(X_train_values, Y_train_values, X_test_values, 2)
    print(predictions)

    acc = np.sum(predictions == Y_test_values)/len(Y_test_values)
    print(acc)

    nb = NaiveBayesClassifier()
    nb.fit(np.array(X_train_values), np.array(Y_train_values))
    predictions_Bayes = nb.predict(np.array(X_test_values))

    print(predictions_Bayes)


    X_train.to_csv('X_train.csv', index=False)
    X_test.to_csv('X_test.csv', index=False)
    train_Y.to_csv('Y_train.csv', index=False)
    test_Y.to_csv('Y_test.csv', index=False)


main()
