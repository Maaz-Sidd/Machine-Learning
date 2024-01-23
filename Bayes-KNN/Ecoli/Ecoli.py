import pandas as pd
import numpy as np
import random 
from collections import Counter
import statistics
import re


csv_file_path = 'ecoli.data'

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
    
    predictions = KNN(X_train_values, Y_train_values, X_test_values, 5)
    print(predictions)

    acc = np.sum(predictions == test_Y.values)/len(Y_test1)
    print(acc)
    
    nb = NaiveBayesClassifier()
    nb.fit(np.array(X_train_values), np.array(Y_train_values))
    predictions_Bayes = nb.predict(np.array(X_test_values))

    print(predictions_Bayes)

    train_X.to_csv('X_train.csv', index=False)
    train_Y.to_csv('Y_train.csv', index=False)
    test_X.to_csv('X_test.csv', index=False)
    test_Y.to_csv('Y_test.csv', index=False)

main()

