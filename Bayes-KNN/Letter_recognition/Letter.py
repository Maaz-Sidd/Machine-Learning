import pandas as pd
import numpy as np
from collections import Counter
import csv


csv_file_path = 'letter-recognition.data'

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


    
data = []
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
