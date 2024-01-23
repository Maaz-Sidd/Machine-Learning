import pandas as pd
import numpy as np
import random 
import re


csv_file_path = 'ecoli.data'

class Node():
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, info_gain=None, value=None):        
        # for decision node
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.info_gain = info_gain
        
        # for leaf node
        self.value = value

class DecisionTreeClassifier():
    def __init__(self, min_samples_split=2, max_depth=2):
        
        # initialize the root of the tree 
        self.root = None
        
        # stopping conditions
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        
    def build_tree(self, dataset, curr_depth=0):
 
        
        X, Y = dataset[:,:-1], dataset[:,-1]
        num_samples, num_features = np.shape(X)
        
        # split until stopping conditions are met
        if num_samples>=self.min_samples_split and curr_depth<=self.max_depth:
            # find the best split
            best_split = self.get_best_split(dataset, num_samples, num_features)
            # check if information gain is positive
            if best_split["info_gain"]>0:
                # recur left
                left_subtree = self.build_tree(best_split["dataset_left"], curr_depth+1)
                # recur right
                right_subtree = self.build_tree(best_split["dataset_right"], curr_depth+1)
                # return decision node
                return Node(best_split["feature_index"], best_split["threshold"], 
                            left_subtree, right_subtree, best_split["info_gain"])
        
        # compute leaf node
        leaf_value = self.calculate_leaf_value(Y)
        # return leaf node
        return Node(value=leaf_value)
    
    def get_best_split(self, dataset, num_samples, num_features):

        
        # dictionary to store the best split
        best_split = {}
        max_info_gain = -float("inf")
        
        # loop over all the features
        for feature_index in range(num_features):
            feature_values = dataset[:, feature_index]
            possible_thresholds = np.unique(feature_values)
            # loop over all the feature values present in the data
            for threshold in possible_thresholds:
                # get current split
                dataset_left, dataset_right = self.split(dataset, feature_index, threshold)
                # check if childs are not null
                if len(dataset_left)>0 and len(dataset_right)>0:
                    y, left_y, right_y = dataset[:, -1], dataset_left[:, -1], dataset_right[:, -1]
                    # compute information gain
                    curr_info_gain = self.information_gain(y, left_y, right_y)
                    # update the best split if needed
                    if curr_info_gain>max_info_gain:
                        best_split["feature_index"] = feature_index
                        best_split["threshold"] = threshold
                        best_split["dataset_left"] = dataset_left
                        best_split["dataset_right"] = dataset_right
                        best_split["info_gain"] = curr_info_gain
                        max_info_gain = curr_info_gain
                        
        # return best split
        return best_split
    
    def split(self, dataset, feature_index, threshold):

        
        dataset_left = np.array([row for row in dataset if row[feature_index]<=threshold])
        dataset_right = np.array([row for row in dataset if row[feature_index]>threshold])
        return dataset_left, dataset_right
    
    def information_gain(self, parent, l_child, r_child):
        
        weight_l = len(l_child) / len(parent)
        weight_r = len(r_child) / len(parent)
        
        gain = self.entropy(parent) - (weight_l*self.entropy(l_child) + weight_r*self.entropy(r_child))
        return gain
    
    def entropy(self, y):
        
        class_labels = np.unique(y)
        entropy = 0
        for cls in class_labels:
            p_cls = len(y[y == cls]) / len(y)
            entropy += -p_cls * np.log2(p_cls)
        return entropy
    
    def calculate_leaf_value(self, Y):
        
        Y = list(Y)
        return max(Y, key=Y.count)
    
    def fit(self, X, Y):
        
        dataset = np.concatenate((X, Y), axis=1)
        self.root = self.build_tree(dataset)
    
    def predict(self, X):
        
        preditions = [self.make_prediction(x, self.root) for x in X]
        return preditions
    
    def make_prediction(self, x, tree):
        
        if tree.value!=None: return tree.value
        feature_val = x[tree.feature_index]
        if feature_val<=tree.threshold:
            return self.make_prediction(x, tree.left)
        else:
            return self.make_prediction(x, tree.right)



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

    data.to_csv('test.csv', index=False)

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

    classifier = DecisionTreeClassifier(min_samples_split=3, max_depth=3)
    classifier.fit(X_train_values,Y_train_values)

    print(classifier.predict(X_test_values))

    train_X.to_csv('X_train.csv', index=False)
    train_Y.to_csv('Y_train.csv', index=False)
    test_X.to_csv('X_test.csv', index=False)
    test_Y.to_csv('Y_test.csv', index=False)

main()

