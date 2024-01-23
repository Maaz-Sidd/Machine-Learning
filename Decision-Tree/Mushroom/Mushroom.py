import pandas as pd
import numpy as np
import random 
import csv


csv_file_path = 'agaricus-lepiota.data'

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

    classifier = DecisionTreeClassifier(min_samples_split=3, max_depth=3)
    classifier.fit(X_train_values,Y_train_values)


    print(classifier.predict(X_test_values))

    
    X_train.to_csv('X_train.csv', index=False)
    X_test.to_csv('X_test.csv', index=False)
    Y_train.to_csv('Y_train.csv', index=False)
    Y_test.to_csv('Y_test.csv', index=False)

main()
