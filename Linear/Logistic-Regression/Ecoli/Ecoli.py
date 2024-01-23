import pandas as pd
import numpy as np
import random 
import re


csv_file_path = 'ecoli.data'

num_iterations = 10000
learning_rate = 0.00001

def Lin_reg (X_train, Y_train, X_test, Y_test):


    n_samples, n_features = X_train.shape
    np.reshape(Y_train, (1, n_samples))
    w1 = np.zeros(( n_features, 1))  
    w0 = 0

    # Gradient descent
    for n in range(num_iterations):
        linear_model =  np.dot( X_train, w1)
        for i in range(len(linear_model)):
            linear_model[i] += w0

        cost = (1/(2*n_samples))*(np.sum(np.square(linear_model - Y_train)))
        if n%10 == 0:
            print(cost)

        # Calculate gradients
        dw1 = (1/n_samples) * np.dot(X_train.T, (linear_model - Y_train))
        dw0 = (1/n_samples) * np.sum((linear_model - Y_train))
        
        
        # Update weights and bias
        w1 -= learning_rate * dw1
        w0 -= learning_rate * dw0

    test = predict(X_test, w1, w0)
    print("Predictions:", test)

# Predict
def predict(X, weights, bias):
    linear_model = np.dot(X, weights) 
    for i in range(len(linear_model)):
            linear_model[i] += bias
    prediction = np.round(linear_model)
    return prediction



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

    Lin_reg(train_X, train_Y, test_X, test_Y)

    train_X.to_csv('X_train.csv', index=False)
    train_Y.to_csv('Y_train.csv', index=False)
    test_X.to_csv('X_test.csv', index=False)
    test_Y.to_csv('Y_test.csv', index=False)

    #test_Y[8] = test_Y[8].map({'pp': 0, 'im': 1, 'imU': 2, 'om': 3, 'imL': 4, 'cp': 5, 'imS': 6, 'omL': 7})
    #train_Y[8] = train_Y[8].map({'pp': 0, 'im': 1, 'imU': 2, 'om': 3, 'imL': 4, 'cp': 5, 'imS': 6, 'omL': 7})

    

main()

