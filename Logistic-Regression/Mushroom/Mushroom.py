import pandas as pd
import numpy as np
import random 
import csv


csv_file_path = 'agaricus-lepiota.data'

num_iterations = 1000
learning_rate = 0.0001



def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def log_reg (X_train, Y_train, X_test, Y_test):

    n_samples, n_features = X_train.shape
    np.reshape(Y_train, (1, n_samples))
    w1 = np.zeros(( n_features, 1))  
    w0 = 0

    # Gradient descent
    for n in range(num_iterations):
        linear_model =  np.dot( X_train, w1)
        for i in range(len(linear_model)):
            linear_model[i] += w0
        predictions = sigmoid(linear_model)

        epsilon = 1e-15  # Small value to avoid division by zero

        # Avoiding division by zero and numerical instability
        y_pred = np.clip(predictions, epsilon, 1 - epsilon)
        cost = -1/n_samples * np.sum(Y_train * np.log(y_pred) + (1 - Y_train) * np.log(1 - y_pred))
        if n%10 == 0:
            print(cost)

        # Calculate gradients
        dw1 = (1/n_samples) * np.dot(X_train.T, (predictions - Y_train))
        dw0 = (1/n_samples) * np.sum((predictions - Y_train))
        
        
        # Update weights and bias
        w1 -= learning_rate * dw1
        w0 -= learning_rate * dw0

    predictions_1 = predict(X_test, w1, w0)
    print("Predictions:", predictions_1)

# Predict
def predict(X, weights, bias):
    linear_model = np.dot(X, weights) 
    for i in range(len(linear_model)):
            linear_model[i] += bias
    predictions = sigmoid(linear_model)
    return [1 if p >= 0.5 else 0 for p in predictions]



    
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

    log_reg(X_train, Y_train, X_test, Y_test)
    
    X_train.to_csv('X_train.csv', index=False)
    X_test.to_csv('X_test.csv', index=False)
    Y_train.to_csv('Y_train.csv', index=False)
    Y_test.to_csv('Y_test.csv', index=False)

main()
'''
    split_point = int(0.8 * len(data))

    df = df.apply(pd.to_numeric, errors='coerce')

    for i in range(len(data)):   
        if i < split_point:
            X_train.append(df.iloc[i, 1:22])
            Y_train.append(df.iloc[i, 0:1])
        else:
            X_test.append(df.iloc[i, 1:22])
            Y_test.append(df.iloc[i, 0:1])
'''