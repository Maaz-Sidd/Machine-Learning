import pandas as pd #import libraries
import numpy as np
import csv

#file path to open
csv_file_path = 'wdbc.data'


data = []

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
    #open file in read mode and read the data
    with open(csv_file_path, 'r') as csvfile: 
        reader = csv.DictReader(csvfile)

        #append data to data list
        for row in reader:
            data.append(row)

    #tuen list into a pandas dataframe
    df = pd.DataFrame(data)

    train_split = 0.8 #train split ratio

    #split data into x_train and X_test data frames
    X_train, X_test = np.split(df.sample(frac=1), [int(train_split*len(df))])
    
    index_to_remove = 1  # Index of the column to remove

    #remove target column from x data and add to y data frame
    column_to_remove = X_train.pop(X_train.columns[index_to_remove])
    Y_train = pd.DataFrame({X_train.columns[index_to_remove]: column_to_remove})

    column_to_remove = X_test.pop(X_test.columns[index_to_remove])
    Y_test = pd.DataFrame({X_test.columns[index_to_remove]: column_to_remove})

    X_test = X_test.apply(pd.to_numeric, errors='coerce')
    X_train = X_train.apply(pd.to_numeric, errors='coerce')

    Y_train["17.99"] = Y_train["17.99"].map({'M': 1, 'B': 0})
    Y_test["17.99"] = Y_test["17.99"].map({'M': 1, 'B': 0})

    del X_train[X_train.columns[0]]
    del X_test[X_test.columns[0]]


    log_reg(X_train, Y_train, X_test, Y_test)

    #convert all data frames to csv files
    X_train.to_csv('X_train.csv', index=False)
    X_test.to_csv('X_test.csv', index=False)
    Y_train.to_csv('Y_train.csv', index=False)
    Y_test.to_csv('Y_test.csv', index=False)


main()
































    
    
