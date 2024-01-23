import pandas as pd
import numpy as np
import csv


csv_file_path = 'letter-recognition.data'
    
data = []

num_iterations = 1000
learning_rate = 0.0001

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
    if prediction.any() > 25:
        prediction = 25
    return prediction


    
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

    Lin_reg(X_train, Y_train, X_test, Y_test)
   
    X_train.to_csv('X_train.csv', index=False)
    X_test.to_csv('X_test.csv', index=False)
    Y_train.to_csv('Y_train.csv', index=False)
    Y_test.to_csv('Y_test.csv', index=False)

main()
