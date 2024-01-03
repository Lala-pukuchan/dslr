import sys
import pandas as pd
import numpy as np


def normalize(data):
    """
    Normalize a column
    """
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    return (data - mean) / std


def sigmoid(x):
    """
    Compute the sigmoid of x
    """
    return 1 / (1 + np.exp(-x))


def predict(x, theta):
    """
    Compute the prediction y_hat based on current theta
    """
    return sigmoid(np.dot(x, theta))


def gradient(x, y, theta):
    """
    Compute the gradient of the logistic loss with respect to theta
    """
    # size of the dataset
    m = len(y)

    # insert a column of 1's in the first position of the feature matrix
    x_insert_one = np.insert(x, 0, 1, axis=1)

    # compute the error
    error = predict(x_insert_one, theta) - y

    # compute the gradient
    return (1 / m) * np.dot(x_insert_one.T, error)


def fit(x, y, theta):
    """
    Fit the model using gradient descent
    """
    for i in range(1000):
        theta = theta - 0.01 * gradient(x, y, theta)


def logreg_train(input_data):
    """
    Train a logistic regression model with the input data
    """

    # features used for training
    features = [
        "Astronomy",
        "Herbology",
        "Defense Against the Dark Arts",
        "Ancient Runes",
    ]

    # target variable
    target = "Hogwarts House"

    # Drop rows with NaN values in selected columns (features + target)
    cleaned_data = input_data[features + [target]].dropna()
    X = cleaned_data[features].values
    y = cleaned_data[target].values.reshape(-1, 1)
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"X: {X}")
    print(f"y: {y}")

    # initialize theta
    theta = np.zeros(X.shape[1] + 1).reshape(-1, 1)
    print(f"theta shape: {theta.shape}")
    print(f"theta: {theta}")

    # train the model
    houses = np.unique(y)
    print(f"houses: {houses}")

    for house in houses:
        y_house = np.where(y == house, 1, 0)
        theta = fit(X, y_house, theta)
        print(f"theta: {theta}")


def main():
    """
    Take a csv file as input and perform logistic regression with the data
    """
    if len(sys.argv) == 2:
        file_name = sys.argv[1]
        print(f"The inputted file name is {file_name}")

        try:
            # Read a csv file
            input_data = pd.read_csv(file_name, index_col=0)
        except FileNotFoundError:
            print(f"File {file_name} not found")
            exit(1)
        except pd.errors.EmptyDataError:
            print(f"File {file_name} is empty")
            exit(1)
        except pd.errors.ParserError:
            print(f"File {file_name} is not a csv file")
            exit(1)
        except Exception as e:
            print(f"Error: {e}")
            exit(1)

        # train logistic regression model with the data
        logreg_train(input_data)

    else:
        print("Usage: python describe.py <file_name>")
        exit(1)


if __name__ == "__main__":
    main()
