import numpy as np
import sys
import pandas as pd
import pickle


class MyLogisticRegression:
    """
    Description:
        My personnal logistic regression to classify things.
    """

    def __init__(self, theta, alpha=0.001, max_iter=1000):
        """
        Description:
            generator of the class, initialize self.
        """
        # error management
        if not isinstance(alpha, float) or alpha <= 0:
            raise ValueError("Alpha must be a positive float")
        if not isinstance(max_iter, int) or max_iter <= 0:
            raise ValueError("max_iter must be a positive integer")
        if not isinstance(theta, (list, np.ndarray)):
            raise TypeError("Theta must be a list or numpy array")

        self.theta = theta
        self.alpha = alpha
        self.max_iter = max_iter

    def sigmoid_(self, x):
        """
        Compute the sigmoid of a vector.
        Args:
            x: has to be a numpy.ndarray of shape (m, 1).
        Returns:
            The sigmoid value as a numpy.ndarray of shape (m, 1).
            None if x is an empty numpy.ndarray.
        Raises:
            This function should not raise any Exception.
        """
        # input validation
        if (not isinstance(x, np.ndarray)) or x.size == 0:
            return None
        return 1 / (1 + np.exp(-x))

    def predict_(self, x):
        """
        Description:
            Prediction of output using the hypothesis function (sigmoid).
        Args:
            x: a numpy.ndarray with m rows and n features.
        Returns:
            The prediction as a numpy.ndarray with m rows.
            None if x is an empty numpy.ndarray.
            None if x does not match the dimension of the
            training set.
        Raises:
            This function should not raise any Exception.
        """
        # input validation
        if (
            not isinstance(x, np.ndarray)
            or x.size == 0
            or x.shape[1] + 1 != self.theta.shape[0]
        ):
            return None

        # append 1 to x's first column
        x_dash = np.insert(x, 0, 1, axis=1)

        # compute y_hat
        y_hat = self.sigmoid_(np.dot(x_dash, self.theta))
        return y_hat

    def loss_elem_(self, y, yhat):
        """
        Description:
            Calculates the loss of each sample.
        Args:
            y: has to be an numpy.ndarray, a vector of dimension m.
            y_hat: has to be an numpy.ndarray, a vector of dimension m.
        Returns:
            yhat - y as a numpy.ndarray of dimension (1, m).
            None if y or y_hat are empty numpy.ndarray.
        Raises:
            This function should not raise any Exception.
        """
        # input validation
        if (
            not isinstance(y, np.ndarray)
            or not isinstance(yhat, np.ndarray)
            or y.size == 0
            or yhat.size == 0
            or y.shape != yhat.shape
        ):
            return None

        # prevent log(0), clip is to limit the value between min and max
        eps = 1e-15
        yhat = np.clip(yhat, eps, 1 - eps)

        # calculate loss with using cross-entropy
        return y * np.log(yhat) + (1 - y) * np.log(1 - yhat)

    def loss_(self, x, y):
        """
        Description:
            Calculates all the losses of the samples.
        Args:
            x: has to be an numpy.ndarray, a vector of dimension m.
            y: has to be an numpy.ndarray, a vector of dimension m.
        Returns:
            The loss as a float.
            None if x or y are empty numpy.ndarray
            and x's
        Raises:
            This function should not raise any Exception.
        """
        # input validation
        if (
            not isinstance(x, np.ndarray)
            or not isinstance(y, np.ndarray)
            or x.size == 0
            or y.size == 0
            or x.shape[0] != y.shape[0]
        ):
            return None

        # predict y_hat
        y_hat = self.predict_(x)

        # compute loss
        loss_elem = self.loss_elem_(y, y_hat)
        m = x.shape[0]
        return (-1) * np.sum(loss_elem) / m

    def vec_log_gradient_(self, x, y):
        """
        Computes a gradient vector from three non-empty numpy.ndarray,
        without any for-loop.
        The three arrays must have comp Args:
        x: has to be an numpy.ndarray, a matrix of shape m * n.
        y: has to be an numpy.ndarray, a vector of shape m * 1.
        theta: has to be an numpy.ndarray, a vector (n +1) * 1.
        Returns:
        The gradient as a numpy.ndarray, a vector of shape n * 1,
        containg the result of the formula for all j.
        None if x, y, or theta are empty numpy.ndarray.
        None if x, y and theta do not have compatible shapes.
        Raises:
        This function should not raise any Exception.
        """

        # input validation
        # if x, y and theta are not numpy.ndarray, return None
        if (
            not isinstance(x, np.ndarray)
            or not isinstance(y, np.ndarray)
            or not isinstance(self.theta, np.ndarray)
        ):
            return None

        # if x, y and theta are empty numpy.ndarray, return None
        if x.size == 0 or y.size == 0 or self.theta.size == 0:
            return None

        # if x, y and theta do not have compatible shapes, return None
        if not (x.shape[0] == y.shape[0] and x.shape[1] + 1 == self.theta.shape[0]):
            return None

        # number of examples
        m = x.shape[0]

        # insert 1 to the first column of x
        x_dash = np.insert(x, 0, 1, axis=1)

        # compute hypothesis with sigmoid
        hypothesis = self.predict_(x)

        # return gradient vector
        return (1 / m) * np.dot(x_dash.T, hypothesis - y)

    def fit_(self, x, y):
        """
        Description:
            Find the right theta to make loss minimum.
        Args:
            x: has to be a numpy.ndarray, a matrix of dimension m * n.
            y: has to be a numpy.ndarray, a vector of dimension m * 1.
        Returns:
            New theta as a numpy.ndarray, a vector of dimension n * 1
        Raises:
            This function should not raise any Exception.
        """
        # input validation
        # if x, y and theta are not numpy.ndarray, return None
        if (
            not isinstance(x, np.ndarray)
            or not isinstance(y, np.ndarray)
            or not isinstance(self.theta, np.ndarray)
        ):
            return None

        # if x, y and theta are empty numpy.ndarray, return None
        if x.size == 0 or y.size == 0 or self.theta.size == 0:
            return None

        # if x, y and theta do not have compatible shapes, return None
        if not (x.shape[0] == y.shape[0] and x.shape[1] + 1 == self.theta.shape[0]):
            return None

        # gradient descent
        for _ in range(self.max_iter):
            self.theta = self.theta - self.alpha * self.vec_log_gradient_(x, y)

        return self.theta


def min_max_scaling(x):
    min = x.min(axis=0)
    max = x.max(axis=0)
    return (x - min) / (max - min)


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
    formatted_data = input_data[features + [target]].dropna()
    x = formatted_data[features].values
    y = formatted_data[target].values.reshape(-1, 1)

    # Normalize data x
    x = min_max_scaling(x)

    print(f"X shape: {x.shape}")
    print(f"y shape: {y.shape}")
    print(f"X: {x}")
    print(f"y: {y}")

    # get unique houses
    houses = np.unique(y)
    print(f"houses: {houses}")

    # storage for probabilities, models
    probabilities = {}
    models = {}

    # convert data for one vs others
    for i, house in enumerate(houses):
        y_house = np.where(y == house, 1, 0).reshape(-1, 1)
        print("y_house.shape:", y_house.shape)

        # initialize model
        theta = np.zeros(x.shape[1] + 1).reshape(-1, 1)
        model = MyLogisticRegression(theta, alpha=1e-2, max_iter=5_000)

        # train model to fit theta
        model.fit_(x, y_house)

        # predict with trained model
        probabilities[i] = model.predict_(x)
        models[house] = model

    # combine probabilities of all houses
    all_houses_probabilities = np.column_stack(
        [probabilities[i] for i in range(len(houses))]
    )
    print("First 5 rows of all_houses_probabilities:\n", all_houses_probabilities[:5])

    # classify the house with the highest probabilities
    predicted_y = np.argmax(all_houses_probabilities, axis=1)
    print("First 5 rows of predicted_y:\n", predicted_y[:5])

    # convert string data to index data
    house_to_index = {house: i for i, house in enumerate(houses)}
    y_indices = np.array([house_to_index[house] for house in y.squeeze()])

    # calculate accuracy
    accuracy = np.sum(y_indices == predicted_y) / len(y) * 100
    print(f"accuracy: {accuracy:.2f}%")

    # Save the trained models to a pickle file
    with open("trained_models.pkl", "wb") as file:
        pickle.dump(models, file)


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
        print("Usage: python logreg_train.py dataset_train.py")
        exit(1)


if __name__ == "__main__":
    main()
