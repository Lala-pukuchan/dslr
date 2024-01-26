import numpy as np
import sys
import pandas as pd
import pickle
from logreg_train import MyLogisticRegression, min_max_scaling


def logreg_predict(input_data, type_gd):
    """
    Predict with trained logistic model
    """

    # depends on trained types of gradient descent, pick up a file
    if type_gd == "0":
        model_file_name = "0_trained_models_gd.pkl"
        output_file_name = "0_houses_gd.csv"
    elif type_gd == "1":
        model_file_name = "1_trained_models_sgd.pkl"
        output_file_name = "1_houses_sgd.csv"
    elif type_gd == "2":
        model_file_name = "2_trained_models_mini_gd.pkl"
        output_file_name = "2_houses_mini_gd.csv"
    elif type_gd == "3":
        model_file_name = "3_trained_models_momentum_sgd.pkl"
        output_file_name = "3_houses_momentum_sgd.csv"

    # Load the trained models
    with open(model_file_name, "rb") as file:
        trained_models = pickle.load(file)

    # features used for training
    features = [
        "Astronomy",
        "Herbology",
        "Defense Against the Dark Arts",
        "Ancient Runes",
    ]

    # Drop rows with NaN values in selected columns
    input_data[features] = input_data[features].fillna(input_data[features].mean())
    x = input_data[features].values

    # Normalize data x
    x = min_max_scaling(x)
    print(f"x shape: {x.shape}")

    # storage for probabilities, models
    probabilities = {}

    # predict with one vs all
    for i, model in enumerate(trained_models.values()):
        probabilities[i] = model.predict_(x)

    # combine probabilities of all houses
    all_houses_probabilities = np.column_stack(
        [probabilities[i] for i in range(len(trained_models))]
    )
    print("First 5 rows of all_houses_probabilities:\n", all_houses_probabilities[:5])

    # classify the house with the highest probabilities
    predicted_y = np.argmax(all_houses_probabilities, axis=1)
    print("First 5 rows of predicted_y:\n", predicted_y[:5])

    # Get houses index and name
    index_to_house = {i: house for i, house in enumerate(trained_models.keys())}

    # Map the predicted indices to house names
    predicted_houses = [index_to_house[idx] for idx in predicted_y]

    # Combine the predicted indices and house names into a DataFrame
    predictions_df = pd.DataFrame(
        {"Hogwarts House": predicted_houses}
    )

    # Display the first few rows
    print(predictions_df.head())

    # Save to a CSV file
    predictions_df.to_csv(output_file_name, index=True)
    print("Predictions saved to ", output_file_name)


def main():
    """
    Take a csv file as input and perform logistic regression with the data
    """
    if len(sys.argv) == 3:
        file_name = sys.argv[1]
        type_gd = sys.argv[2]
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
        logreg_predict(input_data, type_gd)

    else:
        print(
            "Usage: python logreg_predict.py dataset_test.csv <0:BatchGD/1:StochasticGD/2:MiniBatchGD/3:momentumSGD>"
        )
        exit(1)


if __name__ == "__main__":
    main()
