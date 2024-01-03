import pandas as pd
import sys
from histogram import histogram
import os
from describe import is_numeric_dtype
from scatter_plot import scatter_plot

def visualize(data):
    """
    Visualize the data
    """
    # make a directory for visualizations
    directory = "./visualizations"
    os.makedirs(directory, exist_ok=True)
    
    # creaete a histogram for each course
    histogram(data)

    # Get the numeric columns
    numeric_columns = []
    for column in data.columns:
        if is_numeric_dtype(data[column]):
            numeric_columns.append(column)

    # create a scatter plot for each feature
    scatter_plot(data[numeric_columns])


def main():
    """
    Take a csv file as input and visualize the data
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

        # Describe the data
        visualize(input_data)
    else:
        print("Usage: python describe.py <file_name>")
        exit(1)


if __name__ == "__main__":
    main()
