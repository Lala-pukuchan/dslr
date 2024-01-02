import pandas as pd
import sys
import math


def is_numeric_dtype(column_data):
    """
    Check if a column is numeric
    """
    if column_data.isna().all():
        return False
    elif pd.api.types.is_numeric_dtype(column_data):
        return True
    else:
        return False


def count_(column_data):
    """
    Count the number of non-null values in a column
    """
    return sum(pd.notnull(value) for value in column_data)


def mean_(column_data):
    """
    Calculate the mean of a column
    """
    return sum(column_data) / count_(column_data)


def std_(column_data, mean):
    """
    Calculate the standard deviation of a column
    """
    variance = sum((value - mean) ** 2 for value in column_data) / (
        count_(column_data) - 1
    )
    return math.sqrt(variance)


def min_(column_data):
    """
    Return the minimum value of a column
    """
    min = column_data[0]
    for value in column_data:
        if value < min:
            min = value
    return min


def percentile_(column_data, percent):
    """
    Return the value of a column at a given percentile
    """
    # sort the column data
    column_data.sort()

    # Calculate the index of the percentile
    n = count_(column_data)
    index = (n - 1) * percent / 100
    lower_index = int(math.floor(index))
    upper_index = int(math.ceil(index))

    # Ensure the index is within the range
    lower_index = max(min(lower_index, n - 1), 0)
    upper_index = max(min(upper_index, n - 1), 0)

    # Return the value at the index
    if lower_index == upper_index:
        return column_data[lower_index]
    else:
        return (column_data[lower_index] + column_data[upper_index]) / 2


def max_(column_data):
    """
    Return the maximum value of a column
    """
    max = column_data[0]
    for value in column_data:
        if value > max:
            max = value
    return max


def describe(input_data):
    """
    Describe the data with statistics
    """
    # Check each feature is numeric
    numeric_columns = []
    for column in input_data.columns:
        if is_numeric_dtype(input_data[column]):
            numeric_columns.append(column)

    # Create a DataFrame
    stats = ["Count", "Mean", "Std", "Min", "25%", "50%", "75%", "Max"]
    df = pd.DataFrame(index=stats, columns=numeric_columns)

    for column in numeric_columns:
        # Count the number of non-null values
        df.at["Count", column] = count_(input_data[column])
        # execlude NaN values
        cleansed_column = [val for val in input_data[column] if pd.notnull(val)]
        # Calculate the mean
        mean = mean_(cleansed_column)
        df.at["Mean", column] = mean
        # Calculate the standard deviation
        df.at["Std", column] = std_(cleansed_column, mean)
        # Calculate the minimum
        df.at["Min", column] = min_(cleansed_column)
        # Calculate the 25th percentile
        df.at["25%", column] = percentile_(cleansed_column, 25)
        # Calculate the 50th percentile
        df.at["50%", column] = percentile_(cleansed_column, 50)
        # Calculate the 75th percentile
        df.at["75%", column] = percentile_(cleansed_column, 75)
        # Calculate the maximum
        df.at["Max", column] = max_(cleansed_column)

    print(df)

    for column in numeric_columns:
        df.at["Count", column] = input_data[column].count()
        df.at["Mean", column] = input_data[column].mean()
        df.at["Std", column] = input_data[column].std()
        df.at["Min", column] = input_data[column].min()
        df.at["25%", column] = input_data[column].quantile(0.25)
        df.at["50%", column] = input_data[column].quantile(0.5)
        df.at["75%", column] = input_data[column].quantile(0.75)
        df.at["Max", column] = input_data[column].max()

    print(df)


def main():
    """
    Take a csv file as input and describe the data
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
        describe(input_data)
    else:
        print("Usage: python describe.py <file_name>")
        exit(1)


if __name__ == "__main__":
    main()
