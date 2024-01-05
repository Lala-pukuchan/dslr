import pandas as pd
import sys
import math
import matplotlib.pyplot as plt


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


def existing_data_percentage(column_data):
    """
    Percentage of the existing data
    """
    return count_(column_data) / len(column_data) * 100


def mean_(column_data):
    """
    Calculate the mean of a column
    """
    _count = count_(column_data)
    if _count == 0:
        return 0
    return sum(column_data) / _count


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


def iqr_(column_data):
    """
    Q3 - Q1
    """
    return percentile_(column_data, 75) - percentile_(column_data, 25)


def max_(column_data):
    """
    Return the maximum value of a column
    """
    max = column_data[0]
    for value in column_data:
        if value > max:
            max = value
    return max


def skewness_(column_data):
    """
    skewness describes the asymmetry of a distribution in a set of data
    """
    _mean = mean_(column_data)
    _std = std_(column_data, _mean)
    if _std == 0:
        return 0
    return mean_([(x - _mean) ** 3 for x in column_data]) / (_std**3)


def kurtosis_(column_data):
    """
    Kurtosis measures how heavy or light the tails of a distribution are compared to a normal distribution.
    """
    _mean = mean_(column_data)
    _std = std_(column_data, _mean)
    if _std == 0:
        return 0
    return (
        sum((x - _mean) ** 4 for x in column_data) / (len(column_data) * _std**4) - 3
    )


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
    stats = [
        "Count",
        "ExistingData%",
        "Mean",
        "Std",
        "Min",
        "25%",
        "50%",
        "75%",
        "IQR",
        "Max",
        "Skewness",
        "Kurtosis",
    ]
    df = pd.DataFrame(index=stats, columns=numeric_columns)

    for column in numeric_columns:
        # Count the number of non-null values
        df.at["Count", column] = count_(input_data[column])
        # Percentage of existing data
        df.at["ExistingData%", column] = existing_data_percentage(input_data[column])
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
        # Calculate the 75th percentile - 25th percentile
        df.at["IQR", column] = iqr_(cleansed_column)
        # Calculate the maximum
        df.at["Max", column] = max_(cleansed_column)
        # Calculate the skewness
        df.at["Skewness", column] = skewness_(cleansed_column)
        # Calculate the kurtosis
        df.at["Kurtosis", column] = kurtosis_(cleansed_column)

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
