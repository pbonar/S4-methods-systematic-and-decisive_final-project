from typing import Optional
import pandas as pd
import pyreadstat
import csv
from sklearn.model_selection import train_test_split

# Converter between .sav and .csv format
def sav_to_csv(filepath: str) -> str:

    # Checking if the file exists and is accessible
    try:
        # Reading .sav file
        df, meta = pyreadstat.read_sav(filepath)

        # Saving the file as .csv
        new_filepath = filepath[:-3] + 'csv'
        df.to_csv(new_filepath, index=False, sep=",")

        return new_filepath

    # Handling file not found errors
    except FileNotFoundError:
        print("File does not exist")

# Read a CSV file and extract column names into dictionary
def csv_to_table(csv_filepath: str) -> list:
    data_dict = []
    with open(csv_filepath, 'r') as file:
        csv_reader = csv.reader(file, delimiter='\t')
        for row in csv_reader:
            if len(row) == 2:
                num, name = row
                data_dict.append(name)
        return data_dict

# Select columns from a CSV file based on a given list of column names
def csv_column_selector(csv_filepath: str, column_list: list) -> Optional[str]:
    try:
        # Reading CSV file
        df = pd.read_csv(csv_filepath, delimiter=',', low_memory=False)

        # Getting list of columns from the column_list
        selected_columns = [column for column in df.columns if column in column_list]

        # Filtering dataframe with selected columns
        selected_df = df[selected_columns]

        # Getting the file name without extension
        file_name = csv_filepath[:-4]

        # Saving filtered dataframe to CSV with "_shorter" suffix
        new_filepath = file_name + '_selected_columns.csv'
        selected_df.to_csv(new_filepath, index=False)

        return new_filepath

    except FileNotFoundError:
        print("File does not exist")

# Validation of rows in a DataFrame based on specified conditions
def csv_rows_validation(df: pd.DataFrame, row: list) -> pd.DataFrame:
    try:
        column_name = row[1]
        valid_values = row[2]

        # Ensuring the column is treated as string for comparison
        valid_values = [str(value) for value in valid_values]

        # Raising a KeyError if the column doesn't exist in the DataFrame
        if column_name not in df.columns:
            raise KeyError(column_name)

        # Creating a mask and using it to identify rows that need to be replaced or removed
        mask = df[column_name].isin(valid_values)
        for i in range(len(df)):
            value = str(int(df.at[i, column_name]))
            # Checking if value is in valid values
            if value in valid_values:
                mask.iloc[i] = True
            # If it is not valid trying to replace it
            if not mask.iloc[i]:
                replacement = None
                if len(row) > 3:
                    # Looking for replacement
                    for pair in row[3]:
                        if value == str(pair[0]):
                            replacement = str(pair[1])
                            break
                    # Making the replacement
                    if replacement is not None:
                        df.at[i, column_name] = replacement
                        mask.iloc[i] = True  # Mark as valid after replacement

        # Filtering out rows that didn't meet the criteria based on the mask
        validated_df = df[mask]
        return validated_df

    except KeyError as e:
        print(f"Column '{row[1]}' does not exist in the DataFrame")
        return df

# Preparing data for validation and passing on to actual validation
def csv_validate(csv_filepath: str, values_filepath: str) -> None:
    # Reading CSV file
    df = pd.read_csv(csv_filepath, delimiter=',', low_memory=False)
    rows_list = []

    # Opening the file with all the valid values and replacement instructions for each column
    with open(values_filepath, 'r') as file:
        for line in file:
            line = line.strip()
            if line:
                line = line.split('\t')
                # Preparing valid values
                if len(line) > 2:
                    line[2] = line[2].split(',')
                # Preparing replacements
                if len(line) > 3:
                    line[3] = [item.split(',') for item in line[3].split('-')]
                rows_list.append(line)

    # Validation of all of the rows with instructions for different columns
    for row in rows_list:
        df = csv_rows_validation(df, row)

    # Saving filtered dataframe to CSV with "_validated" suffix
    new_filepath = csv_filepath[:-4] + '_validated.csv'
    df.to_csv(new_filepath, index=False)

# Split a CSV file into training and testing datasets
def data_split(csv_filepath: str, train_filepath: str, test_filepath: str, test_size: float) -> None:
    try:
        # Reading CSV file
        df = pd.read_csv(csv_filepath, delimiter=',', low_memory=False)

        # Spliting the data
        train_df, test_df = train_test_split(df, test_size=test_size)

        # Saving the training data to a CSV file
        train_df.to_csv(train_filepath, index=False)

        # Saving the testing data to a CSV file
        test_df.to_csv(test_filepath, index=False)

    except FileNotFoundError:
        print("File does not exist")


def main():
    # Defining paths to files
    filepath_data = "../data/NSDUH_2022.sav"
    filepath_columns = "../data/columns_selected_names.csv"
    filepath_verification = "../data/columns_verification_values.txt"
    train_filepath = "../data/NSDUH_2022_train.csv"
    test_filepath = "../data/NSDUH_2022_test.csv"

    # Convertion from .sav to .csv and get the dictionary
    filepath_data = sav_to_csv(filepath_data)

    # Choosing columns based on the dictionary and save to a new file
    result = csv_to_table(filepath_columns)
    filepath_data = csv_column_selector(filepath_data, result)

    # Validation of data based on specified criteria
    csv_validate(filepath_data, filepath_verification)

    # Spliting the validated data into training and testing datasets
    test_size = 0.2  # 20% of the data for testing
    data_split(filepath_data, train_filepath, test_filepath, test_size)

if __name__ == "__main__":
    main()
