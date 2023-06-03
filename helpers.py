# Import Libraries
import pandas as pd
import numpy as np
pd.set_option('display.width',170, 'display.max_rows',200, 'display.max_columns',900)
from pyspark.sql.types import *




# Define functions to help perform operations in the jupyter notebook

def print_dataframe_info(df):
    """
    Prints information about the input DataFrame.

    Description:
        This function takes an input DataFrame and prints its shape, column headers, and data types.

    Parameters:
        df: A DataFrame to print information about.

    Returns:
        None
    """
    print ("\n\n---------------------")
    print ("DATAFRAME INFORMATION")
    print ("---------------------")
    print ("Dataframe shape:", df.shape, "\n")
    print ("Column Headers:", list(df.columns.values), "\n")
    print (df.dtypes)




def print_dataframe_report(df):
    """
    Prints a report of a DataFrame's unique, missing, and non-numeric values.

    Description:
        This function takes an input DataFrame and prints a report of its unique values, missing values, and non-numeric values. 
        For each column in the DataFrame, the function prints the number of unique values and lists up to 10 unique values. 
        The function also calculates the percentage of missing values in each column and prints a list of columns with missing values. 
        Finally, the function checks for non-numeric values in each column and prints a list of columns with non-numeric values.

    Parameters:
        df: A DataFrame to generate a report for.

    Returns:
        None
    """
    import re
    
    missing = []
    non_numeric = []
    
    for column in df:
        # find unique values
        values = df[column].unique() 
        st = "{} has {} unique values".format(column, values.size)
        print(st)
        
        if (values.size > 10):
            print("Listing up to 10 unique values:")
            print(values[0:10])
            print ("\n---------------------\n")
            
        # find missing values in features
        if (True in pd.isnull(values)):
            percentage = 100 * pd.isnull(df[column]).sum() / df.shape[0]
            s = "{} is missing in {}, {}%.".format(pd.isnull(df[column]).sum(), column, percentage)
            missing.append(s)

        # find non-numeric values in features
        for i in range(1, np.prod(values.shape)):
            if (re.match('nan', str(values[i]))):
                break
            if not (re.search('(^\d+\.?\d*$)|(^\d*\.?\d+$)', str(values[i]))):
                non_numeric.append(column)
                break
    print ("\n~~~~~~~~~~~~~~~~~~~~~~\n")
    print ("Features with missing values:")
    for i in range(len(missing)):
        print("\n{}" .format(missing[i]))  
    print ("\nFeatures with non-numeric values:")
    print("\n{}" .format(non_numeric))
    print ("\n~~~~~~~~~~~~~~~~~~~~~~\n")




def eliminate_missing_data(df):
    """
    Cleans the data within the DataFrame by removing missing values.

    Description:
        This function takes an input DataFrame and removes columns that are missing 90% or more of their values. It also
        removes rows where all elements are missing. The cleaned DataFrame is then returned.

    Parameters:
        df: A DataFrame containing data that needs to be cleaned.

    Returns:
        df: A DataFrame representing the cleaned data.
    """
    print("Dropping missing data...")
    
    columns_to_drop = []
    
    for column in df:
        values = df[column].unique() 
        
        # find missing values in features
        if (True in pd.isnull(values)):
            missing_data_percentage = 100 * pd.isnull(df[column]).sum() / df.shape[0]
            
            if (missing_data_percentage >= 90):
                columns_to_drop.append(column)
    
    # Drop columns that have 90% or more of their values missing
    df = df.drop(columns=columns_to_drop)
    # drop rows where all elements are missing
    df = df.dropna(how='all')

    print("Cleaning complete!")
    return df




def drop_duplicate_rows(df):
    """
    Drops duplicate rows from the input DataFrame.

    Description:
        This function takes an input DataFrame and drops any duplicate rows. The number of rows dropped is printed.

    Parameters:
        df: A DataFrame to drop duplicate rows from.

    Returns:
        df: A DataFrame with duplicate rows removed.
    """
    print("Dropping duplicate rows...")
    row_count_before = df.shape[0]
    # drop duplicate rows
    df = df.drop_duplicates()
    print("{} rows dropped.".format(row_count_before - df.shape[0]))
    return df




def write_to_parquet(df, output_path, table_name):
    """
    Writes the input DataFrame to a parquet file.

    Description:
        This function takes an input DataFrame and writes it to a parquet file at the specified output path with the specified table name. The function prints the table name and file path before writing and prints a confirmation message after writing is complete.

    Parameters:
        df: A DataFrame to write to a parquet file.
        output_path: The path where the output parquet file should be written.
        table_name: The name of the table to be written.

    Returns:
        None
    """
    
    file_path = output_path + table_name
    print("Writing table {} to {}".format(table_name, file_path))
    df.write.mode("overwrite").parquet(file_path)
    print("Write complete!")




def perform_quality_check(input_df, table_name):
    """
    Performs a data quality check on the input DataFrame.

    Description:
        This function takes an input DataFrame and performs a data quality check by ensuring that there are records in the table. The function prints a message indicating whether the data quality check passed or failed.

    Parameters:
        input_df: A DataFrame to perform a data quality check on.
        table_name: The name of the table being checked.

    Returns:
        0
    """
    
    record_count = input_df.count()

    if (record_count == 0):
        print("Data quality check failed for {} with zero records!".format(table_name))
    else:
        print("Data quality check passed for {} with record_count: {} records.".format(table_name, record_count))
        
    return 0
    
    