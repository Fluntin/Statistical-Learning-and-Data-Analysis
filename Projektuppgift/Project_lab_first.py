# Importing necessary libraries for data manipulation and visualization
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
def run_test():
    # Loading the dataset from a CSV file into a DataFrame
    df = pd.read_csv("Datafil.csv")
    
    # Calculating the mean value of the "run 1" column and printing it
    mean_score_run_1 = np.mean(df["run 1"])
    print(mean_score_run_1)

    # Calculating the mean of the "run 1" column for the first 100 rows
    mean_run_1_100 = np.mean(df[0:100]["run 1"])

    # Creating a boolean mask to filter rows where the location is "las vegas"
    vegas_mask = (df["location"] == "las vegas")
    # Calculating the mean of the "run 1" column for rows where the location is "las vegas"
    mean_run_1_vegas = np.mean(df[vegas_mask]["run 1"])
    print("The mean for run1 in Vegas was:", mean_run_1_vegas)

    # Calculating the sum of values in the "run 1" and "run 2" columns row-wise
    run_sums = df[["run 1", "run 2"]].sum(axis=1)
    # Adding a new column "run sums" to the DataFrame containing the calculated sums
    df["run sums"] = run_sums

    # Getting a list of all the column names in the DataFrame
    cols = df.columns.tolist()

    # Defining a function that multiplies its input by 2
    def f(x):
        return 2*x  

    # Applying the function `f` to the "run sums" column to multiply each value by 2
    run_sums_times_two = df["run sums"].apply(f)

    # Concatenating the original DataFrame (without the last column) with the new "run sums times two" column
    df = pd.concat([df[cols[:-1]], run_sums_times_two], axis=1)

    # Creating a NumPy array for testing purposes
    Test = np.array([1,1,2,5,3,3,3,4,2])
    # Printing the sorted version of the 'Test' array
    print(np.sort(Test))

    # Finding unique values in the 'Test' array and their respective counts
    unique_values, counts = np.unique(Test, return_counts=True)
    # Printing the unique values and their counts
    print(f"{unique_values=}")
    print(f"{counts=}")
    
# Calling the `run_test` function to execute the above operations
#run_test()

df = pd.read_csv("Datafil.csv")


def normalize_scores(input_file, output_file):
    with open(input_file, "r") as infile, open(output_file, "w", newline='') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        # Write headers to the new CSV
        headers = next(reader)
        writer.writerow(headers)

        for row in reader:
            # Normalize scores (from 6th column to end of row)
            for i in range(5, len(row)):
                if row[i]:  # Check if cell is not empty
                    row[i] = str(float(row[i]) / 10)
            writer.writerow(row)

# Use the function
# First, save your data to a file named "input_data.csv"
normalize_scores("Datafil.csv", "Datafil_normalized.csv")




