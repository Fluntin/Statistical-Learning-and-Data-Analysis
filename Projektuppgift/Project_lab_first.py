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

# a) Here I normaise all of my data and create a separate file.
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
            
# b) here i create a histogram fr normalized data
def plot_histograms(datafile):
    # Read the normalized data
    df = pd.read_csv(datafile)

    # List of tricks
    tricks = ["trick 1", "trick 2", "trick 3", "trick 4"]

    # Plot histograms for each trick
    for idx, trick in enumerate(tricks, 1):
        plt.subplot(2, 2, idx)  # 2x2 grid of histograms
        plt.hist(df[trick].dropna(), bins=20, alpha=0.7, color='blue')  # dropna() ensures NaN values are ignored
        plt.title(f"Histogram of {trick}")
        plt.xlabel("Score")
        plt.ylabel("Frequency")

    # Adjust layout to prevent overlaps and show the plot
    plt.tight_layout()
    plt.show()

# c) Now we create make_it for each trick 1-4
def add_make_columns(datafile):
    # Read the data
    df = pd.read_csv(datafile)
    
    # Loop through each trick column
    for i in range(1, 4):
        trick_column = f"trick {i}"
        
        # Assuming a trick is executed if its score > 0
        df[f"make {i}"] = df[trick_column].apply(lambda x: 1 if x > 0 else 0)
        
    # Save the modified dataframe back to the same CSV file (or to a new file if desired)
    df.to_csv(datafile, index=False)


# d) Given that they make a trick estimae the probability of them getting a score thats higher then 0.6
def estimate_probabilities(datafile):
    df = pd.read_csv(datafile)
    
    # Antag att varje rad representerar en skateboardåkare
    results = []
    for index, row in df.iterrows():
        tricks = [f"trick {i}" for i in range(1, 4)]
        
        # Count the amount of sucessfull and unsuccessful trick.
        successful_tricks = sum(1 for trick in tricks if row[trick] !=0)
        unsuccessful_tricks = sum(1 for trick in tricks if row[trick] == 0)
        
        # Count the amount of sucessfull scored more than 0.6
        more_than=sum(1 for trick in tricks if row[trick] >=0.6)
        
        # Use the avarage to judge the probability
        prob_success = more_than / (successful_tricks)
        prob_failure = 1 - prob_success
        
        results.append((index, prob_success, prob_failure))
    
    return results

probabilities = estimate_probabilities("Datafil_normalized.csv")
for index, success, failure in probabilities:
    print(f"Skateboardåkare {index+1}: P(success) = {success:.2f}, P(failure) = {failure:.2f}")

# a) First, normalise and save your data "Liam said its cirrect"
normalize_scores("Datafil.csv", "Datafil_normalized.csv")

# b) Plot histograms for trick 1 to 4 "Liam said its cirrect"
plot_histograms("Datafil_normalized.csv")

# c) Now we create make_it for each trick 1-4
add_make_columns("Datafil_normalized.csv")

# d) Given that they make a trick estimae the probability of them getting a score thats higher then 0.6

# Mozes koristiti aritmeticku sredinu...
# Ptrea je koristila medelvärde kao skattning za theta.







