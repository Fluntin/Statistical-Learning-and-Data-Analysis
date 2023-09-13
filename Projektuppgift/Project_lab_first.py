import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("Datafil.csv")

mean_score_run_1 = np.mean(df["run 1"])
print(mean_score_run_1)

# The rest of this part is correct
mean_run_1_100 = np.mean(df[0:100]["run 1"])

# Fix the equality comparison
vegas_mask = (df["location"] == "las vegas")
mean_run_1_vegas = np.mean(df[vegas_mask]["run 1"])

run_sums = df[["run 1", "run 2"]].sum(axis=1)
df["run sums"] = run_sums

cols = df.columns.tolist()

def f(x):
    return 2*x  # Fix the indentation

run_sums_times_two = df["run sums"].apply(f)

# Fix the concatenation
df = pd.concat([df[cols[:-1]], run_sums_times_two], axis=1)  # Keep all columns except the last one

Test = np.array([1,1,2,5,3,3,3,4,2])
print(np.sort(Test))

unique_values, counts = np.unique(Test, return_counts=True)
print(f"{unique_values=}")
print(f"{counts=}")

