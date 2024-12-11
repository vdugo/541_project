import numpy as np
import dask.dataframe as dd
from dask.diagnostics import ProgressBar

# Enable the progress bar
ProgressBar().register()

# Load the data
df = dd.read_csv("./data/kt1_enriched_final_cleaned.csv")

# Filter out rows where elapsed_time is 3000000 or above
df_cleaned = df[df["elapsed_time"] < 3000000]

# Write the cleaned DataFrame back to the same file
df_cleaned.to_csv(
    "./data/kt1_enriched_final_cleaned.csv", single_file=True, index=False
)

print(
    "Rows with elapsed_time >= 3000000 have been removed and saved back to the same file."
)
