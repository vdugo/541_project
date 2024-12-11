import dask.dataframe as dd
from dask.diagnostics import ProgressBar

# Enable Dask progress bar
ProgressBar().register()

# Load the DataFrame
df = dd.read_csv("./data/kt1_combined_with_correctness.csv")

# Drop the specified columns including 'Unnamed: 0' if they exist
columns_to_drop = [
    "timestamp",
    "solving_id",
    "question_id",
    "user_id",
    "correct_answer",
    "Unnamed: 0",
]
df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

# Write the cleaned DataFrame to a new CSV file
# By default, Dask will call pandas.to_csv for each partition
df.to_csv("./data/kt1_enriched_final_cleaned.csv", single_file=True, index=False)

print("Cleaned DataFrame saved to kt1_enriched_final_cleaned.csv")
