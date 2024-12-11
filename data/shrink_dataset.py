import dask.dataframe as dd
from dask.diagnostics import ProgressBar

# Enable Dask progress bar
ProgressBar().register()

# Load the DataFrame
file_path = "./data/kt1_three_columns_only_95_million_rows.csv"
df = dd.read_csv(file_path)

# Randomly sample 25% of the data
# Set a random seed for reproducibility
sampled_df = df.sample(frac=0.25, random_state=42)

# Save the sampled data to a new file
output_file = "./data/kt1_sampled_25_percent.csv"
sampled_df.to_csv(output_file, single_file=True, index=False)

print(f"Sampled DataFrame saved to {output_file}")
