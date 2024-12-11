import dask.dataframe as dd
from dask.diagnostics import ProgressBar

# Enable Dask progress bar
ProgressBar().register()

# Load the DataFrame
df = dd.read_csv("./data/kt1_three_columns_only_95_million_rows.csv")

# Drop the specified columns including 'Unnamed: 0' if they exist
columns_to_drop = ["user_answer"]
df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

# Write the cleaned DataFrame to a new CSV file
# By default, Dask will call pandas.to_csv for each partition
df.to_csv("./data/kt1_two_columns_95_million_rows.csv", single_file=True, index=False)

print("Cleaned DataFrame saved kt1_two_columns_95_million_rows.csv")
