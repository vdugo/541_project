import dask.dataframe as dd
from dask.diagnostics import ProgressBar

# Enable ProgressBar from Dask
ProgressBar().register()

# File path
file_path = "./data/kt1_sampled_25_percent_no_user_answer.csv"

# Load the DataFrame
df = dd.read_csv(file_path)

# Sort the DataFrame by 'elapsed_time'
df_sorted = df.sort_values("elapsed_time")

# Compute and display the top 30 rows (smallest elapsed_time)
print(df_sorted.head(30))

# Compute and display the bottom 30 rows (largest elapsed_time)
print(df_sorted.tail(30))

print(f"len of dataframe: {len(df)}")
