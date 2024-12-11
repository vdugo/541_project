from time import perf_counter
import dask.dataframe as dd

start_time = perf_counter()

# Load the DataFrame with Dask (using Dask Expressions by default)
df = dd.read_csv("./data/kt1_enriched_final.csv")

# Verify the type
print(
    f"Type of df: {type(df)}"
)  # Should print: <class 'dask.dataframe.core.DataFrame'>

# Display the head (this triggers a computation for the first few rows)
print(df.head(40))
# Display the tail
print(df.tail(40))

# Display schema information (non-computational with Dask)
print(df.info())

# Compute null value counts (requires computation)
null_counts = df.isnull().sum()
print(null_counts)

print(f"columns: {df.columns}")

end_time = perf_counter()

print(f"Took {end_time - start_time} seconds to load and analyze the dataframe")
