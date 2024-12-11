import dask.dataframe as dd

# File path
file_path = "./data/kt2.csv"

# Read the first 100 rows into a DataFrame
df = dd.read_csv(file_path, nrows=100)

print(df)
