import dask.dataframe as dd
from dask.diagnostics import ProgressBar

# Enable ProgressBar from Dask
ProgressBar().register()

# Load KT2 using Dask
kt2 = dd.read_csv("./data/kt2.csv")

# Replace null values in 'source' and 'platform' columns with "unknown"
kt2["source"] = kt2["source"].fillna("unknown")
kt2["platform"] = kt2["platform"].fillna("unknown")
