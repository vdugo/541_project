import pandas as pd

df = pd.read_csv("./")

# convert Unix timestamp to datetime
df['datetime'] = pd.to_datetime(df['timestamp'], unit='s') 

# derive elapsed time
first_timestamp = df['timestamp'].min() 
df['elapsed_time'] = df['timestamp'] - first_timestamp 

