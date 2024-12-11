import dask.dataframe as dd
from dask.diagnostics import ProgressBar

# Enable ProgressBar from Dask
ProgressBar().register()

# Load KT1 and KT2 using Dask
kt1 = dd.read_csv("./data/kt1_combined_with_correctness.csv")
kt2 = dd.read_csv("./data/kt2.csv")

# Map source and platform to KT1
source_platform_mapping = kt2[kt2["action_type"] == "respond"][
    ["item_id", "source", "platform", "user_id"]
].drop_duplicates()

# Rename item_id to question_id for the merge
source_platform_mapping = source_platform_mapping.rename(
    columns={"item_id": "question_id"}
)
kt1 = kt1.merge(source_platform_mapping, on=["question_id", "user_id"], how="left")

# Select only the required columns
required_columns = [
    "timestamp",
    "question_id",
    "user_id",
    "elapsed_time",
    "source",
    "platform",
    "got_question_correct",
]
kt1 = kt1[required_columns]

# Filter rows with elapsed_time less than 3000000
kt1_filtered = kt1[kt1["elapsed_time"] < 3000000]

# Save the filtered DataFrame to a CSV file
kt1_filtered.to_csv("kt1_filtered.csv", single_file=True, index=False)
