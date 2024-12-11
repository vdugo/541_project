import os
import csv
from tqdm import tqdm

# Directory path
directory = "./data/KT2"
output_file = "./data/kt2.csv"

# List all .csv files in the directory
csv_files = [
    entry
    for entry in os.scandir(directory)
    if entry.name.endswith(".csv") and entry.is_file()
]

# Create and write to the output file
with open(output_file, mode="w", newline="", encoding="utf-8") as outfile:
    writer = None  # Initialize later to avoid repeating header

    # Iterate over files with a tqdm progress bar
    for entry in tqdm(csv_files, desc="Processing files", unit="file"):
        # Extract user_id from the file name
        user_id = entry.name.replace("u", "").replace(".csv", "")

        # Read the current file
        with open(entry.path, newline="", encoding="utf-8") as csvfile:
            reader = csv.reader(csvfile)
            headers = next(reader)  # Read the headers

            # Initialize writer and add the 'user_id' column in the header
            if writer is None:
                writer = csv.writer(outfile)
                writer.writerow(headers + ["user_id"])

            # Append rows from the current file to the output file
            for row in reader:
                writer.writerow(row + [user_id])

print(f"All interactions have been merged into '{output_file}'.")
