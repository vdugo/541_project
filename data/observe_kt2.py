import os
import csv

# Directory path
directory = "./data/KT2"

try:
    # Open the directory iterator
    with os.scandir(directory) as entries:
        # Use a generator to fetch the first 10 .csv entries
        csv_count = 0
        for entry in entries:
            if csv_count < 10 and entry.name.endswith(".csv") and entry.is_file():
                print(f"\nReading file: {entry.name}")
                csv_count += 1
                # Open and read the CSV file
                with open(entry.path, newline="", encoding="utf-8") as csvfile:
                    reader = csv.reader(csvfile)
                    for row in reader:
                        print(row)
            if csv_count == 10:
                break
except FileNotFoundError:
    print(f"The directory '{directory}' does not exist.")
except PermissionError:
    print(f"Permission denied for accessing the directory '{directory}'.")
