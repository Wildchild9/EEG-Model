# This is a sample Python script.

# Press ⌘R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import os
import csv

path = './Data'

focused_files = []
unfocused_files = []

# Go through only the directories in the path
for subfolder in os.listdir(path):
    if subfolder not in ["Focused", "Unfocused"]:
        continue

    subfolder_path = os.path.join(path, subfolder)

    # Go through only the files in the subdirectories
    for file in os.listdir(subfolder_path):
        # ensure the file is a csv file
        if not file.endswith('.txt'):
            continue
        file_path = os.path.join(subfolder_path, file)
        # Check if subfolder is named "Focused" or "Unfocused"
        if subfolder == "Focused":
            focused_files.append(file_path)
        elif subfolder == "Unfocused":
            unfocused_files.append(file_path)




def file_to_array(file_path):
    lines = []
    with open(file_path, 'r') as file:
        for line in file:
            lines.append(line.strip())
    return lines


def combine_csv_files(file1_path, file2_path, output_path):
    arr1 = file_to_array(file1_path)
    arr2 = file_to_array(file2_path)
    # write the top five lines of the first file to the output file
    with open(output_path, 'w') as file:
        for i in range(5):
            file.write(arr1[i] + '\n')
        arr1 = arr1[5:]
        arr2 = arr2[5:]

        file1_rows = len(arr1)
        file2_rows = len(arr2)

        current_file = 1

        while file1_rows != 0 and file2_rows != 0:
            if current_file == 1:
                lines = min(6000, file1_rows)
                for i in range(lines):
                    file.write(arr1[i] + '\n')
                file1_rows -= lines
                arr1 = arr1[lines:]
                current_file = 2
            else:
                lines = min(6000, file2_rows)
                for i in range(lines):
                    file.write(arr2[i] + '\n')
                file2_rows -= lines
                arr2 = arr2[lines:]
                current_file = 1

        if file1_rows == 0:
            for line in arr2:
                file.write(line + '\n')
        else:
            for line in arr1:
                file.write(line + '\n')


file1 = focused_files[0]
file2 = unfocused_files[0]

# Print the number of lines in file1 and file2
with open(file1, 'r') as f:
    print('Number of lines in file1:', sum(1 for line in f))

with open(file2, 'r') as f:
    print('Number of lines in file2:', sum(1 for line in f))

combine_csv_files(file1, file2, 'combined.csv')

# Print the number of lines in the combined file
with open('combined.csv', 'r') as f:
    print('Number of lines in combined file:', sum(1 for line in f))