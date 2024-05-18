import os
import csv
import json
import shutil

with open('output.json') as json_file:
    data = json.load(json_file)

with open('output.csv', 'w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(['image', 'label'])
    for label, filenames in data.items():
        for filename in filenames:
            os.makedirs("custom-rvlcdip/" + label, exist_ok=True)
            new_path = "custom-rvlcdip/" + label + '/' + filename.split('/')[1]
            shutil.copy(filename, new_path)
            writer.writerow([new_path, label])

print('CSV file created successfully.')
