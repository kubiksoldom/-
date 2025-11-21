import csv

with open("fills_all.csv", "r", encoding="utf-8") as f:
    reader = csv.reader(f)
    for i, row in enumerate(reader):
        print(i, row)
        if i == 10:
            break
