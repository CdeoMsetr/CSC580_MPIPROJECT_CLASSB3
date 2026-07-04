
import csv, random
with open("dataset/dataset.csv","w",newline="") as f:
    w=csv.writer(f)
    w.writerow(["value"])
    for _ in range(100000):
        w.writerow([random.uniform(0,1000)])
print("dataset generated")
