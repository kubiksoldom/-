with open("fills_all.csv", "rb") as f:
    for i in range(5):
        line = f.readline()
        print(i, line)
