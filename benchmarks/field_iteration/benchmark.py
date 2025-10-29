point = {"x": 0, "y": 0}

for i in range(1, 100001):
    point["x"] = i
    point["y"] += i

print(point["x"], point["y"])

