let point = #{ x: 0, y: 0 };

for i in 1..=100000 {
    point.x = i;
    point.y += i;
}

print(point.x.to_string() + " " + point.y.to_string());
