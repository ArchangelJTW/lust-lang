local point = { x = 0, y = 0 }

for i = 1, 100000 do
	point.x = i
	point.y = point.y + i
end

print(point.x, point.y)
