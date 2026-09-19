local point = { x = 0, y = 0 }
local i = 1
while i <= 10000000 do
    point.x = i
    point.y = point.y + i
    i = i + 1
end
print(point.y)
