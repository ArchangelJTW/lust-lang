local x = 0.0
local acc = 0.0
local i = 0
while i < 10000000 do
    x = x + 0.001
    acc = acc + x * x - acc * 0.5
    i = i + 1
end
print(acc)
