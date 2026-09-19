local function step(a, b)
    return a + b * 2
end
local acc = 0
local i = 0
while i < 10000000 do
    acc = step(acc, i)
    i = i + 1
end
print(acc)
