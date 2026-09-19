local total = 0
local i = 0
while i < 3000 do
    local j = 0
    while j < 3000 do
        total = total + (i * j) % 7
        j = j + 1
    end
    i = i + 1
end
print(total)
