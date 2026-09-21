local N = 10000000
local flags = {}
local i = 0
while i <= N do
    flags[i] = true
    i = i + 1
end
local count = 0
i = 2
while i <= N do
    if flags[i] then
        count = count + 1
        local j = i * i
        while j <= N do
            flags[j] = false
            j = j + i
        end
    end
    i = i + 1
end
print(count)
