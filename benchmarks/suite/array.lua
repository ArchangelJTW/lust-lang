local arr = {}
local i = 0
while i < 1000000 do
    arr[#arr + 1] = i
    i = i + 1
end
local total = 0
local pass = 0
while pass < 10 do
    local j = 1
    local n = #arr
    while j <= n do
        local v = arr[j]
        if v ~= nil then total = total + v end
        j = j + 1
    end
    pass = pass + 1
end
print(total)
