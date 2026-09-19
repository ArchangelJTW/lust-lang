local function build(depth, value)
    if depth == 0 then
        return { value = value, left = nil, right = nil }
    end
    local l = build(depth - 1, value * 2)
    local r = build(depth - 1, value * 2 + 1)
    return { value = value, left = l, right = r }
end

local function sum(node)
    local s = node.value
    if node.left ~= nil then
        s = s + sum(node.left)
    end
    if node.right ~= nil then
        s = s + sum(node.right)
    end
    return s
end

local function count_even(arr)
    local c = 0
    local i = 1
    while i <= #arr do
        local v = arr[i]
        if v % 2 == 0 then
            c = c + 1
        end
        i = i + 1
    end
    return c
end

local tree = build(14, 1)
local total = 0
local pass = 0
while pass < 20 do
    total = total + sum(tree)
    pass = pass + 1
end
local arr = {}
local k = 0
while k < 1000 do
    arr[#arr + 1] = k * 3
    k = k + 1
end
local evens = 0
local j = 0
while j < 2000 do
    evens = evens + count_even(arr)
    j = j + 1
end
print(total + evens)
