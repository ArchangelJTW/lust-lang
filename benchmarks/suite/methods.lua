local Counter = {}
Counter.__index = Counter
function Counter:bump(d)
    self.value = self.value + d
    return self.value
end
local c = setmetatable({ value = 0 }, Counter)
local last = 0
local i = 0
while i < 10000000 do
    last = c:bump(i)
    i = i + 1
end
print(last)
