local s = ""
local i = 0
while i < 100000 do
    s = s .. tostring(i % 10)
    i = i + 1
end
print(#s)
