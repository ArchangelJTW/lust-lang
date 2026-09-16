local total, i = 0, 0
while i < 50000000 do total = total + (i % 7); i = i + 1 end
print(total)
