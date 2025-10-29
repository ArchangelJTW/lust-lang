local function main()
    local sum = 0
    local i = 1
    local n = 10000000

    while i <= n do
        sum = sum + i
        i = i + 1
    end

    print("Sum: " .. sum)
end

main()
