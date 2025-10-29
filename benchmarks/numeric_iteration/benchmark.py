#!/usr/bin/env python3
# Simple benchmark: sum numbers from 1 to N

def main():
    sum_val = 0
    i = 1
    n = 10_000_000  # 10 million iterations

    while i <= n:
        sum_val += i
        i += 1

    print(f"Sum: {sum_val}")

if __name__ == "__main__":
    main()

