def fibonacci(n):
    if n == 0:
        return []
    elif n == 1:
        return [0]
    dp = [0] * n
    dp[1] = 1
    for i in range(2, n):
        dp[i] = dp[i-1] + dp[i-2]
    return dp