import random
import math
import argparse
import simdlib
import numpy as np
import time

parser = argparse.ArgumentParser()
parser.add_argument("--naive", help="Benchmark naive implementation", action="store_true")
parser.add_argument("--optimized", help="Benchmark SIMD implementation", action="store_true")
parser.add_argument("--numpy", help="Benchmark NumPy implementation", action="store_true")

args = parser.parse_args()


# simulates stock prices using geometric Brownian motion (log-normal random walk).
def generate_stock_prices(
    start_price=100.0, 
    num_points=1_000_000, 
    drift=0.0002, 
    volatility=0.01, 
    seed=None
):
    if seed is not None:
        random.seed(seed)

    prices = [start_price]
    for _ in range(1, num_points):
        # Generate a normally distributed return using Box-Muller transform
        u1, u2 = random.random(), random.random()
        z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2 * math.pi * u2)
        return_pct = drift + volatility * z

        # Geometric Brownian motion step
        new_price = prices[-1] * math.exp(return_pct)
        prices.append(new_price)

    return prices


# calculate sharpe ratio from list of prices
def sharpe_ratio(prices, risk_free_rate=0.0):
    if len(prices) < 2:
        return 0.0

    log_returns = []
    for i in range(1, len(prices)):
        p1, p2 = prices[i - 1], prices[i]
        if p1 <= 0 or p2 <= 0:
            continue
        log_return = math.log(p2 / p1)
        log_returns.append(log_return)

    if not log_returns:
        return 0.0

    length = len(log_returns)
    if args.naive:
      mean_return = sum(log_returns) / length
    if args.optimized:
      mean_return = simdlib.sum_list(log_returns) / length
    if args.numpy:
      mean_return = np.sum(log_returns) / length

    if args.naive:
      variance = sum((r - mean_return) ** 2 for r in log_returns) / length
    if args.optimized:
      variance = simdlib.sum_list(simdlib.subtract_and_square_each_list(log_returns, mean_return))
    if args.numpy:
      log_returns = np.array(log_returns)
      variance = np.mean((log_returns - mean_return) ** 2)

    std_dev = math.sqrt(variance)

    if std_dev == 0:
        return 0.0
    return (mean_return - risk_free_rate) / std_dev

if __name__ == "__main__":
    input = generate_stock_prices()
    times = []
    for _ in range(50):
      start = time.time()
      sharpe_ratio(input)
      times.append(time.time() - start)
    print(sum(times) / len(times))