import gym
import numpy as np
import random
import time
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
import heapq
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import seaborn as sns


def compare_algorithms(array):
    results = {}

    # Bubble Sort
    bubble_arr = array.copy()
    bubble_sorted, bubble_time = bubble_sort(bubble_arr)
    results['Bubble Sort'] = bubble_time

    # Merge Sort
    merge_arr = array.copy()
    start_time = time.time()
    merge_sorted = merge_sort(merge_arr)
    merge_time = time.time() - start_time
    results['Merge Sort'] = merge_time

    return results


def plot_results(results):
    sns.barplot(x=list(results.keys()), y=list(results.values()))
    plt.title('Comparison of Algorithm Efficiency')
    plt.xlabel('Algorithm')
    plt.ylabel('Time (seconds)')
    plt.show()


# Brute Force: Bubble Sort

def bubble_sort(arr):
    n = len(arr)
    start_time = time.time()
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    end_time = time.time()
    return arr, end_time - start_time


# Divide and Conquer: Merge Sort

def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return merge(left, right)


def merge(left, right):
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result


# Dynamic Programming: Fibonacci

def fibonacci(n):
    fib = [0, 1]
    for i in range(2, n + 1):
        fib.append(fib[i - 1] + fib[i - 2])
    return fib[n]


# Greedy Approach: Huffman Coding

def huffman_encoding(freq):
    heap = [[weight, [symbol, '']] for symbol, weight in freq.items()]
    heapq.heapify(heap)
    while len(heap) > 1:
        lo = heapq.heappop(heap)
        hi = heapq.heappop(heap)
        for pair in lo[1:]:
            pair[1] = '0' + pair[1]
        for pair in hi[1:]:
            pair[1] = '1' + pair[1]
        heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])
    return sorted(heapq.heappop(heap)[1:], key=lambda p: (len(p[-1]), p))


# Backtracking: N-Queens Problem

def n_queens(N):
    def solve(queens, xy_dif, xy_sum):
        p = len(queens)
        if p == N:
            result.append(queens)
            return None
        for q in range(N):
            if q not in queens and p - q not in xy_dif and p + q not in xy_sum:
                solve(queens + [q], xy_dif + [p - q], xy_sum + [p + q])

    result = []
    solve([], [], [])
    return result


# Reinforcement Learning - Sorting Environment
class SortingEnv(gym.Env):
    def __init__(self, array):
        super(SortingEnv, self).__init__()
        self.array = array
        self.action_space = gym.spaces.Discrete(len(array) - 1)
        self.observation_space = gym.spaces.Box(
            low=0, high=np.max(array), shape=(len(array),), dtype=np.int32)

    def reset(self):
        self.array = np.random.randint(0, 100, len(self.array))
        return self.array

    def step(self, action):
        reward = 0
        if self.array[action] > self.array[action + 1]:
            self.array[action], self.array[action +
                                           1] = self.array[action + 1], self.array[action]
            reward = 1
        done = np.all(self.array[:-1] <= self.array[1:])
        return self.array, reward, done, {}


def train_rl_agent(array):
    env = SortingEnv(array)
    model = PPO('MlpPolicy', env, verbose=0)
    model.learn(total_timesteps=1000)
    return model


if __name__ == "__main__":
    array_to_sort = np.random.randint(0, 100, 10)
    print("\nOriginal array:", array_to_sort)

    # Compare Algorithm Efficiency
    results = compare_algorithms(array_to_sort)
    plot_results(results)

    # Train Reinforcement Learning Agent
    model = train_rl_agent(array_to_sort)
    print("\nReinforcement Learning agent training completed.")
