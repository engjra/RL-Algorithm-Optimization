import gym
import numpy as np
import random
import time
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
import heapq
import matplotlib.pyplot as plt
import seaborn as sns
import math


def compare_algorithms(array):
    results = {}

    # Bubble Sort (Brute Force)
    bubble_arr = array.copy()
    bubble_sorted, bubble_time = bubble_sort(bubble_arr)
    results['Bubble Sort'] = bubble_time

    # Merge Sort (Divide and Conquer)
    merge_arr = array.copy()
    start_time = time.time()
    merge_sorted = merge_sort(merge_arr)
    merge_time = time.time() - start_time
    results['Merge Sort'] = merge_time

    # Fibonacci (Dynamic Programming) - Example for Performance Comparison
    n = 30  # You can change this value depending on your performance testing
    start_time = time.time()
    fibonacci_result = fibonacci(n)
    fibonacci_time = time.time() - start_time
    results['Fibonacci (DP)'] = fibonacci_time

    # Huffman Coding (Greedy Approach)
    start_time = time.time()
    huffman_result = huffman_coding("this is a test sentence")
    huffman_time = time.time() - start_time
    results['Huffman Coding'] = huffman_time

    # N-Queens Problem (Backtracking)
    n_queens_start_time = time.time()
    n_queens_result = n_queens(8)  # Solve the 8-Queens problem
    n_queens_time = time.time() - n_queens_start_time
    results['N-Queens Problem'] = n_queens_time

    return results


def plot_theoretical_complexities():
    # Generate a range of n values
    n_values = np.arange(1, 21)

    # Theoretical time complexities
    bubble_sort_complexity = n_values ** 2
    merge_sort_complexity = n_values * np.log2(n_values)
    fibonacci_complexity = 2 ** n_values

    # Approximate complexities for Huffman Coding (O(n)) and N-Queens (O(n!))
    huffman_complexity = n_values  # Linear complexity for Huffman Coding (O(n))
    n_queens_complexity = np.array([math.factorial(n) for n in n_values])  # Exponential complexity for N-Queens (O(n!))

    # Create a single plot
    plt.figure(figsize=(10, 6))

    # Plot all three complexities
    plt.plot(n_values, bubble_sort_complexity, label='Bubble Sort (O(n^2))', color='darkblue', linewidth=2)
    plt.plot(n_values, merge_sort_complexity, label='Merge Sort (O(n log n))', color='green', linewidth=2)
    plt.plot(n_values, fibonacci_complexity, label='Fibonacci (O(2^n))', color='red', linewidth=2)

    # Plot the approximations for Huffman and N-Queens
    plt.plot(n_values, huffman_complexity, label='Huffman Coding (O(n))', color='purple', linestyle='--', linewidth=2)
    plt.plot(n_values, n_queens_complexity, label='N-Queens (O(n!))', color='orange', linestyle='--', linewidth=2)

    # Set the title and labels
    plt.title('Comparison of Time Complexities', fontsize=14, fontweight='bold')
    plt.xlabel('Input Size (n)', fontsize=12)
    plt.ylabel('Time Complexity (Operations)', fontsize=12)

    # Set logarithmic scale for better visualization
    plt.yscale('log')

    # Show the legend
    plt.legend(fontsize=12)

    # Show the plot
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Show the time complexity graph
    plot_theoretical_complexities()
