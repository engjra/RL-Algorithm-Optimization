# RL-Algorithm-Optimization

# Reinforcement Learning for Algorithm Optimization

This project demonstrates the use of various algorithmic techniques and Reinforcement Learning to optimize sorting algorithms. The code compares the efficiency of traditional sorting algorithms with a Reinforcement Learning-based approach.

## 📌 Features
- **Traditional Algorithms:**
  - Bubble Sort (Brute Force)
  - Merge Sort (Divide and Conquer)
  - Fibonacci (Dynamic Programming)
  - Huffman Coding (Greedy Approach)
  - N-Queens Problem (Backtracking)
- **Reinforcement Learning:**
  - Uses PPO (Proximal Policy Optimization) to train an agent to sort arrays.
  - Custom Gym environment for sorting tasks.

## 📂 Project Structure
- `main.py`: Contains all the implemented algorithms and RL model.

## 📥 Installation
Clone the repository:
```bash
 git clone https://github.com/engjengjra/RL-Algorithm-Optimization.git
 cd RL-Algorithm-Optimization
```

Install dependencies:
```bash
 pip install -r requirements.txt
```

## 🚀 Usage
To run the project:
```bash
 python main.py
```

## 📖 Code Explanation
### Sorting Algorithms
- **Bubble Sort:** Compares adjacent elements and swaps them if they are in the wrong order.
- **Merge Sort:** Divides the array into two halves, sorts each recursively, and merges them.

### Other Algorithms
- **Fibonacci Calculation (Dynamic Programming):** Efficiently calculates Fibonacci numbers.
- **Huffman Coding (Greedy Approach):** Generates optimal prefix codes for data compression.
- **N-Queens Problem (Backtracking):** Finds solutions to the N-Queens puzzle.

### Reinforcement Learning Model
- The model uses `stable_baselines3` with `PPO` to train an agent on sorting tasks.
- Custom Gym environment (`SortingEnv`) defined for sorting arrays using a reinforcement learning approach.

## 📊 Results & Analysis
- Comparison of traditional algorithms is plotted using `seaborn` and `matplotlib`.
- RL agent performance is demonstrated by sorting arrays efficiently after training.

## 🔍 Future Work
- Experimenting with different RL algorithms for enhanced sorting performance.
- Comparing performance with additional sorting algorithms.


