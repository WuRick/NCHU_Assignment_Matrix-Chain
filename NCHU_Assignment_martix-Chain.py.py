#!/usr/bin/env python
# coding: utf-8

# In[3]:


import time
import numpy as np
import matplotlib.pyplot as plt

# Algorithm 1: Brute-Force Algorithm
def matrix_chain_mult_brute_force(P, i, j):
    if i == j:
        return 0, f"A{i}"
    
    min_cost = float('inf')
    min_parenthesization = ""
    
    for k in range(i, j):
        cost_left, parenthesization_left = matrix_chain_mult_brute_force(P, i, k)
        cost_right, parenthesization_right = matrix_chain_mult_brute_force(P, k + 1, j)
        cost = cost_left + cost_right + P[i-1] * P[k] * P[j]
        
        if cost < min_cost:
            min_cost = cost
            min_parenthesization = f"({parenthesization_left}) x ({parenthesization_right})"
    
    return min_cost, min_parenthesization


# Algorithm 2: Dynamic Programming Algorithm
def matrix_chain_mult_dynamic(P):
    n = len(P) - 1
    
    m = np.full((n, n), np.inf, dtype=float)
    s = np.zeros((n, n), dtype=int)
    
    for i in range(n):
        m[i][i] = 0
    
    for l in range(2, n + 1):
        for i in range(n - l + 1):
            j = i + l - 1
            m[i][j] = np.inf
            
            for k in range(i, j):
                temp_cost = m[i][k] + m[k+1][j] + P[i-1] * P[k] * P[j]
                
                if temp_cost < m[i][j]:
                    m[i][j] = temp_cost
                    s[i][j] = k
    
    min_cost = m[0][n - 1]
    
    # Constructing the parenthesization iteratively
    stack = [(0, n - 1)]
    min_parenthesization = ""
    
    while stack:
        i, j = stack.pop()
        
        if i == j:
            min_parenthesization += f"A{i}"
        else:
            k = s[i][j]
            min_parenthesization += "("
            stack.append((k + 1, j))
            stack.append((i, k))
            min_parenthesization += ")"
            
            if stack:
                min_parenthesization += " x "
    
    return min_cost, min_parenthesization


# In[4]:


# Test the algorithms and measure running times
input_sizes = [5, 10, 15, 20]
running_times_brute_force = []
running_times_dynamic = []

for size in input_sizes:
    P = np.random.randint(1, 10, size + 1)
    
    start_time = time.time()
    matrix_chain_mult_brute_force(P, 1, size)
    running_times_brute_force.append(time.time() - start_time)
    
    start_time = time.time()
    matrix_chain_mult_dynamic(P)
    running_times_dynamic.append(time.time() - start_time)


# Generate the comparison graph
plt.plot(input_sizes, running_times_brute_force, label='Brute Force')
plt.plot(input_sizes, running_times_dynamic, label='Dynamic Programming')
plt.xlabel('Input Size')
plt.ylabel('Running Time (seconds)')
plt.legend


# In[ ]:




