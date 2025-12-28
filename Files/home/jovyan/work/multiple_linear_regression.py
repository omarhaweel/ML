import copy
import math
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('./deeplearning.mplstyle')
np.set_printoptions(precision=2)  # reduced display precision on numpy arrays

# Training data
X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
y_train = np.array([460, 232, 178])


def compute_cost(X, y, w, b):
    """
    Compute the cost function for multiple linear regression.
    
    Args:
        X (ndarray): Training examples (m, n)
        y (ndarray): Target values (m,)
        w (ndarray): Model parameters (n,)
        b (float): Bias parameter
        
    Returns:
        float: Cost value
    """
    m = X.shape[0]
    cost = 0
    for i in range(m):
        f_wb = X[i].dot(w) + b
        cost = cost + (f_wb - y[i]) ** 2
    cost = cost / (2 * m)
    return cost


def compute_gradient(X, y, w, b):
    """
    Compute the gradient for multiple linear regression.
    
    Args:
        X (ndarray): Training examples (m, n)
        y (ndarray): Target values (m,)
        w (ndarray): Model parameters (n,)
        b (float): Bias parameter
        
    Returns:
        tuple: (dj_db, dj_dw) - gradients with respect to b and w
    """
    m, n = X.shape  # (number of examples, number of features)
    dj_dw = np.zeros((n,))
    dj_db = 0.0
    
    for i in range(m):
        err = (np.dot(X[i], w) + b) - y[i]
        for j in range(n):
            dj_dw[j] = dj_dw[j] + err * X[i, j]
        dj_db = dj_db + err
    
    dj_dw = dj_dw / m
    dj_db = dj_db / m
    return dj_db, dj_dw


def gradient_descent(X, y, w_in, b_in, alpha, num_iters, cost_function, gradient_function):
    """
    Perform gradient descent to fit w and b.
    
    Args:
        X (ndarray): Training examples (m, n)
        y (ndarray): Target values (m,)
        w_in (ndarray): Initial model parameters (n,)
        b_in (float): Initial bias parameter
        alpha (float): Learning rate
        num_iters (int): Number of iterations
        cost_function: Function to compute cost
        gradient_function: Function to compute gradient
        
    Returns:
        tuple: (w, b, J_history) - final parameters and cost history
    """
    J_history = []
    w = copy.deepcopy(w_in)
    b = b_in
    
    for i in range(num_iters):
        dj_db, dj_dw = gradient_function(X, y, w, b)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
        
        J_history.append(cost_function(X, y, w, b))
        
        if i % math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i:4}: Cost {J_history[-1]:0.2e}")
    
    return w, b, J_history


# Initialize parameters
initial_w = np.zeros(X_train.shape[1])
initial_b = 0.0

# Gradient descent settings
iterations = 1000
alpha = 5.0e-7

# Run gradient descent
w_final, b_final, J_hist = gradient_descent(
    X_train, y_train, initial_w, initial_b,
    alpha, iterations, compute_cost, compute_gradient
)

print(f"\nb, w found by gradient descent: {b_final:0.2f}, {w_final}")

# Display predictions
m, _ = X_train.shape
print("\nPredictions vs target values:")
for i in range(m):
    prediction = np.dot(X_train[i], w_final) + b_final
    print(f"  Prediction: {prediction:0.2f}, Target: {y_train[i]}")

# Plot cost versus iteration
fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12, 4))

ax1.plot(J_hist)
ax1.set_title("Cost vs. iteration")
ax1.set_ylabel('Cost')
ax1.set_xlabel('Iteration step')

ax2.plot(100 + np.arange(len(J_hist[100:])), J_hist[100:])
ax2.set_title("Cost vs. iteration (tail)")
ax2.set_ylabel('Cost')
ax2.set_xlabel('Iteration step')

plt.show()
