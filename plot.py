import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']

def plot_LP(A, b, c, steps, max_scale=100):
    assert A.shape[1] == 2, "Cannot plot if it is no 2D"
    plt.figure(figsize=(8, 6))
    
    # plot constraints
    for i in range(A.shape[0]):
        if A[i][0] == 0:
            y = np.full(1000, b[i] / A[i][1])
            x = np.linspace(0, max_scale, 1000)
        elif A[i][1] == 0:
            x = np.full(1000, b[i] / A[i][0])
            y = np.linspace(0, max_scale, 1000)
        else:
            x = np.linspace(0, max_scale, 1000)
            y = (b[i] - A[i][0]*x) / A[i][1]
        plt.plot(x, y, 'k')
        
    # plot steps of the algorithm
    x_iter, y_iter, *_ = zip(*steps)
    plt.plot(x_iter, y_iter, marker='o', color='red')
    
    # plot objective function
    plt.quiver(0, 0, -c[0], -c[1], angles='xy', scale_units='xy', scale=2, color='green', label='Objective Function Direction')

    plt.axhline(0, color='black', linewidth=2, linestyle='--')
    plt.axvline(0, color='black', linewidth=2, linestyle='--')
    plt.xlabel('x-axis')
    plt.ylabel('y-axis')
    plt.title('LP Problem')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    
def plot_LMOLP(A, b, objectives, steps, max_scale=100):
    assert A.shape[1] == 2, "Cannot plot if it is no 2D"
    plt.figure(figsize=(8, 6))
    
    # plot constraints
    for i in range(A.shape[0]):
        if A[i][0] == 0:
            y = np.full(1000, b[i] / A[i][1])
            x = np.linspace(0, max_scale, 1000)
        elif A[i][1] == 0:
            x = np.full(1000, b[i] / A[i][0])
            y = np.linspace(0, max_scale, 1000)
        else:
            x = np.linspace(0, max_scale, 1000)
            y = (b[i] - A[i][0]*x) / A[i][1]
        plt.plot(x, y, 'k')
        
    # plot steps of the algorithm
    x_iter, y_iter, *_ = zip(*steps)
    plt.plot(x_iter, y_iter, marker='o', color='red')
    
    # plot objective function
    for index, obj in enumerate(objectives, 1):
        c = obj['c']
        Q = obj['Q']
        if obj['Q'] is None:
            plt.quiver(0, 0, -c[0], -c[1], 
                       angles='xy', 
                       scale_units='xy', 
                       scale=2, 
                       color=colors[index-1], 
                       label=f'Objective Function {index}')
        else:
            # quadratic objective function
            quadratic_part = lambda x, y: 0.5 * (Q[0, 0] * x**2 + (Q[0, 1] + Q[1, 0]) * x * y + Q[1, 1] * y**2) + c[0] * x + c[1] * y
            x_range = np.linspace(0, max_scale, 1000)
            y_range = np.linspace(0, max_scale, 1000)
            X, Y = np.meshgrid(x_range, y_range)
            Z = quadratic_part(X, Y)
            plt.contour(X, Y, Z, levels=40, cmap='viridis') 

    plt.axhline(0, color='black', linewidth=2, linestyle='--')
    plt.axvline(0, color='black', linewidth=2, linestyle='--')
    plt.xlabel('x-axis')
    plt.ylabel('y-axis')
    plt.title('LP Problem')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_QP(A, b, c, Q, steps, max_scale=100):
    assert A.shape[1] == 2, "Cannot plot if it is no 2D"
    plt.figure(figsize=(8, 6))
    
    # quadratic objective function
    quadratic_part = lambda x, y: 0.5 * (Q[0, 0] * x**2 + (Q[0, 1] + Q[1, 0]) * x * y + Q[1, 1] * y**2) + c[0] * x + c[1] * y
    x_range = np.linspace(0, max_scale, 1000)
    y_range = np.linspace(0, max_scale, 1000)
    X, Y = np.meshgrid(x_range, y_range)
    Z = quadratic_part(X, Y)
    plt.contour(X, Y, Z, levels=40, cmap='viridis') 
    
    # plot constraints
    for i in range(A.shape[0]):
        if A[i][0] == 0:
            y = np.full(1000, b[i] / A[i][1])
            x = np.linspace(0, max_scale, 1000)
        elif A[i][1] == 0:
            x = np.full(1000, b[i] / A[i][0])
            y = np.linspace(0, max_scale, 1000)
        else:
            x = np.linspace(0, max_scale, 1000)
            y = (b[i] - A[i][0]*x) / A[i][1]
        plt.plot(x, y, 'k')
        
    # plot steps of the algorithm
    x_iter, y_iter, *_ = zip(*steps)
    plt.plot(x_iter, y_iter, marker='o', color='red')
    

    plt.axhline(0, color='black', linewidth=2, linestyle='--')
    plt.axvline(0, color='black', linewidth=2, linestyle='--')
    plt.xlabel('x-axis')
    plt.ylabel('y-axis')
    plt.title('QP Problem')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    
def plot_LMOLP_3d(A, b, objectives, steps, max_scale=100):
    assert A.shape[1] == 3, "Can only plot 3D problems"
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot constraints
    for i in range(A.shape[0]):
        x = np.linspace(0, max_scale, 100)
        y = np.linspace(0, max_scale, 100)
        X, Y = np.meshgrid(x, y)
        Z = (b[i] - A[i][0]*X - A[i][1]*Y) / A[i][2]
        ax.plot_surface(X, Y, Z, alpha=0.5, color='gray')

    # Plot steps of the algorithm
    x_iter, y_iter, z_iter, *_ = zip(*steps)
    ax.scatter(x_iter, y_iter, z_iter, marker='o', color='red')

    # Plot objective functions
    for index, obj in enumerate(objectives, 1):
        c = obj['c']
        Q = obj['Q']
        if obj['Q'] is None:
            ax.quiver(0, 0, 0, -c[0], -c[1], -c[2], color=colors[index-1], label=f'Objective Function {index}')
        else:
            # Quadratic objective function
            quadratic_part = lambda x, y: 0.5 * (Q[0, 0] * x**2 + Q[1, 1] * y**2) + c[0] * x + c[1] * y
            x_range = np.linspace(0, max_scale, 100)
            y_range = np.linspace(0, max_scale, 100)
            X, Y = np.meshgrid(x_range, y_range)
            Z = quadratic_part(X, Y)
            ax.plot_surface(X, Y, Z, alpha=0.5, cmap='viridis')

    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    ax.set_title('LMOLP Problem (3D)')
    plt.legend()
    plt.show()