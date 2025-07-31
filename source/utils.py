import numpy as np

# Function to create a phase portrait of a given 2D problem, returns the grid and the vector field
def create_2Dphaseportrait(problem, x_lims, y_lims, t, N=100, normalize=True): 
    # Create a grid of points
    x = np.linspace(x_lims[0], x_lims[1], N)
    y = np.linspace(y_lims[0], y_lims[1], N)
    X, Y = np.meshgrid(x, y)
    
    # Compute the derivatives at each point
    U = np.zeros_like(X)
    V = np.zeros_like(Y)
    for i in range(N):
        for j in range(N):
            U[i, j], V[i, j] = problem(t, [X[i, j], Y[i, j]])
    # Normalize the vectors
    vec_mag = np.sqrt(U**2 + V**2)
    if normalize:
        U /= vec_mag; V /= vec_mag
    
    return X, Y, U, V

