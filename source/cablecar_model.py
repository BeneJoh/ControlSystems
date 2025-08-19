import numpy as np

def cablecar_ode(t, y, u, params): 
    # Read out inputs and parameters
    phi, omega, x, v = y[0], y[1], y[2], y[3]
    M, m, l, g, gammax, gammaphi = map(params.get, ['M', 'm', 'l', 'g', 'gammax', 'gammaphi'])
    F = u[0]

    # Compute derivatives
    dphi = omega
    dx = v
    dv = (omega**2 * np.sin(phi) * M * l + np.cos(phi) * (g * np.sin(phi)- v * omega * np.sin(phi)) * M + F - gammax * v)/ (M + m - np.cos(phi)**2 * M) # gammaphi missing 
    domega = (-g * np.sin(phi) + v * omega * np.sin(phi) - gammaphi * omega /(M * l)- np.cos(phi) * dv)/ ( l )
    return np.array([dphi, domega, dx, dv])

def cablecar_output(t, x, u, params): 
    return np.array([x[0], x[1], x[2], x[3]])  # return trolley position and velocity 


def cablecar_smallangle_ode(t, y, u, params): 
    # Read out inputs and parameters
    phi, omega, x, v = y[0], y[1], y[2], y[3]
    M, m, l, g, gammax, gammaphi = map(params.get, ['M', 'm', 'l', 'g', 'gammax', 'gammaphi'])
    F = u[0]

    # Compute derivatives
    dphi = omega
    dx = v
    dv =  (F - gammax * v  + M * g * phi + gammaphi * omega/ l ) / ( m )
    domega =  (-g * (1 - M/m) * phi - gammaphi * omega / l * (1 + M/(l * m)) - (F - v * omega)/ m) / (l)
    return np.array([dphi, domega, dx, dv])
