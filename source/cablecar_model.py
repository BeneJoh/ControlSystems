import numpy as np



def cablecar_ode(t, y, u, params): 
    # Read out inputs and parameters
    phi, omega, x, v = y[0], y[1], y[2], y[3]
    M, m1, m2, l, g, b1, b2, r = map(params.get, ['M', 'm1', 'm2', 'l', 'g', 'b1', 'b2', 'r'])
    F = u[0]

    # Define some quantities
    B = m1 +  m2/2
    C = m1 + m2 + M
    D = 2/5 * m1 * r**2
    eps1 = F - b2 * v + omega**2 * l * B * np.sin(phi)
    eps2 = b1 * omega + g * l * B * np.sin(phi)
    eps3 = l**2 * B**2 * (np.cos(phi))**2 - l**2 * B * C - D * C 

    # Compute derivatives
    dphi = omega
    domega = (l * B * np.cos(phi) * eps1 + C * eps2)/ eps3
    dx = v
    dv = ((l**2 * B + D) * eps1 + (l * B * np.cos(phi)) * eps2) / (-eps3)
    return np.array([dphi, domega, dx, dv])

def cablecar_output(t, x, u, params): 
    return np.array([x[0], x[1], x[2], x[3]])  # return trolley position and velocity 

# Two area model
# class GeneralModel:
#     def __init__(self, Ms, Ds, Ps, Kmatrix, events=np.array([])):
#         self.N = len(Ms)
#         # Check size of input arrays
#         assert len(Ds) == self.N
#         assert len(Ps) == self.N
#         assert Kmatrix.shape == (self.N, self.N)
#         # diagonals of Kmatrix must be zero
#         assert np.all(np.diag(Kmatrix) == 0)
#         # Set instance parameters
#         self.Ms = Ms # inertias
#         self.Ds = Ds # dampings
#         self.Ps = Ps  # power injections    
#         self.Kmatrix = np.array(Kmatrix) # coupling matrix 
#         self.events = events

#     def __call__(self, t, u):
#         # Check for events
#         self.check_events(t)
#         # Get states 
#         phis = u[0:(self.N)]
#         omegas = u[self.N:]
#         # Compute interactions
#         interactions = np.array([np.sum([self.Kmatrix[i, j] * np.sin(phis[j] - phis[i]) for j in range(self.N)]) for i in range(self.N)])
#         # Compute derivatives
#         dphis = omegas.copy()
#         domegas = (self.Ps - self.Ds * omegas + interactions) / self.Ms
#         return np.concatenate((dphis, domegas))
    
#     def jacobian(self, u, t):
#         J = np.zeros((2 * self.N, 2 * self.N))
#         # N is an even number 
#         for i in range(0, self.N):
#             J[2*i, 2*i+1] = 1
#             for j in range(self.N):
#                 if i != j:
#                     J[2*i+1, 2*j] = self.Kmatrix[i, j] * np.cos(u[2*j] - u[2*i]) / self.Ms[i] 
#             J[2*i+1, 2*i+1] = -self.Ds[i] / self.Ms[i]
#         return J 

#     # Check for events at time t
#     def check_events(self, t):
#         occured_event_idx = None 
#         for (i, event) in enumerate(self.events):
#             if event['time'] <= t:
#                 if 'load_jump' in event:
#                     print('Load jump event at time {}:'.format(t))
#                     assert len(event['load_jump']) == len(self.Ps)
#                     self.Ps += event['load_jump']
#                 if 'line_drop' in event:
#                     print('Line drop event at time {}:'.format(t))
#                     line_tuple = event['line_drop']
#                     assert len(line_tuple) == 2 # must be a tuple 
#                     # Set corresponding entries from coupling matrix to zero
#                     self.Kmatrix[*event['line_drop']] = 0
#                     self.Kmatrix[*event['line_drop'][::-1]] = 0
#                 occured_event_idx = i
#         if occured_event_idx != None: 
#             # Delete event from list
#             self.events.pop(occured_event_idx)
                    