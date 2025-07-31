import numpy as np


# Single machine infinite bus system
class SmibSystem:
    def __init__(self, M, D, P, K, verbose=True):
        self.M = M # inertia
        self.D = D # damping
        self.P = P # power injection/consumption
        self.K = K # coupling constant
        if verbose: 
            print('SMIB system initialized with M={}, D={}, P={}, K={}'.format(M, D, P, K))

    def __call__(self, t, u):
        phi, omega = u
        dphi = omega
        domega = (self.P - self.D * omega - self.K * np.sin(phi)) / self.M
        return [dphi, domega]
    
    def jacobian(self, u, t):
        phi, omega = u
        J = np.array([[0, 1], [-self.K * np.cos(phi) / self.M, -self.D / self.M]])
        return J

# Two area model
class GeneralModel:
    def __init__(self, Ms, Ds, Ps, Kmatrix, events=np.array([])):
        self.N = len(Ms)
        # Check size of input arrays
        assert len(Ds) == self.N
        assert len(Ps) == self.N
        assert Kmatrix.shape == (self.N, self.N)
        # diagonals of Kmatrix must be zero
        assert np.all(np.diag(Kmatrix) == 0)
        # Set instance parameters
        self.Ms = Ms # inertias
        self.Ds = Ds # dampings
        self.Ps = Ps  # power injections    
        self.Kmatrix = np.array(Kmatrix) # coupling matrix 
        self.events = events

    def __call__(self, t, u):
        # Check for events
        self.check_events(t)
        # Get states 
        phis = u[0:(self.N)]
        omegas = u[self.N:]
        # Compute interactions
        interactions = np.array([np.sum([self.Kmatrix[i, j] * np.sin(phis[j] - phis[i]) for j in range(self.N)]) for i in range(self.N)])
        # Compute derivatives
        dphis = omegas.copy()
        domegas = (self.Ps - self.Ds * omegas + interactions) / self.Ms
        return np.concatenate((dphis, domegas))
    
    def jacobian(self, u, t):
        J = np.zeros((2 * self.N, 2 * self.N))
        # N is an even number 
        for i in range(0, self.N):
            J[2*i, 2*i+1] = 1
            for j in range(self.N):
                if i != j:
                    J[2*i+1, 2*j] = self.Kmatrix[i, j] * np.cos(u[2*j] - u[2*i]) / self.Ms[i] 
            J[2*i+1, 2*i+1] = -self.Ds[i] / self.Ms[i]
        return J 

    # Check for events at time t
    def check_events(self, t):
        occured_event_idx = None 
        for (i, event) in enumerate(self.events):
            if event['time'] <= t:
                if 'load_jump' in event:
                    print('Load jump event at time {}:'.format(t))
                    assert len(event['load_jump']) == len(self.Ps)
                    self.Ps += event['load_jump']
                if 'line_drop' in event:
                    print('Line drop event at time {}:'.format(t))
                    line_tuple = event['line_drop']
                    assert len(line_tuple) == 2 # must be a tuple 
                    # Set corresponding entries from coupling matrix to zero
                    self.Kmatrix[*event['line_drop']] = 0
                    self.Kmatrix[*event['line_drop'][::-1]] = 0
                occured_event_idx = i
        if occured_event_idx != None: 
            # Delete event from list
            self.events.pop(occured_event_idx)
                    