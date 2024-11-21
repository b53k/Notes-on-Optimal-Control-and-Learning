import numpy as np

activation_function = lambda x: np.tanh(x)

def dynamics_ode(control, X, num_layers, dimension, alpha = 1):
    '''Returns: 
        O: array of output for each input sample
        T: list of trajectories for each input sample. It contains all intermediate states x_1, x_2, ..., x_final across all layers'''
        
    O = []
    T = []
    for k in range(X.shape[0]):
        x0 = X[k]
        xn = x0.copy()
        x = [xn]

        # Simulate the forward-pass of a sample input to the NN; as a discrete approximation of an ODE:
        for i in range(num_layers):
            ix = (dimension * (dimension + 1) + 1) * i      # 0, 7, 14, 21, 28, 35, ....
            jx = ix + dimension * dimension                 # 4, 11, 18, 25, 32, 39, ....
            zx = jx + dimension                             # 6, 13, 20, 27, 34, 41, ....
            K = control[ix:jx]                              # Current linear-transformation params
            bias = control[jx:zx]                           # Bias associated with current layer
            Dt = control[zx]                                # Discrete time-step for each layer
            KK = np.array([
                [K[0], K[1]],
                [K[2], K[3]]
            ])

            # At each layer the output is obtained as: x_{n+1} = x_n + \alpha * Dt * \phi() where \alpha is scaling factor
            xn = (xn + alpha * Dt * activation_function(KK @ xn + bias)).copy()
            xn = xn
            x.append(xn)

        ix = dimension*(dimension+1)
        jx = -ix+dimension*dimension
        W = control[-ix:jx]
        mu = control[jx:]

        # WW maps the final state after all layers to the output space with mu as the output layer bias
        WW = np.array([
            [W[0], W[1]],
            [W[2], W[3]]
        ]) 
        xf = (WW@xn + mu).copy()
        x.append(xf)
        T.append(x)
        sxf = np.exp(xf)
        sxf /= sum(sxf)
        o = sxf[1]
        O.append(o)
    O = np.array(O)

    return O, T


def obj_ode(control, X, y, num_layers, dimension, alpha = 1):
    O, _ = dynamics_ode(control, X, num_layers, dimension, alpha)
    L = y[:O.shape[0]]
    Dts = []
                
    for i in range(num_layers):
        ix = (dimension*(dimension+1)+1)*i
        jx = ix+dimension*dimension
        zx = jx+dimension
        Dt = control[zx]
        Dts.append(Dt)

    Dts = np.array(Dts)

    return 0.5 * sum((O-L)*(O-L)) + 0.01*sum((1-2*((Dts>0)*(Dts<1)))>0)



def dynamics_resnet(control, X, num_layers, dimension, alpha=1):
    O = []
    T = []
    dt = 1/num_layers

    for k in range(X.shape[0]):
        x0 = X[k]
        xn = x0.copy()
        x = [xn]
        for i in range(num_layers):
            ix = dimension*(dimension+1)*i
            jx = ix+dimension*dimension
            zx = jx+dimension
            K = control[ix:jx]
            be = control[jx:zx]
            KK = np.array([
                [K[0], K[1]],
                [K[2], K[3]]
            ])
            xn = (xn + alpha*dt*activation_function(KK@xn + be)).copy()
            xn = xn
            x.append(xn)

        ix = dimension*(dimension+1)
        jx = -ix+dimension*dimension
        W = control[-ix:jx]
        mu = control[jx:]
        WW = np.array([
            [W[0], W[1]],
            [W[2], W[3]]
        ]) 
        xf = (WW@xn + mu).copy()
        x.append(xf)
        T.append(x)
        sxf = np.exp(xf)
        sxf /= sum(sxf)
        o = sxf[1]
        O.append(o)
    O = np.array(O)
    return O, T

def obj_resnet(control, X, y, num_layers, dimension, alpha=1):
    O, _ = dynamics_resnet(control, X, num_layers, dimension, alpha)
    L = y[:O.shape[0]]
    return 0.5*sum((O-L)*(O-L))