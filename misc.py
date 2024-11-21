import numpy as np
import matplotlib.pyplot as plt
from dynamics import dynamics_ode, dynamics_resnet

def plot_datasets(train_data, test_data, title_train="Train Data", title_test="Test Data"):

    X_train, y_train = train_data
    X_test, y_test = test_data
    
    # Create subplots
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))

    ax[0].scatter(X_train[y_train == 0][:, 0], X_train[y_train == 0][:, 1], color='blue', label='Class 0')
    ax[0].scatter(X_train[y_train == 1][:, 0], X_train[y_train == 1][:, 1], color='red', label='Class 1')
    ax[0].set_title(title_train)
    ax[0].set_xlabel('Feature 1')
    ax[0].set_ylabel('Feature 2')
    ax[0].legend()
    
    # Plot test data
    ax[1].scatter(X_test[y_test == 0][:, 0], X_test[y_test == 0][:, 1], color='blue', label='Class 0')
    ax[1].scatter(X_test[y_test == 1][:, 0], X_test[y_test == 1][:, 1], color='red', label='Class 1')
    ax[1].set_title(title_test)
    ax[1].set_xlabel('Feature 1')
    ax[1].set_ylabel('Feature 2')
    ax[1].legend()
    
    # Show the plots
    plt.tight_layout()
    plt.show()


def plot_iterations(num_layers, T, y, WW, mu):
    '''Visualize the evolution of data points across each layer of the Network'''
    fig, ax = plt.subplots(1, num_layers+1, figsize=(6*(num_layers+1), 5))
    h=0.1
    for i in range(0,num_layers+1):
        Xi = np.array(T)[:, i, :]
        x_min, x_max = Xi[:,0].min()-1, Xi[:,0].max()+1
        y_min, y_max = Xi[:,1].min()-1, Xi[:,1].max()+1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        Z = clf_cls(np.c_[xx.ravel(), yy.ravel()], WW, mu)
        Z = Z.reshape(xx.shape)
        ax[i].contourf(xx, yy, Z, cmap="RdYlBu", alpha=0.5)
        ax[i].scatter(Xi[:,0], Xi[:,1], c=y, cmap="RdYlBu")
        ax[i].set_xticks([])
        ax[i].set_yticks([]);
        ax[i].set_title(f"Layer: {i}")


def plot_region(X, y, control, num_layers, dimension, alpha=1, model=""):
    '''Produces a single plot showing the overall decision boundary learned by the model'''
    fig, ax = plt.subplots(1, 1, figsize=(6*1, 5))
    h=0.1
    Xi = X
    x_min, x_max = Xi[:,0].min()-1, Xi[:,0].max()+1
    y_min, y_max = Xi[:,1].min()-1, Xi[:,1].max()+1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = clf(np.c_[xx.ravel(), yy.ravel()], control, alpha, model, num_layers, dimension)
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, cmap="RdYlBu", alpha=0.5)
    ax.scatter(Xi[:,0], Xi[:,1], c=y, cmap="RdYlBu")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(model, fontsize=12)
    ax.set_ylabel("Prediction", fontsize=12)


def plot_transformation(T, y, WW, mu, model=""):
    '''Visualize the effect of the final layer transformation on the data points'''
    fig, ax = plt.subplots(1, 1, figsize=(6*1, 5))
    h=0.1
    i = -2
    Xi = np.array(T)[:, i, :]
    x_min, x_max = Xi[:,0].min()-1, Xi[:,0].max()+1
    y_min, y_max = Xi[:,1].min()-1, Xi[:,1].max()+1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
    Z = clf_cls(np.c_[xx.ravel(), yy.ravel()], WW, mu)
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, cmap="RdYlBu", alpha=0.5)
    ax.scatter(Xi[:,0], Xi[:,1], c=y, cmap="RdYlBu")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(model, fontsize=12)
    ax.set_ylabel("Transformation", fontsize=12)


def get_cls(control, dimension):
        ix = dimension*(dimension+1)
        jx = -ix+dimension*dimension
        W = control[-ix:jx]
        mu = control[jx:]
        WW = np.array([
            [W[0], W[1]],
            [W[2], W[3]]
        ]) 
        return WW, mu

def clf_cls(x, WW, mu):
    o = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        xf = WW@x[i]+mu
        sxf = np.exp(xf)
        sxf /= sum(sxf)
        o[i] = sxf[1]
    return o


def clf(x, control, alpha, model, num_layers, dimension):
    if model == 'ODENet':
        O, _ = dynamics_ode(control, x, num_layers, dimension, alpha)
    elif model == 'ResNet':
        O, _ = dynamics_resnet(control, x, num_layers, dimension, alpha)
    return O


