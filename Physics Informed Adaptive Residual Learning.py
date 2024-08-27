import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata
from pyDOE import lhs
import time
import os
import pandas as pd
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

datarho = pd.read_csv('//home//naman//Rui Jiang Model//NGSIM_US101_Density_Data.csv')
datau = pd.read_csv('//home//naman//Rui Jiang Model//NGSIM_US101_Velocity_Data.csv')

Exactrho = np.real(datarho).T
Exactu = np.real(datau).T
Exactu = Exactu/65
Exactrho = Exactrho/0.30

layers = [2, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 2]
# percent = 0.25
percent = 0.5
# percent = .75
# percent = 0.9
num = 1
input(str(num)+' Enter')
N_u = int(percent*(540*(num+2)+104))
N_f = 40000
uxn = 104
xlo = 0.
xhi = 2060

utn = 540
tlo = 0.
thi = 2695
x = np.linspace(xlo, xhi, uxn)
t = np.linspace(tlo, thi, utn)

X, T = np.meshgrid(x, t)
X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
rho_star = Exactrho.flatten()[:, None]
u_star = Exactu.flatten()[:, None]
                                  
######################## Eulerian Training Data #################################
X, T = np.meshgrid(x, t)
X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
rho_star = Exactrho.flatten()[:, None]
u_star = Exactu.flatten()[:, None]
######################## Eulerian Training Data #################################

xx1 = np.hstack((X[0:1, :].T, T[0:1, :].T))
xx2 = np.hstack((X[:, 0:1], T[:, 0:1]))
xx3 = np.hstack((X[:, -1:], T[:, -1:]))
xx4 = np.hstack((X[:, 15:16], T[:, 15:16]))
xx5 = np.hstack((X[:, 90:91], T[:, 90:91]))
xx6 = np.hstack((X[:, 25:26], T[:, 25:26]))
xx7 = np.hstack((X[:, 75:76], T[:, 75:76]))

rho1 = Exactrho[0:1, :].T
rho2 = Exactrho[:, 0:1]
rho3 = Exactrho[:,-1:]
rho4 = Exactrho[:, 15:16]
rho5 = Exactrho[:, 90:91]
rho6 = Exactrho[:, 25:26]
rho7 = Exactrho[:, 75:76]

u1 = Exactu[0:1, :].T
u2 = Exactu[:, 0:1]
u3 = Exactu[:,-1:]
u4 = Exactu[:, 15:16]
u5 = Exactu[:, 90:91]
u6 = Exactu[:, 25:26]
u7 = Exactu[:, 75:76]
X_u_train1 = np.vstack([xx1, xx2, xx3, xx6])#, xx7, xx4, xx5])#, xx5])#, xx7, xx4, xx5])#xx4])#, xx5, xx6, xx7])
rho_train1 = np.vstack([rho1, rho2, rho3, rho6])#, rho7, rho4, rho5])#, rho5])#, rho7, rho4, rho5])#rho4])#, rho5, rho6, rho7])
u_train1 = np.vstack([u1, u2, u3, u6])#, u7, u4, u5])#, u5])#, u7, u4, u5])# u4])#, u5, u6, u7])
idx1 = np.random.choice(X_u_train1.shape[0], N_u, replace=False)
X_u_train = X_u_train1[idx1, :]
u_train = u_train1[idx1, :]
rho_train = rho_train1[idx1,:]
############################## Collocation Points ################################

lb = X_star.min(0)
ub = X_star.max(0)
X_f_train = lb + (ub - lb) * lhs(2, N_f)
X_f_train = np.vstack((X_f_train, X_u_train))

se = 1234
np.random.seed(se)
tf.set_random_seed(se)
u_f = 1
rho_m = 1
tau = 10#2
c0 = 30/65

class PhysicsInformedNN:
    def __init__(self, X_u, rho, u, X_f, layers, lb, ub):

        self.lb = lb
        self.ub = ub
        self.x_u = X_u[:,0:1]
        # self.x_u2 = X_u2[:,0:1]
        # self.x_u3 = X_u3[:,0:1]
        self.t_u= X_u[:,1:2]
        # self.t_u2 = X_u2[:,1:2]
        # self.t_u3 = X_u3[:,1:2]
        self.x_f = X_f[:,0:1]
        self.t_f = X_f[:,1:2]
        self.u = u
        self.rho = rho
        self.layers = layers
        self.weights, self.biases, self.skip_weights = self.initialize_NN(layers)
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))

        self.x_u_tf = tf.placeholder(tf.float32, shape=[None, self.x_u.shape[1]])
        # self.x2_u_tf = tf.placeholder(tf.float32, shape=[None, self.x_u2.shape[1]])
        # self.x3_u_tf = tf.placeholder(tf.float32, shape=[None, self.x_u3.shape[1]])
        self.t_u_tf = tf.placeholder(tf.float32, shape=[None, self.t_u.shape[1]])
        # self.t2_u_tf = tf.placeholder(tf.float32, shape=[None, self.t_u2.shape[1]])
        # self.t3_u_tf = tf.placeholder(tf.float32, shape=[None, self.t_u2.shape[1]])
        self.u_tf   = tf.placeholder(tf.float32, shape=[None, self.u.shape[1]])
        self.rho_tf = tf.placeholder(tf.float32, shape=[None, self.rho.shape[1]])
        self.x_f_tf = tf.placeholder(tf.float32, shape=[None, self.x_f.shape[1]])
        self.t_f_tf = tf.placeholder(tf.float32, shape=[None, self.t_f.shape[1]])
        self.Y_pred = self.net_u(self.x_u_tf, self.t_u_tf)
        # self.Y2_pred = self.net_u(self.x2_u_tf, self.t2_u_tf)
        # self.Y3_pred = self.net_u(self.x3_u_tf, self.t3_u_tf)
        self.rho_pred = self.Y_pred[:,0:1]
        # self.rho2_pred = self.Y2_pred[:,0:1]
        # self.rho3_pred = self.Y3_pred[:,0:1]
        self.u_pred = self.Y_pred[:,1:2] 
        # self.u2_pred = self.Y2_pred[:,1:2]
        # self.u3_pred = self.Y3_pred[:,1:2]
        self.f_pred = self.net_f(self.x_f_tf, self.t_f_tf)
        self.f1 = self.f_pred[0]
        self.f2 = self.f_pred[1]

        self.loss = tf.reduce_mean(tf.square(self.u_tf - self.u_pred)) + tf.reduce_mean(tf.square(self.rho_tf - self.rho_pred)) + tf.reduce_mean(tf.square(self.f1)) + tf.reduce_mean(tf.square(self.f2))# + tf.reduce_mean(tf.square(self.rho2_pred-self.rho3_pred)) + tf.reduce_mean(tf.square(self.u2_pred - self.u3_pred))
        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss,
                                                                method = 'L-BFGS-B',
                                                                options = {'maxiter': 5000,
                                                                           'maxfun': 50000,
                                                                           'maxcor': 50,
                                                                           'maxls': 50,
                                                                           'ftol' :1.0 * np.finfo(float).eps})

        init = tf.global_variables_initializer()
        self.sess.run(init)

    def initialize_NN(self, layers):
        weights = []
        biases = []
        skip_weights = []
        L = len(layers)
        n0 = layers[0]
        for l in range(L-1):
            m = layers[l]
            n = layers[l + 1]
            if l < L-1:
                SW = tf.Variable(tf.truncated_normal([1, ], mean = 0, stddev = 0.01, seed = se), dtype = tf.float32)
            skip_weights.append(SW)
            if l >= 1 and l <= L-3:
                sigma = np.sqrt(2 / (m+n0+n))
                W = tf.Variable(tf.truncated_normal([m+n0, n], mean = 0, stddev = sigma, seed = se), dtype = tf.float32)
                b = tf.Variable(tf.zeros([1, n]), dtype = tf.float32)
                weights.append(W)
                biases.append(b)
            else:
                sigma = np.sqrt(2 / (m + n))
                W = tf.Variable(tf.truncated_normal([m, n], mean = 0, stddev=sigma, seed=se), dtype = tf.float32)
                b = tf.Variable(tf.zeros([1, n]), dtype=tf.float32)
                weights.append(W)
                biases.append(b)
        return weights, biases, skip_weights

    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2/(in_dim + out_dim))
        ##################### Addition ##############################
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev, seed=se), dtype=tf.float32)
        #############################################################

    def neural_net(self, X, weights, biases, skip_weights):
        L = len(weights) + 1
        A0 = 2*(X-self.lb)/(self.ub-self.lb)-1
        W = weights[0]
        b = biases[0]
        Z = tf.add(tf.matmul(A0, W), b)
        A = tf.tanh(Z)
        for l in range(1, L-2):
            W = weights[l]
            b = biases[l]
            sw = skip_weights[l]
            A1 = tf.concat([A, A0], axis = 1)
            Z = tf.add(tf.matmul(A1, W), b)
            A = 1-tf.tanh(Z)**2+sw*A
        W = weights[-1]
        b = biases[-1]
        Z = tf.add(tf.matmul(A, W), b)
        Y = Z
        return Y

    def net_u(self, x, t):
        u = self.neural_net(tf.concat([x,t],1), self.weights, self.biases, self.skip_weights)
        return u

    def net_f(self, x,t):
        NN = self.net_u(x,t)
        rho = NN[:,0:1]
        u = NN[:,1:2]
        # U_eq = u_f*(1-tf.exp(1-tf.exp(cm/u_f*(rho_m/rho-1))))
        U_eq = u_f*((1+tf.exp((rho/rho_m-0.25)/0.06))**-1)-u_f*372*10**-8
        drho_t = tf.gradients(rho,t)[0]
        drhou_x = tf.gradients(rho*u,x)[0]
        du_t = tf.gradients(u,t)[0]
        du_x = tf.gradients(u,x)[0]
        f1 = drho_t+drhou_x
        f2 = du_t+u*du_x-(U_eq-u)/tau-c0*du_x
        return [f1,f2]

    def callback(self, loss):
        print('Loss:', loss)

    def train(self):
        tf_dict = {self.x_u_tf: self.x_u, self.t_u_tf: self.t_u,
                   self.u_tf: self.u, self.rho_tf: self.rho,
                   self.x_f_tf: self.x_f, self.t_f_tf: self.t_f}
        self.optimizer.minimize(self.sess,
                                feed_dict = tf_dict,
                                fetches = [self.loss],
                                loss_callback = self.callback)

    def predict(self, X_star):
        Y_star = self.sess.run(self.Y_pred, {self.x_u_tf: X_star[:,0:1], self.t_u_tf: X_star[:,1:2]})
        f_star = self.sess.run(self.f_pred, {self.x_f_tf: X_star[:,0:1], self.t_f_tf: X_star[:,1:2]})
        return Y_star, f_star
    
model = PhysicsInformedNN(X_u_train, rho_train, u_train, X_f_train, layers, lb, ub)
start_time = time.time()
model.train()
elapsed = time.time() - start_time
print('Training time: %.4f' % elapsed)
Y_pred, f_pred = model.predict(X_star)
u = Y_pred[:,1:2]
rho = Y_pred[:,0:1]

# input('Enter')
# U_pred = griddata(X_star, u.flatten(), (X, T), method='cubic')
# rho_pred = griddata(X_star, rho.flatten(), (X, T), method='cubic')
pd.DataFrame(u).to_csv('//home//naman/Traffic Flow//'+str(num)+'Velocity'+str(int(percent*100))+'.csv', index = False)
pd.DataFrame(rho).to_csv('//home//naman/Traffic Flow//'+str(num)+'Density'+str(int(percent*100))+'.csv', index = False)
print(num)
print(N_f)
# error_u = np.linalg.norm(u_star - u_pred, 2) / np.linalg.norm(u_star, 2)
# print('Error u: %e' % error_u)
# U_pred = griddata(X_star, u_pred.flatten(), (X, T), method='cubic')
# Error = np.abs(Exact - U_pred)