import numpy as np
from numpy.linalg import inv

class KalmanFilter():
    """
    This class define the Core algorithm of Kalman filter (RTS smoothing)
    """
    def __init__(self, X0, P0, M0, H=None, R0=None):
        # States
        self.M_measurements=[]
        self.X_predicted=[]
        self.X_filtered=[]
        self.X_smoothed=[]
        # Predict Matrices
        self.A_s=[0] # from 1-N
        # Measurement Matrices
        self.H_s=[0] # from 1-N
        # Variation Matrices
        self.Q_s=[0] # Predict unc., from 1-N
        self.P_predicted=[] # from 0-N
        self.P_filtered=[] # from 0-N
        self.P_smoothed=[] # from 0-N
        self.R_s=[] # Measurement unc., from 0-N
        # Gains
        self.K_s=[0] # from 1-N
        self.G_s=[0] # from 1-N
        
        #------------
        # initialize
        self.X_predicted.append(X0)
        self.X_filtered.append(X0)
        self.P_predicted.append(P0)
        self.P_filtered.append(P0)
        self.M_measurements.append(M0)
        if H is not None:
            self.H_s.append(H)
        if R0 is not None:
            self.R_s.append(R0)
        else:
            self.R_s.append(np.diag(np.ones_like(M0)))
            
        
    # A): Forward Recursion
    def predict_foward(self, A, Q, X_previous=None, P=None, f=None):
        if X_previous is None:
            X_previous=self.X_filtered[-1]
        if P is None:
            P = self.P_filtered[-1]
        
        if f is None: # if linear
            X_predicted = A.dot(X_previous)
        else:
            X_predicted = f(X_previous)
        P_predicted = A.dot(P).dot(A.T) + Q
        
        self.X_predicted.append(X_predicted)
        self.A_s.append(A)
        self.Q_s.append(Q)
        self.P_predicted.append(P_predicted)
        
        return X_predicted, P_predicted

    def filter_forward(self,M_measure, R, X_predicted=None, P=None, H=None):
        if X_predicted is None:
            X_predicted = self.X_predicted[-1]
        if P is None:
            P=self.P_predicted[-1]
        if H is None:
            H=self.H_s[-1]
        else:
            self.H_s.append(H)
        
        # Calculating the Kalman Gain K
        S = inv(H.dot(P).dot(H.T) + R)
        K = P.dot(H.T).dot(S)

        # Combination of the predicted state, measured values, covariance matrix and Kalman Gain
        X_filtered = X_predicted + K.dot(M_measure - H.dot(X_predicted))

        # Update Process Covariance Matrix
        P_filtered = (np.identity(len(K)) - K.dot(H)).dot(P)
        
        self.X_filtered.append(X_filtered)
        self.P_filtered.append(P_filtered)
        self.K_s.append(K)
        self.M_measurements.append(M_measure)
        self.R_s.append(R)

        return X_filtered, P_filtered

    # B): Backward Recursion
    def _init_smooth(self):
        # Initialized with the last filtered X(state) and P(cov)
        self.X_smoothed.append(self.X_filtered[-1])
        self.P_smoothed.append(self.P_filtered[-1])
        self.TOTAL_STEPS=range(len(self.X_filtered)-2,-1,-1)
        self.CURRENT_STEP=self.TOTAL_STEPS[0]
        
    def _smooth_step(self):
        i=self.CURRENT_STEP
        if i==-1:
            print("Smoothing done, the current state is already the first step")
            return -1
        # Calculating the backward Kalman Gain G
        G = self.P_filtered[i].dot(self.A_s[i+1].T).dot(inv(self.P_predicted[i+1]))
        
        # Combination of the predicted state, measured values, covariance matrix and Kalman Gain
        X_smoothed = self.X_filtered[i] + G.dot(self.X_smoothed[0] - self.X_predicted[i+1])

        # Update Process Covariance Matrix
        P_smoothed = self.P_filtered[i] + G.dot(self.P_smoothed[0] - self.P_predicted[i+1]).dot(G.T)

        # Store G,X_smoothed,P_smoothed
        self.X_smoothed.insert(0,X_smoothed)
        self.P_smoothed.insert(0,P_smoothed)
        self.G_s.insert(0,G)    
        
        self.CURRENT_STEP-=1
        return X_smoothed, P_smoothed
        
    def filter_backward(self):
        self._init_smooth()
        while self.CURRENT_STEP>=0:
            X_smoothed, P_smoothed = self._smooth_step()

        return X_smoothed, P_smoothed