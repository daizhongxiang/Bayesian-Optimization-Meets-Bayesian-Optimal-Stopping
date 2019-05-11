from .kern import Kern
import numpy as np
from GPy.core.parameterization.param import Param

class ExpKernel(Kern):
    def __init__(self,input_dim, alp=1.0, bet=1.0, active_dims=None):
        super(ExpKernel, self).__init__(input_dim, active_dims, 'exp kernel')
        assert input_dim == 1, "For this kernel we assume input_dim=1"
        self.alp = Param("alp", alp)
        self.bet = Param("bet", bet)
        self.link_parameters(self.alp, self.bet)
    
    def K(self,X,X2):
        if X2 is None: X2 = X
        return (self.bet ** self.alp) / (X + X2.T + self.bet) ** self.alp
    
    def Kdiag(self,X):
        return np.squeeze((self.bet ** self.alp) / (X + X + self.bet) ** self.alp)    
    
    def parameters_changed(self):
        pass
    
    def update_gradients_full(self, dL_dK, X, X2):
        if X2 is None: X2 = X
        
        denom = (X + X2.T + self.bet) ** self.alp
        d_alp_num = np.log(self.bet) * (self.bet**self.alp) * denom - np.log(X + X2.T + self.bet) * denom * (self.bet**self.alp)
        d_alp_den = denom ** 2
        d_bet_num = self.alp * (self.bet**(self.alp-1)) * denom - self.alp * ((X + X2.T + self.bet)**(self.alp-1)) * (self.bet**self.alp)
        d_bet_den = denom ** 2

        self.alp.gradient = np.sum((d_alp_num/d_alp_den) * dL_dK)
        self.bet.gradient = np.sum((d_bet_num/d_bet_den) * dL_dK)

    def update_gradients_diag(self, dL_dKdiag, X):
        pass
    def gradients_X(self,dL_dK,X,X2):
        pass
    def gradients_X_diag(self,dL_dKdiag,X):
        pass
