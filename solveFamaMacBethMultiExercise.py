# solveFamaMacBethMultiExercise.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Compute_Momentum.standardiseFactor import standardiseFactor

def famaMacBethMulti(returns, factors):

    [T, N, K] = factors.shape

    gamma = np.full((T,K), np.nan)
    resid = np.full((T,N), np.nan)

    for t in np.arange(1,T):

        Y = returns[t,np.newaxis].T
        X = np.hstack((np.ones((N,1)), factors[t-1,:,:]))

        coefs = np.linalg.lstsq(X, Y, rcond=None)[0].T
        gamma[t, :] = coefs[0,1:]
        resid[t,:] = Y.T - X.dot(coefs.T).T

    return gamma, resid


datadir = '/Users/berndhanke/Dropbox (Personal)/Cass Course (Quant Trading)/'

dates = pd.read_excel(datadir+'uk_factordata.xlsx',sheet_name='Returns',usecols=[0]).values
returns = pd.read_excel(datadir+'uk_factordata.xlsx',sheet_name='Returns',usecols=[1,2,3,4,5,6,7,8,9])
names = returns.columns.tolist()
returns = returns.values

factors = np.full(returns.shape+(3,), np.nan)

# Place factors into 3-dimensional array
factors[:,:,0] = pd.read_excel(datadir+'uk_factordata.xlsx',sheet_name='Book2Price',usecols=[1,2,3,4,5,6,7,8,9]).values
factors[:,:,1] = pd.read_excel(datadir+'uk_factordata.xlsx',sheet_name='Earnings2Price',usecols=[1,2,3,4,5,6,7,8,9]).values
factors[:,:,2] = pd.read_excel(datadir+'uk_factordata.xlsx',sheet_name='Momentum',usecols=[1,2,3,4,5,6,7,8,9]).values

# obtain the number of dates (rows), number of stocks (columns) and the
# number of factors (layers in the array)
T,N,K = factors.shape

for i in np.arange(0, K):
    factors[:, :, i] = standardiseFactor(factors[:, :, i])

# run Fama-MacBeth repeated cross-sectional regressions and collect
# regression coefficients (factor returns) and regression residuals
# (stock-specific returns)
gamma, residuals = famaMacBethMulti(returns, factors)

# graph cumulative factor return series
plt.plot(dates[1:],np.cumsum(gamma[1:,:],axis=0))
plt.title('Cumulative Factor Returns')
plt.legend(('Book/Price','Earnings/Price','Momentum'))
plt.show()

# extract a two-dimensional array of the latest factor exposures of each of the stocks
fac = factors[-1,:,:].reshape(N,K)

# add the systematic component of the covariance matrix estimate and the
# stock specific component to obtain the final covariance estimate
sigma_factor = fac.dot(np.cov(gamma[1:,:],rowvar=False)).dot(fac.T) + np.diag(np.nanvar(residuals,axis=0))

# compute the covariance from historical returns directly (note: this does
# not provide a robust covariance matrix estimate whenever the number of
# stock is large relative to the number of historical return periods used)
sigma_simple = np.cov(returns,rowvar=False)
