# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 11:29:39 2017

@author: utente

Sbilanciamento 6 -- CONTROL COHERENCY FORECAST MODULE --

"""

import pandas as pd
import numpy as np
import datetime
import os
import scipy

####################################################################################################
def fourierExtrapolation(x, n_predict, n_harmonics = 0):
    x = np.array(x)
    n = x.size
    if n_harmonics == 0:
        n_harm = 100                     # number of harmonics in model
    else:
        n_harm = n_harmonics
    t = np.arange(0, n)
    p = np.polyfit(t, x, 1)         # find linear trend in x
    x_notrend = x - p[0] * t        # detrended x
    x_freqdom = np.fft.fft(x_notrend)  # detrended x in frequency domain
    f = np.fft.fftfreq(n)              # frequencies
    indexes = list(range(n))
    # sort indexes by frequency, lower -> higher
    indexes.sort(key = lambda i: np.absolute(f[i]))
 
    t = np.arange(0, n + n_predict)
    restored_sig = np.zeros(t.size)
    for i in indexes[:1 + n_harm * 2]:
        ampli = np.absolute(x_freqdom[i]) / n   # amplitude
        phase = np.angle(x_freqdom[i])          # phase
        restored_sig += ampli * np.cos(2 * np.pi * f[i] * t + phase)
    return restored_sig + p[0] * t        
####################################################################################################
def H1_norm(x):
    h1 = np.mean(x**2) + np.mean(np.diff(x)**2)
    return np.sqrt(h1)
####################################################################################################
def CompareSubsequentPrediction(zona, td = 1):
    ED = pd.read_hdf('C:/Users/utente/Documents/Sbilanciamento/variazioni_gg_cons_' + zona + '.h5')
    today = datetime.datetime.now()
    if not os.path.exists('C:/Users/utente/Documents/Sbilanciamento/forecast_' + zona + '.h5' ):
        print 'No forecast yet'
        return 0        
    else:
        forecast = pd.read_hdf('C:/Users/utente/Documents/Sbilanciamento/forecast_' + zona + '.h5')
        yft = forecast.ix[forecast.index.date == today.date() + datetime.timedelta(days = td)]
        yfy = forecast.ix[forecast.index.date == today.date()]
        HN = H1_norm((yft.values.ravel() - yfy.values.ravel()))
        eddata = pd.DatetimeIndex(ED['Data'].values.ravel()).to_pydatetime()
        EDData = [x.weekday() for x in eddata]
        var_nw = ED['var'].ix[np.where(np.array(EDData) < 5)[0]].values.ravel()
        qhat = np.where(var_nw <= HN)[0].size/var_nw.size
        phat = np.where(var_nw >= HN)[0].size/var_nw.size
        print 'estimated quantile point of new variation: {}'.format(qhat)
        print 'estimated percentile point of new variation: {}'.format(phat)
        newED = pd.DataFrame({'Data':[ today.date() + datetime.timedelta(days = td)],'var':[HN],'giorno':[(today.date() + datetime.timedelta(days = td)).weekday()]})
        ED = ED.append(newED)
        ED.to_hdf('C:/Users/utente/Documents/Sbilanciamento/variazioni_gg_cons_' + zona + '.h5', 'VAR')
    return 1
####################################################################################################
def CompareSimilarDayPredictions(zona):
    ED = pd.read_hdf('C:/Users/utente/Documents/Sbilanciamento/variazioni_cons_similar_days_' + zona + '.h5')
    today = datetime.datetime.now()
    if not os.path.exists('C:/Users/utente/Documents/Sbilanciamento/forecast_' + zona + '.h5' ):
        print 'No forecast yet'
        return 0        
    else:
        forecast = pd.read_hdf('C:/Users/utente/Documents/Sbilanciamento/forecast_' + zona + '.h5')
        yft = forecast.ix[forecast.index.date == (today.date() + datetime.timedelta(days = 1))]
        yfy = forecast.ix[forecast.index.date == (today.date() - datetime.timedelta(days = 7))]
        HN = H1_norm((yft.values.ravel() - yfy.values.ravel()))
        eddata = pd.DatetimeIndex(ED['Data'].values.ravel()).to_pydatetime()
        EDData = [x.weekday() for x in eddata]
        var_nw = ED['var'].ix[np.where(np.array(EDData) < 5)[0]].values.ravel()
        qhat = np.where(var_nw <= HN)[0].size/var_nw.size
        phat = np.where(var_nw >= HN)[0].size/var_nw.size
        print 'estimated quantile point of new variation: {}'.format(qhat)
        print 'estimated percentile point of new variation: {}'.format(phat)
        newED = pd.DataFrame({'Data':[ today.date() + datetime.timedelta(days = 1)],'var':[HN],'giorno':[(today.date() + datetime.timedelta(days = 1)).weekday()]})
        ED = ED.append(newED)
        ED.to_hdf('C:/Users/utente/Documents/Sbilanciamento/variazioni_cons_similar_days_' + zona + '.h5', 'similar')
    return 1    
####################################################################################################
### The idea is to use a Likelihood Ratio, but probably needs more data
#def CompareZonalTrend(df, zona):
#    forecast = pd.read_hdf('C:/Users/utente/Documents/Sbilanciamento/forecast_' + zona + '.h5')
#    fy = max(forecast.index.year)
#    
#    FM = forecast.resample('M').mean()
#    ftrend = FM.ix[FM.index.year == min(FM.index.year)].values.ravel()/np.trapz(FM.ix[FM.index.year == min(FM.index.year)].values.ravel())
#    
##    for y in range(min(FM.index.year)+1, fy, 1):
##        ftrend = np.concatenate((ftrend, FM.ix[FM.index.year == y].values.ravel()/np.trapz(FM.ix[FM.index.year == y].values.ravel())))
#    
#    ftrend = ftrend.reshape((len(range(min(FM.index.year)+1, fy, 1)),12))
#    #### comparison of monthly trend ####
#    dfm = df['MO [MWh]'].resample('M').mean()
#    trend = dfm.ix[dfm.index.year == min(dfm.index.year)].values.ravel()/np.trapz(dfm.ix[dfm.index.year == min(dfm.index.year)].values.ravel())
#    
#    for y in range(min(dfm.index.year)+1, fy, 1):
#        trend = np.concatenate((trend, dfm.ix[dfm.index.year == y].values.ravel()/np.trapz(dfm.ix[dfm.index.year == y].values.ravel())))
#
#    trend = trend.reshape((1+len(range(min(dfm.index.year)+1, fy, 1)),12))
#
#    mu = np.mean(trend, axis = 0)    
#    Sigma = np.cov(trend, rowvar = False)
#    
#    LTheta = scipy.stats.multivariate_normal.pdf(x = mu, mean = mu, cov = Sigma)
    
    
    