# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 14:42:29 2017

@author: utente

Sbilanciamento Terna
"""

from __future__ import division
import pandas as pd
from pandas.tools import plotting
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api
import calendar
import scipy
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from collections import OrderedDict
import datetime
#from statsmodels.tsa.stattools import adfuller

today = datetime.datetime.now()
####################################################################################################
### @param: y1 and y2 are the years to be compared; y1 < y2 and y1 will bw taken as reference, unless it is a leap year
def SimilarDaysError(df, y1, y2):
    errors = []
    y = y1
    if y % 4 == 0:
        y = y2
    for m in range(1,13,1):
        dim = calendar.monthrange(y, m)[1]
        dfm = df.ix[df.index.month == m]
        dfm5 = dfm.ix[dfm.index.year == y]
        dfm6 = dfm.ix[dfm.index.year == y2]
        for d in range(1, dim, 1):
            ddfm5 = dfm5.ix[dfm5.index.day == d]
            ddfm6 = dfm6.ix[dfm6.index.day == d]
            if ddfm5.shape[0] == ddfm6.shape[0]:
                errors.extend(ddfm6['FABBISOGNO REALE'].values.ravel() - ddfm5['FABBISOGNO REALE'].values.ravel().tolist())
    return errors
####################################################################################################
def AddHolidaysDate(vd):
    
  ##### codifica numerica delle vacanze
  ## 1 Gennaio = 1, Epifania = 2
  ## Pasqua = 3, Pasquetta = 4
  ## 25 Aprile = 5, 1 Maggio = 6, 2 Giugno = 7,
  ## Ferragosto = 8, 1 Novembre = 9
  ## 8 Dicembre = 10, Natale = 11, S.Stefano = 12, S.Silvestro = 13
    holidays = 0
    pasquetta = [datetime.datetime(2015,4,6), datetime.datetime(2016,3,28), datetime.datetime(2017,4,17)]
    pasqua = [datetime.datetime(2015,4,5), datetime.datetime(2016,3,27), datetime.datetime(2017,4,16)]
  
    if vd.month == 1 and vd.day == 1:
        holidays = 1
    if vd.month  == 1 and vd.day == 6: 
        holidays = 1
    if vd.month  == 4 and vd.day == 25: 
        holidays = 1
    if vd.month  == 5 and vd.day == 1: 
        holidays = 1
    if vd.month  == 6 and vd.day == 2: 
        holidays = 1
    if vd.month  == 8 and vd.day == 15: 
        holidays = 1
    if vd.month  == 11 and vd.day == 1: 
        holidays = 1
    if vd.month  == 12 and vd.day == 8: 
        holidays = 1
    if vd.month  == 12 and vd.day == 25: 
        holidays = 1
    if vd.month  == 12 and vd.day == 26: 
        holidays = 1
    if vd.month  == 12 and vd.day == 31: 
        holidays = 1
    if vd in pasqua:
        holidays = 1
    if vd in pasquetta:
        holidays = 1
  
    return holidays
####################################################################################################
def GetMeanCurve(df, var):
    mc = OrderedDict()
    for y in [2015, 2016]:
        dfy = df[var].ix[df.index.year == y]
        for m in range(1,13,1):
            dfym = dfy.ix[dfy.index.month == m]
            Mean = []
            for h in range(24):
                dfymh = dfym.ix[dfym.index.hour == h].mean()
                Mean.append(dfymh)
            mc[str(m) + '_' + str(y)] = Mean
    mc = pd.DataFrame.from_dict(mc, orient = 'index')
    return mc
####################################################################################################
def MakeDatasetTS(df, meteo):
    dts = OrderedDict()
    #mc = GetMeanCurve(df, 'FABBISOGNO REALE')
    for i in df.index.tolist():
        wd = i.weekday()
        h = i.hour
        dy = i.timetuple().tm_yday
        Tmax = meteo['Tmax'].ix[meteo['DATA'] == i.date()].values.ravel()[0]
        rain = meteo['PIOGGIA'].ix[meteo['DATA'] == i.date()].values.ravel()[0]
        wind = meteo['VENTOMEDIA'].ix[meteo['DATA'] == i.date()].values.ravel()[0]
        hol = AddHolidaysDate(i.date())
        dts[i] = [wd, h, dy, Tmax, rain, wind, hol, df['FABBISOGNO REALE'].ix[i]]
    dts = pd.DataFrame.from_dict(dts, orient = 'index')
    return dts
####################################################################################################
def MakeDatasetTSCurve(df, meteo):
    dts = OrderedDict()
    for i in df.index.tolist():
        m = i.month
        y = i.year
        dfm = df.ix[df.index.month == m]
        dfmy = dfm.ix[dfm.index.year == y]
        wd = i.weekday()
        h = i.hour
        dy = i.timetuple().tm_yday
        Tmax = meteo['Tmax'].ix[meteo['DATA'] == i.date()].values.ravel()[0]
        rain = meteo['PIOGGIA'].ix[meteo['DATA'] == i.date()].values.ravel()[0]
        wind = meteo['VENTOMEDIA'].ix[meteo['DATA'] == i.date()].values.ravel()[0]
        hol = AddHolidaysDate(i.date())
        cm = GetMeanCurve(dfmy.ix[dfmy.index.date <= pd.to_datetime(i).date()],'FABBISOGNO REALE').dropna().values.ravel()
        ll = [wd, h, dy, Tmax, rain, wind, hol]
        ll.extend(cm.tolist())
        ll.extend([df['FABBISOGNO REALE'].ix[i]])
        dts[i] =  ll
    dts = pd.DataFrame.from_dict(dts, orient = 'index')
    return dts
####################################################################################################
def getindex(m, y):
    if y == 2016:
        if m == 1:
            return '11_2015'
        elif m == 2:
            return '12_2015'
        else:
            return str(m-2) + '_' + str(y)
    else:
        return str(m-2) + '_' + str(y)
####################################################################################################
def MakeDatasetTSFixedCurve(df, meteo):
    dts = OrderedDict()
    cm = GetMeanCurve(df,'FABBISOGNO REALE')
    df5 = df.ix[df.index.year == 2015]
    df6 = df.ix[df.index.year == 2016]
    df = df5.ix[df5.index.month > 2].append(df6)
    for i in df.index.tolist():
        m = i.month
        y = i.year
        cmym = cm.ix[getindex(m, y)]
        wd = i.weekday()
        h = i.hour
        dy = i.timetuple().tm_yday
        Tmax = meteo['Tmax'].ix[meteo['DATA'] == i.date()].values.ravel()[0]
        rain = meteo['PIOGGIA'].ix[meteo['DATA'] == i.date()].values.ravel()[0]
        wind = meteo['VENTOMEDIA'].ix[meteo['DATA'] == i.date()].values.ravel()[0]
        hol = AddHolidaysDate(i.date())
        ll = [wd, h, dy, Tmax, rain, wind, hol]
        ll.extend(cmym.tolist())
        ll.extend([df['FABBISOGNO REALE'].ix[i]])
        dts[i] =  ll
    dts = pd.DataFrame.from_dict(dts, orient = 'index')
    dts.columns = [['weekday', 'hour', 'pday', 'Tmax', 'pioggia', 'vento', 'holiday', 
                    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', 
                    '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23',
                    'y']]
    return dts
####################################################################################################
def percentageConsumption(db, All, zona):
    dr = pd.date_range('2017-01-01', '2017-03-01', freq = 'D')
    diz = OrderedDict()
    dbz = db.ix[db["Area"] == zona]
    for d in dr:
        pods = dbz["POD"].ix[dbz["Giorno"] == d].values.ravel().tolist()
        All2 = All.ix[All["Trattamento_01"] == 'O']
        totd = np.sum(np.nan_to_num([All2["CONSUMO_TOT"].ix[y] for y in All2.index if All2["POD"].ix[y] in pods]))/1000
        #totd = All2["CONSUMO_TOT"].ix[All2["POD"].values.ravel() in pods].sum()
        tot = All2["CONSUMO_TOT"].sum()/1000
        p = totd/tot
        diz[d] = [p]
    diz = pd.DataFrame.from_dict(diz, orient = 'index')
    return diz
####################################################################################################
def MakeDatasetTSLYFixedCurve(df, meteo):
    dts = OrderedDict()
    cm = GetMeanCurve(df,'FABBISOGNO REALE')
    df = df.ix[df.index.year >= 2016]
    for i in df.index.tolist():
        m = i.month
        y = 2015 if i.year == 2016 else 2016
        cmym = cm.ix[str(m) + '_' + str(y)]
        wd = i.weekday()
        h = i.hour
        dy = i.timetuple().tm_yday
        Tmax = meteo['Tmax'].ix[meteo['DATA'] == i.date()].values.ravel()[0]
        rain = meteo['PIOGGIA'].ix[meteo['DATA'] == i.date()].values.ravel()[0]
        wind = meteo['VENTOMEDIA'].ix[meteo['DATA'] == i.date()].values.ravel()[0]
        hol = AddHolidaysDate(i.date())
        ll = [wd, h, dy, Tmax, rain, wind, hol]
        ll.extend(cmym.tolist())
        ll.extend([df['FABBISOGNO REALE'].ix[i]])
        dts[i] =  ll
    dts = pd.DataFrame.from_dict(dts, orient = 'index')
    dts.columns = [['weekday', 'hour', 'pday', 'Tmax', 'pioggia', 'vento', 'holiday', 
                    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', 
                    '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23',
                    'y']]
    return dts
####################################################################################################
def MakeDatasetWithSampleCurve(df, db, meteo, All, zona):
#### @PARAM: df is the dataset from Terna, db, All zona those for computing the perc consumption
#### and the sample curve
    psample = percentageConsumption(db, All, zona)
    psample = psample.set_index(pd.date_range('2017-01-01', '2017-03-01', freq = 'D'))
    dts = OrderedDict()
    df = df.ix[df.index.date >= datetime.date(2017,1,3)]
    for i in df.index.tolist():
        cmym = db[db.columns[10:34]].ix[db["Giorno"] == (i.date()- datetime.timedelta(days = 2))].sum(axis = 0).values.ravel()/1000
        wd = i.weekday()
        h = i.hour
        dy = i.timetuple().tm_yday
        Tmax = meteo['Tmax'].ix[meteo['DATA'] == i.date()].values.ravel()[0]
        rain = meteo['PIOGGIA'].ix[meteo['DATA'] == i.date()].values.ravel()[0]
        wind = meteo['VENTOMEDIA'].ix[meteo['DATA'] == i.date()].values.ravel()[0]
        hol = AddHolidaysDate(i.date())
        ps = psample.ix[psample.index.date == i.date()]
        ll = [wd, h, dy, Tmax, rain, wind, hol, ps[0].values[0]]
        ll.extend(cmym.tolist())
        ll.extend([df['MO [MWh]'].ix[i]])
        dts[i] =  ll
    dts = pd.DataFrame.from_dict(dts, orient = 'index')
    dts.columns = [['weekday','hour','pday','tmax','pioggia','vento','holiday','perc',
    '0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','y']]
    return dts
####################################################################################################
#def test_stationarity(timeseries):
#    
#    #Determing rolling statistics
#    rolmean = pd.rolling_mean(timeseries, window=12)
#    rolstd = pd.rolling_std(timeseries, window=12)
#    
#    #Plot rolling statistics:
#    fig = plt.figure(figsize=(12, 8))
#    orig = plt.plot(timeseries, color='blue',label='Original')
#    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
#    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
#    plt.legend(loc='best')
#    plt.title('Rolling Mean & Standard Deviation')
#    plt.show()
#        
#    #Perform Dickey-Fuller test:
#    print 'Results of Dickey-Fuller Test:'
#    dftest = adfuller(timeseries, autolag='AIC')
#    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
#    for key,value in dftest[4].items():
#        dfoutput['Critical Value (%s)'%key] = value
#    print dfoutput 
####################################################################################################


sbil = pd.read_excel('C:/Users/utente/Documents/misure/aggregato_sbilanciamento.xlsx')
nord = sbil.ix[sbil['CODICE RUC'] == 'UC_DP1608_NORD']
nord.index = pd.date_range('2015-01-01', '2017-01-02', freq = 'H')[:nord.shape[0]]


nord['FABBISOGNO REALE'].plot()

nord['FABBISOGNO REALE'].resample('D').max()
nord['FABBISOGNO REALE'].resample('D').min()
nord['FABBISOGNO REALE'].resample('D').std()

nrange = nord['FABBISOGNO REALE'].resample('D').max() - nord['FABBISOGNO REALE'].resample('D').min()

plt.figure()
plt.plot(nrange)

dec = statsmodels.api.tsa.seasonal_decompose(nord['FABBISOGNO REALE'].values.ravel(), freq = 24)
dec.plot()

errn = SimilarDaysError(nord)

plt.figure()
plt.plot(np.array(errn), color = 'red')
plt.axhline(y = np.mean(errn), color = 'navy')
plt.axhline(y = np.median(errn), color = 'gold')
plt.axhline(y = scipy.stats.mstats.mquantiles(errn, prob = 0.025), color = 'black')
plt.axhline(y = scipy.stats.mstats.mquantiles(errn, prob = 0.975), color = 'black')


np.mean(errn)
np.median(errn)
np.std(errn)


wderrn = np.array(errn)[np.array(errn) <= 20]
wderrn = wderrn[wderrn >= -20]
wderrn.size/len(errn)

np.median(wderrn)
np.mean(wderrn)

plt.figure()
plt.plot(wderrn)

x = np.linspace(0, 8760, num = 8760)[:, np.newaxis]
y = nord['FABBISOGNO REALE'].ix[nord.index.year == 2015].values.ravel()
regr = AdaBoostRegressor(DecisionTreeRegressor(max_depth = 24),n_estimators=3000)

regr.fit(x, y)
yhat = regr.predict(x)

plt.figure()
plt.plot(yhat, color = 'blue', marker = 'o')
plt.plot(y, color = 'red')

plt.figure()
plt.plot(y - yhat)

#### fabbisogno 2009
sbil2009 = pd.read_excel('C:/Users/utente/Documents/misure/aggregato_sbilanciamento2009.xlsx')
nord2009 = sbil2009.ix[sbil2009['CODICE RUC'] == 'UC_DP1608_NORD']
nord2009.index = pd.date_range('2009-01-01', '2010-01-02', freq = 'H')[:nord2009.shape[0]]

#### difference between 2015 and 2009 since they were identical years (same days were on the same days)
diff = nord['FABBISOGNO REALE'].ix[nord.index.year == 2015].values.ravel() - nord2009['FABBISOGNO REALE'].values.ravel()

plt.figure()
plt.plot(diff)

##### experiment AdaBoost + Decision Trees 2015 to 2016
cnord = sbil.ix[sbil['CODICE RUC'] == 'UC_DP1608_CNOR']
cnord.index = pd.date_range('2015-01-01', '2017-02-02', freq = 'H')[:cnord.shape[0]]
fi5 = pd.read_excel('C:/Users/utente/Documents/PUN/Firenze 2015.xlsx')
fi5 = fi5.ix[:364].set_index(pd.date_range('2015-01-01', '2015-12-31', freq = 'D'))
fi6 = pd.read_excel('C:/Users/utente/Documents/PUN/Firenze 2016.xlsx')
fi6 = fi6.ix[:365].set_index(pd.date_range('2016-01-01', '2016-12-31', freq = 'D'))
fi7 = pd.read_excel('C:/Users/utente/Documents/PUN/Firenze 2017.xlsx')
fi7 = fi7.set_index(pd.date_range('2017-01-01', '2017-01-31', freq = 'D'))
fi6 = fi6.append(fi7)
fi = fi5.append(fi6)
sard = sbil.ix[sbil['CODICE RUC'] == 'UC_DP1608_SARD']
sard.index = pd.date_range('2015-01-01', '2017-01-02', freq = 'H')[:sard.shape[0]]



DT5 = MakeDatasetTS(cnord.ix[cnord.index.year == 2015], fi5)
DT6 = MakeDatasetTS(cnord.ix[cnord.index.year == 2016], fi6)

regr = AdaBoostRegressor(DecisionTreeRegressor(max_depth = 24),n_estimators=5000)

DT5s = DT5.sample(frac = 1).reset_index(drop = True)

x = DT5[DT5.columns[:7]]
y = DT5[DT5.columns[7]]
xs = DT5s[DT5s.columns[:7]]
ys = DT5s[DT5s.columns[7]]
x6 = DT6[DT6.columns[:7]]
y6 = DT6[DT6.columns[7]]

regr.fit(xs, ys)
yhat = regr.predict(x)

regrR2 = 1 - (np.sum((y - yhat)**2))/(np.sum((y - np.mean(y))**2))

plt.figure()
plt.plot(yhat, color = 'blue', marker = 'o')
plt.plot(y.values.ravel(), color = 'red')

plt.figure()
plt.plot(y - yhat)

yhat6 = regr.predict(x6)

regr6R2 = 1 - (np.sum((y6 - yhat6)**2))/(np.sum((y6 - np.mean(y6))**2))


plt.figure()
plt.plot(yhat6, color = 'navy', marker = 'o')
plt.plot(y6.values.ravel(), color = 'coral')

plt.figure()
plt.plot(y6 - yhat6)
(y6 - yhat6).ix[(y6 - yhat6).index.month >= 8].mean()
(y6 - yhat6).ix[(y6 - yhat6).index.month < 8].mean()
(y6 - yhat6).ix[(y6 - yhat6).index.month >= 8].mean()
(y6 - yhat6).ix[(y6 - yhat6).index.month < 8].mean()


rfregr = AdaBoostRegressor(RandomForestRegressor(criterion = 'mse', max_depth = 24), n_estimators=3000)
rfregr.fit(xs, ys)
yhat = rfregr.predict(x)

regrR2 = 1 - (np.sum((y - yhat)**2))/(np.sum((y - np.mean(y))**2))

yhat6 = rfregr.predict(x6)
regr6R2 = 1 - (np.sum((y6 - yhat6)**2))/(np.sum((y6 - np.mean(y6))**2))

plt.figure()
plt.plot(yhat6, color = 'navy', marker = 'o')
plt.plot(y6.values.ravel(), color = 'coral')

plt.figure()
plt.plot(y6 - yhat6)
(y6 - yhat6).ix[(y6 - yhat6).index.month >= 8].mean()
(y6 - yhat6).ix[(y6 - yhat6).index.month < 8].mean()
(y6 - yhat6).ix[(y6 - yhat6).index.month >= 8].std()
(y6 - yhat6).ix[(y6 - yhat6).index.month < 8].std()

mc = GetMeanCurve(cnord, 'FABBISOGNO REALE')

mc15 = mc.ix[mc.index[:12]]
mc16 = mc.ix[mc.index[12:]]

mc15.T.plot(legend = False)
mc16.T.plot(legend = False)
ydiff = mc16.reset_index(drop = True) - mc15.reset_index(drop = True)
ytdiff = fi6['Tmedia'].resample('M').mean().values.ravel() - fi5['Tmedia'].resample('M').mean().values.ravel()

plt.figure()
plt.plot(ytdiff)

plt.figure()
ydiff.T.plot(legend = False)
plt.axhline(y = 0)


DTC = MakeDatasetTSCurve(cnord, fi)

DTC.to_excel('DTC.xlsx') #### in Users/utente

DTC = pd.read_excel('C:/Users/utente/DTC.xlsx')

DTCs = DTC.sample(frac = 1).reset_index(drop = True)
trs = np.random.randint(0, DTC.shape[0], np.ceil(DTC.shape[0] * 0.85))
tes = list(set(range(DTC.shape[0] )).difference(set(trs)))


x = DTC[DTC.columns[:31]]
y = DTC[DTC.columns[31]]
xs = DTCs[DTCs.columns[:31]]
ys = DTCs[DTCs.columns[31]]

rfregr = AdaBoostRegressor(RandomForestRegressor(criterion = 'mse', max_depth = 24), n_estimators=3000)
rfregr = AdaBoostRegressor(DecisionTreeRegressor(criterion = 'mse', max_depth = 24), n_estimators=3000)
rfregr.fit(DTCs[DTCs.columns[:31]].ix[trs], DTCs[DTCs.columns[31]].ix[trs])
yhat = rfregr.predict(DTCs[DTCs.columns[:31]].ix[trs])

regrR2 = 1 - (np.sum((DTCs[DTCs.columns[31]].ix[trs] - yhat)**2))/(np.sum((DTCs[DTCs.columns[31]].ix[trs] - np.mean(DTCs[DTCs.columns[31]].ix[trs]))**2))

yhat6 = rfregr.predict(DTCs[DTCs.columns[:31]].ix[tes])
regr6R2 = 1 - (np.sum((DTCs[DTCs.columns[31]].ix[tes] - yhat6)**2))/(np.sum((DTCs[DTCs.columns[31]].ix[tes] - np.mean(DTCs[DTCs.columns[31]].ix[tes]))**2))

plt.figure()
plt.plot(yhat6, color = 'navy', marker = 'o')
plt.plot(DTCs[DTCs.columns[31]].ix[tes].values.ravel(), color = 'coral')

y6 = DTCs[DTCs.columns[31]].ix[tes].values.ravel()

plt.figure()
plt.plot(y6 - yhat6)
(y6 - yhat6).ix[(y6 - yhat6).index.month >= 8].mean()
(y6 - yhat6).ix[(y6 - yhat6).index.month < 8].mean()
(y6 - yhat6).ix[(y6 - yhat6).index.month >= 8].std()
(y6 - yhat6).ix[(y6 - yhat6).index.month < 8].std()

np.mean(y6 - yhat6)
np.median(y6 - yhat6)
np.std(y6 - yhat6)

###### Try MakeDatasetTSFixedCurve

DTFC = MakeDatasetTSFixedCurve(cnord, fi)
DTFC = MakeDatasetTSLYFixedCurve(cnord, fi)

#test_stationarity(DTFC['y'].values.ravel())

import statsmodels

DTFC2 = DTFC.ix[DTFC.index.year == 2017]
DTFC = DTFC.ix[DTFC.index.year < 2017]

mod = statsmodels.api.tsa.statespace.SARIMAX(DTFC['y'].values.ravel(), exog = DTFC[DTFC.columns[:31]], trend='n', order=(24,0,24), seasonal_order=(1,1,1,24), enforce_stationarity = False, enforce_invertibility = False)

results = mod.fit()
print results.summary()
results.plot_diagnostics()
res = results.resid
##### http://www.statsmodels.org/dev/generated/statsmodels.tsa.statespace.sarimax.SARIMAXResults.html#statsmodels.tsa.statespace.sarimax.SARIMAXResults
#####ake dataset for 2017 (Jan) and try this model on it 
s_prediction = results.forecast(steps = 31*24, exog = DTFC2[DTFC2.columns[:31]])

plt.figure()
plt.plot(s_prediction.values.ravel())
plt.plot(DTFC2['y'].values.ravel())

### shuffle the dataset and build a model leaving the time dependence structure out
trs = np.random.randint(0, DTFC.shape[0], np.ceil(DTFC.shape[0] * 0.85))
tes = list(set(range(DTFC.shape[0] )).difference(set(trs)))

### treat the dataset as a true time series
trs = np.arange(int(np.ceil(DTFC.shape[0] * 0.85)))
np.random.shuffle(trs)
tes = list(set(range(DTFC.shape[0])).difference(set(trs.tolist())))

wdtrs = DTFC.ix[trs]
wdtrs = wdtrs.ix[wdtrs['holiday'] == 0]
wdtrs = wdtrs.ix[wdtrs['weekday'] < 5]

wdtes = DTFC.ix[tes]
wdtes = wdtes.ix[wdtes['holiday'] == 0]
wdtes = wdtes.ix[wdtes['weekday'] < 5]

wetrs = DTFC.ix[trs]
wetrs = wetrs.ix[wetrs['holiday'] == 1].append(wetrs.ix[wetrs['weekday'] >= 5])


wetes = DTFC.ix[tes]
wetes = wetes.ix[wetes['holiday'] == 1].append(wetes.ix[wetes['weekday'] >= 5])

ffregr = AdaBoostRegressor(DecisionTreeRegressor(criterion = 'mse', max_depth = 24), n_estimators=3000)
#ffregr = AdaBoostRegressor(RandomForestRegressor(criterion = 'mse', max_depth = 24), n_estimators=1500) #not bad anyway
#ffregr = AdaBoostRegressor(RandomForestRegressor(criterion = 'mae', max_depth = 24), n_estimators=1500)

ffregr.fit(wdtrs[wdtrs.columns[:31]], wdtrs[wdtrs.columns[31]])
########### weekdays ###############################################################################
fyhat = ffregr.predict(wdtrs[wdtrs.columns[:31]])

fregrR2 = 1 - (np.sum((wdtrs[wdtrs.columns[31]] - fyhat)**2))/(np.sum((wdtrs[wdtrs.columns[31]] - np.mean(wdtrs[wdtrs.columns[31]]))**2))

fyhat6 = ffregr.predict(wdtes[wdtes.columns[:31]])
fregr6R2 = 1 - (np.sum((wdtes[wdtes.columns[31]] - fyhat6)**2))/(np.sum((wdtes[wdtes.columns[31]] - np.mean(wdtes[wdtes.columns[31]]))**2))

plt.figure()
plt.plot(fyhat6, color = 'blue', marker = 'o')
plt.plot(wdtes[wdtes.columns[31]].values.ravel(), color = 'red')

wderr = wdtes[wdtes.columns[31]].values.ravel() - fyhat6
plt.figure()
plt.plot(wderr)
wdmae = np.abs(wderr)/wdtes[wdtes.columns[31]].values.ravel()
plt.figure()
plt.plot(wdmae)

plt.figure()
plt.hist(wdmae, bins = 20)

dfyh = pd.DataFrame(fyhat6).set_index(wdtes.index)
realmean = []
hatmean = []
for h in range(24):
    realmean.append(wdtes['y'].ix[wdtes.index.hour == h].mean())
    hatmean.append(dfyh.ix[dfyh.index.hour == h].mean())

plt.figure()
plt.plot(np.array(realmean))
plt.plot(np.array(hatmean), color = 'red', marker = 'o')

########################### weekends and holidays ##################################################

ffregr.fit(wetrs[wetrs.columns[:31]], wetrs[wetrs.columns[31]])
fyhat = ffregr.predict(wetrs[wetrs.columns[:31]])

fregrR2 = 1 - (np.sum((wetrs[wetrs.columns[31]] - fyhat)**2))/(np.sum((wetrs[wetrs.columns[31]] - np.mean(wetrs[wetrs.columns[31]]))**2))

fyhat6 = ffregr.predict(wetes[wetes.columns[:31]])
fregr6R2 = 1 - (np.sum((wetes[wetes.columns[31]] - fyhat6)**2))/(np.sum((wetes[wetes.columns[31]] - np.mean(wetes[wetes.columns[31]]))**2))

plt.figure()
plt.plot(fyhat6, color = 'blue', marker = 'o')
plt.plot(wetes[wetes.columns[31]].values.ravel(), color = 'red')

weerr = wetes[wetes.columns[31]].values.ravel() - fyhat6
plt.figure()
plt.plot(weerr)
wemae = np.abs(weerr)/wetes[wetes.columns[31]].values.ravel()
plt.figure()
plt.plot(wemae)

plt.figure()
plt.hist(wemae, bins = 20)

dfyh = pd.DataFrame(fyhat6).set_index(wetes.index)
realmean = []
hatmean = []
for h in range(24):
    realmean.append(wetes['y'].ix[wetes.index.hour == h].mean())
    hatmean.append(dfyh.ix[dfyh.index.hour == h].mean())

plt.figure()
plt.plot(np.array(realmean))
plt.plot(np.array(hatmean), color = 'red', marker = 'o')





## http://stackoverflow.com/questions/23118309/scikit-learn-randomforest-memory-error
#rfregr = AdaBoostRegressor(RandomForestRegressor(criterion = 'mse', max_depth = 24), n_estimators=3000)
ffregr = AdaBoostRegressor(DecisionTreeRegressor(criterion = 'mse', max_depth = 24), n_estimators=3000)
ffregr.fit(DTFC[DTFC.columns[:31]].ix[trs], DTFC[DTFC.columns[31]].ix[trs])
fyhat = ffregr.predict(DTFC[DTFC.columns[:31]].ix[trs])

fregrR2 = 1 - (np.sum((DTFC[DTFC.columns[31]].ix[trs] - fyhat)**2))/(np.sum((DTFC[DTFC.columns[31]].ix[trs] - np.mean(DTFC[DTFC.columns[31]].ix[trs]))**2))

fyhat6 = ffregr.predict(DTFC[DTFC.columns[:31]].ix[tes])
fregr6R2 = 1 - (np.sum((DTFC[DTFC.columns[31]].ix[tes] - fyhat6)**2))/(np.sum((DTFC[DTFC.columns[31]].ix[tes] - np.mean(DTFC[DTFC.columns[31]].ix[tes]))**2))

plt.figure()
plt.plot(fyhat6, color = 'blue', marker = 'o')
plt.plot(DTFC[DTFC.columns[31]].ix[tes].values.ravel(), color = 'red')

fy6 = DTFC[DTFC.columns[31]].ix[tes].values.ravel()


np.mean(fy6 - fyhat6)
np.median(fy6 - fyhat6)
np.std(fy6 - fyhat6)
np.max(fy6 - fyhat6)
fMAE = np.abs(fy6 - fyhat6)/fy6

plt.figure()
plt.plot(fy6 - fyhat6)
plt.axvline(x = fMAE.tolist().index(np.max(fMAE)), color = 'red')

np.mean(fMAE)
np.median(fMAE)
np.max(fMAE)
np.std(fMAE)
scipy.stats.mstats.mquantiles(fMAE, prob = [0.6, 0.7, 0.8, 0.9, 0.95, 0.975, 0.98, 0.99])

Err = pd.DataFrame(fy6 - fyhat6)


plt.figure()
plotting.autocorrelation_plot( Err)

dfy6 = np.diff(fy6)
dfyhat6 = np.diff(fyhat6)

plt.figure()
plt.hist(dfy6, bins = 20)
plt.figure()
plt.hist(dfyhat6, bins = 20, color = 'green')
scipy.stats.mstats.mquantiles(dfy6, prob = [0.6, 0.7, 0.8, 0.9, 0.95, 0.975, 0.98, 0.99])
scipy.stats.mstats.mquantiles(dfyhat6, prob = [0.6, 0.7, 0.8, 0.9, 0.95, 0.975, 0.98, 0.99])
plotting.autocorrelation_plot(dfy6)
plotting.autocorrelation_plot(dfyhat6, color = 'green')

###### TOT ORDERED DATA:
YH = ffregr.predict(DTFC[DTFC.columns[:31]])

R2 = 1 - (np.sum((DTFC[DTFC.columns[31]] - YH)**2))/(np.sum((DTFC[DTFC.columns[31]] - np.mean(DTFC[DTFC.columns[31]]))**2))
Y = DTFC[DTFC.columns[31]].values.ravel()

plt.figure()
plt.plot(Y)
plt.plot(YH, marker = 'o', color = 'grey')

Err = Y - YH
MAE = np.abs(Err)/Y

plt.figure()
plt.hist(Y, bins = 20)
plt.figure()
plt.hist(YH, bins = 20, color = "red")


plt.figure()
plt.plot(Err, color = 'red')
plt.figure()
plt.plot(MAE, color = 'orange')
plt.axhline(y = scipy.stats.mstats.mquantiles(MAE, prob = 0.99))
plt.figure()
plt.hist(MAE, bins = 40, color = 'orange')

### in what hours is MAE greater than 0.15?
### if I put the 99% quantile the num of observation greater than it is 161 => the 99% quantile is < 0.15
dfmae = pd.DataFrame(MAE)
dfmae = dfmae.set_index(DTFC.index)

over = dfmae.ix[dfmae[0] > 0.15].index
hover = over.hour
xh = [tuple([h]) for h in hover.tolist()]
from collections import Counter

freq_h = Counter(xh)


########
gen = scipy.stats.pareto.fit(MAE)
pareto_sample = scipy.stats.pareto.rvs(gen[0], gen[1], gen[2], size = MAE.size)

#### looks *very* similar to MAE
plt.figure()
plt.plot(pareto_sample)
np.where(pareto_sample >= scipy.stats.mstats.mquantiles(pareto_sample, prob = 0.99))[0].size/pareto_sample.size ### same number!!!

#### what is the distribution of the imbalance in 2015, 2016?
imb = cnord['SBILANCIAMENTO FISICO [MWh]']
measured_imb = np.abs(imb.values.ravel())/cnord['FABBISOGNO REALE'].values.ravel()

plt.figure()
plt.plot(measured_imb, color = 'red')
plt.axhline(y = 0.15, color = 'black')
plt.axhline(y = scipy.stats.mstats.mquantiles(measured_imb, prob = 0.90), color = 'purple')
plt.figure()
plotting.autocorrelation_plot(measured_imb)

scipy.stats.mstats.mquantiles(MAE, prob = [0.1,0.2,0.3,0.4,0.5,0.6, 0.7, 0.8, 0.9, 0.95, 0.975, 0.98, 0.99])
np.where(MAE >= scipy.stats.mstats.mquantiles(MAE, prob = 0.99))[0].size/MAE.size

plt.figure()
plotting.autocorrelation_plot(Err)
plt.figure()
plotting.autocorrelation_plot(Y)
plt.figure()
plotting.autocorrelation_plot(YH, color = 'green')

import statsmodels.graphics
statsmodels.graphics.tsaplots.plot_acf(Y, lags = 30*24)
plotting.lag_plot(Y, lag = 30*24)
plt.figure()
plotting.autocorrelation_plot(YH, color = 'green')

col1 = []
col2 = []
i = 0
j = 30*24 
while j < DTFC.shape[0]:
    col1.append(DTFC[DTFC.columns[31]].ix[i])
    col2.append(DTFC[DTFC.columns[31]].ix[j])
    i += 1
    j += 1
    
plt.figure()
plt.plot(np.array(col1))
plt.figure()
plt.plot(np.array(col2), color = 'magenta')
plt.figure()
plt.plot(scipy.signal.correlate(col1,col2,mode="full")/np.var(scipy.signal.correlate(col1,col2,mode="full")))
np.corrcoef(col1, col2)

######## RANDOM FOREST
rfregr = AdaBoostRegressor(RandomForestRegressor(criterion = 'mse', max_depth = 24, n_jobs = 1), n_estimators=3000)
rfregr.fit(DTFC[DTFC.columns[:31]].ix[trs], DTFC[DTFC.columns[31]].ix[trs])
fyhat = rfregr.predict(DTFC[DTFC.columns[:31]].ix[trs])

rfregrR2 = 1 - (np.sum((DTFC[DTFC.columns[31]].ix[trs] - fyhat)**2))/(np.sum((DTFC[DTFC.columns[31]].ix[trs] - np.mean(DTFC[DTFC.columns[31]].ix[trs]))**2))

fyhat6 = ffregr.predict(DTFC[DTFC.columns[:31]].ix[tes])
fregr6R2 = 1 - (np.sum((DTFC[DTFC.columns[31]].ix[tes] - fyhat6)**2))/(np.sum((DTFC[DTFC.columns[31]].ix[tes] - np.mean(DTFC[DTFC.columns[31]].ix[tes]))**2))

plt.figure()
plt.plot(fyhat6, color = 'blue', marker = 'o')
plt.plot(DTFC[DTFC.columns[31]].ix[tes].values.ravel(), color = 'red')

fy6 = DTFC[DTFC.columns[31]].ix[tes].values.ravel()

####################################################################################################
db = pd.read_excel("C:/Users/utente/Documents/Sbilanciamento/DB_2017_copia.xlsm", sheetname = "DB_SI_perd")
db["Giorno"] = pd.to_datetime(db["Giorno"].values)

dt = pd.read_excel("C:/Users/utente/Documents/Sbilanciamento/Aggregatore_orari - 17_03.xlsm",
                   skiprows = [0], sheetname = "Consumi base 24")

dt.columns = [str(i) for i in dt.columns]

db = pd.read_excel("C:/Users/utente/Documents/Sbilanciamento/Agg_consumiGG.xlsx")
All = pd.read_excel("C:/Users/utente/Documents/Sbilanciamento/_All_CRPP_01_2017.xlsx")

pJan = percentageConsumption(dt, All, "NORD")


sbil = pd.read_excel('C:/Users/utente/Documents/misure/aggregato_sbilanciamento.xlsx')
nord = sbil.ix[sbil['CODICE RUC'] == 'UC_DP1608_NORD']
nord.index = pd.date_range('2015-01-01', '2017-12-31', freq = 'H')[:nord.shape[0]]
mi6 = pd.read_excel('C:/Users/utente/Documents/PUN/Milano 2016.xlsx')
mi6 = mi6.ix[:365].set_index(pd.date_range('2016-01-01', '2016-12-31', freq = 'D'))
mi7 = pd.read_excel('C:/Users/utente/Documents/PUN/Milano 2017.xlsx')
mi7 = mi7.set_index(pd.date_range('2017-01-01', '2017-03-31', freq = 'D'))
mi = mi6.append(mi7)
#mi = fi5.append(fi6)

DB = MakeDatasetWithSampleCurve(nord, dt, mi, All, "NORD")

train = DB.ix[:int(np.ceil(0.8*DB.shape[0]))].sample(frac = 1)
test = DB.ix[1 + int(np.ceil(0.8*DB.shape[0])):]

ffregr = AdaBoostRegressor(DecisionTreeRegressor(criterion = 'mse', max_depth = 24), n_estimators=3000)
ffregr =  AdaBoostRegressor(RandomForestRegressor(criterion = 'mse', max_depth = 24, n_jobs = 1), n_estimators=3000)
ffregr.fit(train[train.columns[:32]], train[train.columns[32]])
yhat_train = ffregr.predict(train[train.columns[:32]])

fregrR2 = 1 - (np.sum((train[train.columns[32]] - yhat_train)**2))/(np.sum((train[train.columns[32]] - np.mean(train[train.columns[32]]))**2))

plt.figure()
plt.plot(yhat_train, color = 'skyblue', marker = 'o')
plt.plot(train[train.columns[32]].values.ravel(), color = 'orange', marker = '+')


yhat_test = ffregr.predict(test[test.columns[:32]])
fregrR2_test = 1 - (np.sum((test[test.columns[32]] - yhat_test)**2))/(np.sum((test[test.columns[32]] - np.mean(test[test.columns[32]]))**2))

plt.figure()
plt.plot(yhat_test, color = 'blue', marker = 'o')
plt.plot(test[test.columns[32]].values.ravel(), color = 'red', marker = '+')

ABdiff = yhat_test - test[test.columns[32]].values.ravel()

plt.figure()
plt.plot(ABdiff, color = 'coral')

np.mean(ABdiff)
np.median(ABdiff)
np.std(ABdiff)
scipy.stats.skew(ABdiff)
scipy.stats.kurtosis(ABdiff)

plt.figure()
plt.hist(ABdiff, bins = 20)

MAE = np.abs(ABdiff)/test[test.columns[32]].values.ravel()

plt.figure()
plt.plot(MAE, color = 'pink')
plt.figure()
plt.hist(MAE, bins = 20, color = "pink")
plt.figure()
plt.bar(np.arange(MAE.size), MAE, color = 'pink')
plt.figure()
plotting.autocorrelation_plot(MAE)


train.ix[train.index.date == datetime.date(2017,1,19)]
train.ix[train.index.date == datetime.date(2017,1,20)]

### http://scikit-learn.org/stable/modules/cross_validation.html
from sklearn.model_selection import cross_val_score
from sklearn import model_selection
scores5 = cross_val_score(ffregr, X = train[train.columns[:32]], y = train[train.columns[32]], cv = 5)

parameters = {'n_estimators': [24, 50, 100, 500], 'max_depth': [24, 48, 168] }
ABRF = AdaBoostRegressor(RandomForestRegressor(n_jobs = 1))
RF = RandomForestRegressor(criterion = 'mse', n_jobs = 1)

grid_RF = model_selection.GridSearchCV(RF, parameters, cv = 5)
grid_RF.fit(X = train[train.columns[:32]], y = train[train.columns[32]])

brf = RandomForestRegressor(criterion = 'mse', max_depth = 48, n_estimators = 24, n_jobs = 1)

brf.fit(train[train.columns[:32]], train[train.columns[32]])
yhat_train = brf.predict(train[train.columns[:32]])

rfR2 = 1 - (np.sum((train[train.columns[32]] - yhat_train)**2))/(np.sum((train[train.columns[32]] - np.mean(train[train.columns[32]]))**2))

plt.figure()
plt.plot(yhat_train, color = 'skyblue', marker = 'o')
plt.plot(train[train.columns[32]].values.ravel(), color = 'orange', marker = '+')

yhat_test = brf.predict(test[test.columns[:32]])
rfR2_test = 1 - (np.sum((test[test.columns[32]] - yhat_test)**2))/(np.sum((test[test.columns[32]] - np.mean(test[test.columns[32]]))**2))

plt.figure()
plt.plot(yhat_test, color = 'blue', marker = 'o')
plt.plot(test[test.columns[32]].values.ravel(), color = 'red', marker = '+')

importance = brf.feature_importances_
importance = pd.DataFrame(importance, index=train.columns[:32], columns=["Importance"])

importance["Std"] = np.std([tree.feature_importances_ for tree in brf.estimators_], axis=0)

x = range(importance.shape[0])
y = importance.ix[:, 0]
yerr = importance.ix[:, 1]

plt.figure()
plt.bar(x, y, yerr=yerr, align="center")
