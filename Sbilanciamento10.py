# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 12:17:28 2017

@author: utente

Sbilanciamento 10 -- Functions to be used in the quasi automatic running
"""

from __future__ import division
import pandas as pd
from pandas.tools import plotting
import numpy as np
import matplotlib.pyplot as plt
import calendar
import scipy
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import sklearn.preprocessing
from sklearn.tree import DecisionTreeRegressor
from collections import OrderedDict
import datetime
import time
import os
from sklearn.externals import joblib
from statsmodels.tsa.ar_model import AR

#from statsmodels.tsa.stattools import adfuller

today = datetime.datetime.now()
####################################################################################################
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
    pasquetta = [datetime.date(2015,4,6), datetime.date(2016,3,28), datetime.date(2017,4,17)]
    pasqua = [datetime.date(2015,4,5), datetime.date(2016,3,27), datetime.date(2017,4,16)]

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
def StartsDaylightSaving(vd):
    dls = 0
    DLS = [datetime.date(2016,10,30), datetime.date(2017,10,29)]
    if vd in DLS:
        dls = 1
    return dls
####################################################################################################
def EndsDaylightSaving(vd):
    dls = 0
    DLS = [datetime.date(2016,3,27), datetime.date(2017,3,26)]
    if vd in DLS:
        dls = 1
    return dls
####################################################################################################
def Bridge(vd):
    
    bridge = 0
    if vd.weekday() == 0:
        Tues = vd + datetime.timedelta(days = 1)
        if AddHolidaysDate(Tues) == 1:
            bridge = 1
    elif vd.weekday() == 4:
        Thur = vd - datetime.timedelta(days = 1)
        if AddHolidaysDate(Thur) == 1:
            bridge = 1    
    else:
        pass
    
    return bridge
####################################################################################################
def convertDates(vec):
    CD = vec.apply(lambda x: datetime.datetime(year = int(str(x)[6:10]), month = int(str(x)[3:5]), day = int(str(x)[:2]), hour = int(str(x)[11:13])))
    return CD
####################################################################################################
def Get_SampleAsTS(db, zona):
    db["Giorno"] = pd.to_datetime(db["Giorno"])
    db = db.ix[db["Area"] == zona]
    final = max(db["Giorno"])
    strm = str(final.month) if len(str(final.month)) > 1 else "0" + str(final.month)
    strd = str(final.day) if len(str(final.day)) > 1 else "0" + str(final.day)
    final_date = str(final.year) + '-' + strm + '-' + strd
    dr = pd.date_range('2017-01-01', final_date, freq = 'D')
    res = []
    for i in dr.tolist():
        dbd = db[db.columns[3:]].ix[db["Giorno"] == i].sum()/1000
        res.extend(dbd.values.tolist())
        diz = pd.DataFrame(res)
    diz.columns = [['MO [MWh]']]
    diz = diz.set_index(pd.date_range('2017-01-01', '2017-12-31', freq = 'H')[:diz.shape[0]])
    return diz
####################################################################################################
def Get_SampleAsTS_AtDay(db, zona, di, df):
    db["Giorno"] = pd.to_datetime(db["Giorno"])
    db = db.ix[db["Area"] == zona]
    dr = pd.date_range(di, df, freq = 'D')
    res = []
    for i in dr.tolist():
        dbd = db[db.columns[3:]].ix[db["Giorno"] == i].sum()/1000
        res.extend(dbd.values.tolist())
        diz = pd.DataFrame(res)
    diz.columns = [['MO [MWh]']]
    diz = diz.set_index(pd.date_range(di, '2017-12-31', freq = 'H')[:diz.shape[0]])
    return diz
####################################################################################################
def Get_OutOfSample(df, db, zona):
    db["Giorno"] = pd.to_datetime(db["Giorno"])
    db = db.ix[db["Area"] == zona]
    df = df.ix[df["CODICE RUC"] == "UC_DP1608_" + zona]
    df = df.ix[df.index.date > datetime.date(2015,12,31)]
    dr = pd.date_range('2017-01-01', max(df.index.date), freq = 'D')
    res = []
    for i in dr.tolist():
        if i.to_pydatetime().date() not in [datetime.date(2016,3,27), datetime.date(2016,10,30),datetime.date(2017,3,26), datetime.date(2017,10,29)]:
            dbd = db[db.columns[3:]].ix[db["Giorno"] == i].sum()/1000
            dfd = df.ix[df.index.date == i.to_pydatetime().date()]
            res.extend((dfd['MO [MWh]'].values - dbd.values).tolist())
        else:
            dbd = db[db.columns[3:]].ix[db["Giorno"] == i].sum()/1000
            dfd = df.ix[df.index.date == i.to_pydatetime().date()]
            for hour in range(24):
                dfdh = dfd.ix[dfd.index.hour == hour]
                sam = dbd.ix[str(hour + 1)]
                if dfdh.shape[0] == 0:
                    res.append(0)
                elif dfdh.shape[0] == 2:
                    res.append(dfdh["MO [MWh]"].sum() - sam)
                else:
                    res.append(dfdh["MO [MWh]"].values[0] - sam)
    diz = pd.DataFrame(res)
    diz.columns = [['MO [MWh]']]
    diz = diz.set_index(pd.date_range('2017-01-01', '2017-12-31', freq = 'H')[:diz.shape[0]])
    return diz
####################################################################################################
def SampleAtDay(db, dtd, zona):
    db = db.ix[db["Area"] == zona]
    return list(set(db["POD"].ix[db["Giorno"] == dtd].tolist()))
####################################################################################################
def GetRicHoliday(vd):
    pasquetta = [datetime.date(2015,4,6), datetime.date(2016,3,28), datetime.date(2017,4,17),datetime.date(2018,4,1),
                 datetime.date(2019,4,21)]
    pasqua = [datetime.date(2015,4,5), datetime.date(2016,3,27), datetime.date(2017,4,16),datetime.date(2018,4,2),
              datetime.date(2018,4,22)]

    if vd.month == 1 and vd.day == 1:
        return datetime.date(vd.year - 1, 1,1)
    if vd.month  == 1 and vd.day == 6: 
        return datetime.date(vd.year - 1, 1,6)
    if vd.month  == 4 and vd.day == 25: 
        return datetime.date(vd.year - 1, 4,25)
    if vd.month  == 5 and vd.day == 1: 
        return datetime.date(vd.year - 1, 5,1)
    if vd.month  == 6 and vd.day == 2: 
        return datetime.date(vd.year - 1, 6,2)
    if vd.month  == 8 and vd.day == 15: 
        return datetime.date(vd.year - 1, 8,15)
    if vd.month  == 11 and vd.day == 1: 
        return datetime.date(vd.year - 1, 11,1)
    if vd.month  == 12 and vd.day == 8: 
        return datetime.date(vd.year - 1, 12,8)
    if vd.month  == 12 and vd.day == 25: 
        return datetime.date(vd.year - 1, 12,25)
    if vd.month  == 12 and vd.day == 26: 
        return datetime.date(vd.year - 1, 12,26)
    if vd.month  == 12 and vd.day == 31: 
        return datetime.date(vd.year - 1, 12,31)
    if vd in pasqua:
        return pasqua[pasqua.index(vd)-1]
    if vd in pasquetta:
        return pasquetta[pasquetta.index(vd)-1]
####################################################################################################
def GetRicHoliday2(vd):
    pasquetta = [datetime.date(2015,4,6), datetime.date(2016,3,28), datetime.date(2017,4,17)]
    pasqua = [datetime.date(2015,4,5), datetime.date(2016,3,27), datetime.date(2017,4,16)]

    if vd.month == 1 and vd.day == 1:
        return datetime.date(vd.year - 2, 1,1)
    if vd.month  == 1 and vd.day == 6: 
        return datetime.date(vd.year - 2, 1,6)
    if vd.month  == 4 and vd.day == 25: 
        return datetime.date(vd.year - 2, 4,25)
    if vd.month  == 5 and vd.day == 1: 
        return datetime.date(vd.year - 2, 5,1)
    if vd.month  == 6 and vd.day == 2: 
        return datetime.date(vd.year - 2, 6,2)
    if vd.month  == 8 and vd.day == 15: 
        return datetime.date(vd.year - 2, 8,15)
    if vd.month  == 11 and vd.day == 1: 
        return datetime.date(vd.year - 2, 11,1)
    if vd.month  == 12 and vd.day == 8: 
        return datetime.date(vd.year - 2, 12,8)
    if vd.month  == 12 and vd.day == 25: 
        return datetime.date(vd.year - 2, 12,25)
    if vd.month  == 12 and vd.day == 26: 
        return datetime.date(vd.year - 2, 12,26)
    if vd.month  == 12 and vd.day == 31: 
        return datetime.date(vd.year - 2, 12,31)
    if vd in pasqua:
        return pasqua[pasqua.index(vd)-2]
    if vd in pasquetta:
        return pasquetta[pasquetta.index(vd)-2]
####################################################################################################
def GetRicDate(dtf):
### Get the "most similar" corresponding date, given history   
    strm = str(dtf.month) if len(str(dtf.month)) > 1 else "0" + str(dtf.month)

    monthrange = pd.date_range(str(dtf.year) + '-' + strm + '-01', str(dtf.year) + '-' + strm + '-' + str(dtf.day), freq = 'D')
    todow = map(lambda date: date.weekday(), monthrange)

    monthrangey1 = pd.date_range(str(dtf.year - 1) + '-' + strm + '-01', str(dtf.year - 1) + '-' + strm + '-' + str(calendar.monthrange(dtf.year - 1, dtf.month)[1]), freq = 'D')
    todowy1 = map(lambda date: date.weekday(), monthrangey1)
    
    dow = dtf.weekday()
    dow_counter = np.where(np.array(todow) == dow)[0].size - 1
    
    dow_countery11 = [i for i, x in enumerate(todowy1) if x == dow]
    
#    if dtf == datetime.date(2017,10,29):
#        return datetime.date(2016,10,30)
#    if dtf == datetime.date(2017,3,26):
#        return datetime.date(2016,3,27)
    
    
    if AddHolidaysDate(dtf) == 0:
        try:
            dow_countery1 = dow_countery11[dow_counter]
        except:
            dow_countery1 = dow_countery11[-1]
        if AddHolidaysDate(monthrangey1[dow_countery1]) == 0:
            return monthrangey1[dow_countery1]
        else:
            try: 
                cand = monthrangey1[dow_countery11[dow_countery11.index(dow_countery1)-1]]
            except:
                cand = monthrangey1[dow_countery11[dow_countery11.index(dow_countery1)+1]]
            return cand    
    else:
        return GetRicHoliday(dtf)
####################################################################################################
def GetRicDate2(dtf):
### Get the "most similar" corresponding date, given history   
    strm = str(dtf.month) if len(str(dtf.month)) > 1 else "0" + str(dtf.month)

    monthrange = pd.date_range(str(dtf.year) + '-' + strm + '-01', str(dtf.year) + '-' + strm + '-' + str(dtf.day), freq = 'D')
    todow = map(lambda date: date.weekday(), monthrange)

    monthrangey1 = pd.date_range(str(dtf.year - 2) + '-' + strm + '-01', str(dtf.year - 2) + '-' + strm + '-' + str(calendar.monthrange(dtf.year - 2, dtf.month)[1]), freq = 'D')
    todowy1 = map(lambda date: date.weekday(), monthrangey1)
    
    dow = dtf.weekday()
    dow_counter = np.where(np.array(todow) == dow)[0].size - 1
    
    dow_countery11 = [i for i, x in enumerate(todowy1) if x == dow]

    if dtf == datetime.date(2016,10,30):
        return datetime.date(2015,10,25)
    if dtf == datetime.date(2016,3,27):
        return datetime.date(2015,3,29)
    
    if AddHolidaysDate(dtf) == 0:
        try:
            dow_countery1 = dow_countery11[dow_counter]
        except:
            dow_countery1 = dow_countery11[-1]
        if AddHolidaysDate(monthrangey1[dow_countery1]) == 0:
            return monthrangey1[dow_countery1]
        else:
            try: 
                cand = monthrangey1[dow_countery11.index(dow_countery1)+1]
            except:
                cand = monthrangey1[dow_countery11.index(dow_countery1)-1]
            return cand    
    else:
        return GetRicHoliday2(dtf)
####################################################################################################
def Get_MeanDependencyWithTemperature(df, meteo):
### @PARAM: df is the dataset I want to compute the dependency of: it could be Sample, OOS or Zonal and
### needs to be a time series
    
#    final = min(max(df.index.date), max(meteo.index.date))
    final = min(max(df.index.date), max(meteo.index))
    strm = str(final.month) if len(str(final.month)) > 1 else "0" + str(final.month)
    strd = str(final.day) if len(str(final.day)) > 1 else "0" + str(final.day)
    final_date = str(final.year) + '-' + strm + '-' + strd
    
    #di = max(min(df.index.date), min(meteo.index.date))  
    di = max(min(df.index.date), min(meteo.index)) 
    basal_cons = min(df.ix[df.index.month == 4].mean().values.ravel()[0] ,df.ix[df.index.month == 5].mean().values.ravel()[0])   
    
    dts = OrderedDict()
    indices = pd.date_range(di, final_date, freq = 'D')
    for i in indices:
        bri = Bridge(i.date())
        dls = StartsDaylightSaving(i.date())
        edls = EndsDaylightSaving(i.date())
        ll = []        
        dvector = np.repeat(0, 7)
        mvector = np.repeat(0,12)
        wd = i.weekday()        
        dvector[wd] = 1
        mvector[(i.month-1)] = 1
        dy = i.timetuple().tm_yday
#        Tmax = meteo['Tmax'].ix[meteo['DATA'] == i.date()].values.ravel()[0]
#        rain = meteo['PIOGGIA'].ix[meteo['DATA'] == i.date()].values.ravel()[0]
#        wind = meteo['VENTOMEDIA'].ix[meteo['DATA'] == i.date()].values.ravel()[0]
        Tmax = meteo['Tmax'].ix[meteo.index == i.date()].values.ravel()[0]
        rain = meteo['pioggia'].ix[meteo.index == i.date()].values.ravel()[0]
        wind = meteo['vento'].ix[meteo.index == i.date()].values.ravel()[0]        
        hol = AddHolidaysDate(i.date())
        ll.extend(dvector.tolist())
        ll.extend(mvector.tolist())
        ll.extend([dy, Tmax, rain, wind, hol, bri, dls, edls])
#        ll.append(meteo['Tmax'].ix[meteo.index.date == i.date()].mean() - meteo['Tmax'].ix[meteo.index.date == (i.date() - datetime.timedelta(days = 7))].mean())

        y = df.ix[i].mean()
        ### detrend by the mean?               
        
        ll.append(y - basal_cons)        
        dts[i] =  ll
        
    dts = pd.DataFrame.from_dict(dts, orient = 'index')
    dts.columns =[['Lun','Mar','Mer','Gio','Ven','Sab','Dom','Gen','Feb','March','Apr','Mag','Giu','Lug','Ago','Set','Ott','Nov','Dic',
    'pday','tmax','pioggia','vento','hol','ponte','dls','edls','y']]
    return dts
####################################################################################################
def DependencyWithTemperature(df, dtf, meteo):
### @PARAM: df is the dataset I want to compute the dependency of: it could be Sample, OOS or Zonal and
### needs to be a time series
    
    #basal_cons = min(df.ix[df.index.month == 4].mean().values.ravel()[0] ,df.ix[df.index.month == 5].mean().values.ravel()[0])   

    dts = OrderedDict()
    for d in dtf:
        bri = Bridge(d.date())
        dls = StartsDaylightSaving(d.date())
        edls = EndsDaylightSaving(d.date())
        ll = []        
        hvector = np.repeat(0, 24)
        dvector = np.repeat(0, 7)
        mvector = np.repeat(0,12)
        wd = d.weekday()        
        dvector[wd] = 1
        h = d.hour
        hvector[h] = 1
        mvector[(d.month-1)] = 1
        dy = d.timetuple().tm_yday
        Tmax = meteo['Tmax'].ix[meteo.index.date == d.date()].values.ravel()[0]
        rain = meteo['pioggia'].ix[meteo.index.date == d.date()].values.ravel()[0]
        wind = meteo['vento'].ix[meteo.index.date == d.date()].values.ravel()[0]
        hol = AddHolidaysDate(d.date())
        ll.extend(dvector.tolist())
        ll.extend(mvector.tolist())
        ll.extend(hvector.tolist())        
        ll.extend([dy, Tmax, rain, wind, hol, bri, dls, edls])
    #        ll.append(meteo['Tmax'].ix[meteo.index.date == i.date()].mean() - meteo['Tmax'].ix[meteo.index.date == (i.date() - datetime.timedelta(days = 7))].mean())
    
            
        dts[d] =  ll
        
    dts = pd.DataFrame.from_dict(dts, orient = 'index')
    dts.columns =[['Lun','Mar','Mer','Gio','Ven','Sab','Dom','Gen','Feb','March','Apr','Mag','Giu','Lug','Ago','Set','Ott','Nov','Dic',
    't0','t1','t2','t3','t4'	,'t5'	,'t6','t7','t8','t9','t10','t11','t12','t13','t14','t15','t16','t17','t18'	,'t19','t20','t21','t22','t23',
    'pday','tmax','pioggia','vento','hol','ponte','dls','edls']]
    return dts
####################################################################################################
def ModelDependencyTemperature(df, meteo):
    
    DWT = Get_MeanDependencyWithTemperature(df, meteo)
    
    DWT = DWT.sample(frac = 1)

    rf = RandomForestRegressor(criterion = 'mse', n_estimators = 3000, n_jobs = 1)
    rf.fit(DWT[DWT.columns[:28]], DWT['y'])
    
    print r2_score(DWT['y'], rf.predict(DWT[DWT.columns[:51]]))
    print mean_squared_error(DWT['y'], rf.predict(DWT[DWT.columns[:51]]))
    
#    print r2_score(ESte['y'], rf.predict(ESte[ESte.columns[:28]]))
####################################################################################################
def Get_WModel(df, meteo, zona, what):
### @BRIEF: function to compute the model describing the weather-time dependency
    DTS = Get_MeanDependencyWithTemperature(df, meteo)
    rf = RandomForestRegressor(criterion = 'mse', n_estimators = 3000, n_jobs = 1)        
    DTStrain = DTS.sample(frac = 1)
    rf.fit(DTStrain[DTStrain.columns[:51]], DTStrain['y'])
    print r2_score(DTStrain['y'], rf.predict(DTStrain[DTStrain.columns[:51]]))
    print mean_squared_error(DTStrain['y'], rf.predict(DTStrain[DTStrain.columns[:51]]))
    
    joblib.dump('C:/Users/utente/Documents/Sbilanciamento/model_weather_' + what + '_' + zona + '.pkl')
    
    return rf
####################################################################################################    
def computePerditaCRPP(d, podlist):

    perdite = pd.read_excel('C:/Users/utente/Documents/Sbilanciamento/perdite.xlsx')               
    dm = d.month

    strm = str(dm) if len(str(dm)) > 1 else "0" + str(dm)
    crpp = pd.read_excel('H:/Energy Management/02. EDM/01. MISURE/4. CRPP/2017/' + strm + '-2017/_All_CRPP_' + strm + '_2017.xlsx')
    crpp = crpp.ix[crpp['Trattamento_'+strm] == 'O']    
    
    ps = []
    cons = []
    for p in podlist:
        per = perdite['PERDITE'].ix[perdite['POD'] == p].values.ravel()    
        
        if crpp.ix[crpp['POD'] == p].shape[0] > 0:
            cons.append(crpp['CONSUMO_TOT'].ix[crpp['POD'] == p].values.ravel()[0])
            if per.size > 1:
                per = per[np.where(per == max(per))[0]]
                ps.append(per[0])
            else:
                ps.append(per[0])
        else:
            next
    
    wm = np.sum(np.array(ps) * np.array(cons)/1000)/np.sum(np.array(cons)/1000)
    
    return wm
####################################################################################################
def computePerdita(d, podlist):

    perdite = pd.read_excel('C:/Users/utente/Documents/Sbilanciamento/perdite.xlsx')               
    dm = d.month

    ml = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    mn = ['01','02','03','04','05','06','07','08','09','10','11','12']
    
    strm = str(dm) if len(str(dm)) > 1 else "0" + str(dm)
    crpp = pd.read_excel('H:/Energy Management/02. EDM/01. MISURE/4. CRPP/2017/' + strm + '-2017/_All_CRPP_' + strm + '_2017.xlsx')
    crppa = pd.read_excel('H:/Energy Management/02. EDM/01. MISURE/CRPP_' + ml[mn.index(strm)] + '_2017_artigianale.xlsx')    
    
    ps = []
    cons = []
    for p in podlist:
        per = perdite['PERDITE'].ix[perdite['POD'] == p].values.ravel()    
        
        if crppa.ix[crppa['pod'] == p].shape[0] > 0:
            cons.append(crppa['Consumo_' + strm].ix[crppa['pod'] == p].values.ravel()[0])
            if per.size > 1:
                per = per[np.where(per == max(per))[0]]
                ps.append(per[0])
            else:
                ps.append(per[0])
        elif crppa.ix[crppa['pod'] == p].shape[0] == 0 and crpp.ix[crpp['POD'] == p].shape[0] > 0:
            cons.append(crpp['CONSUMO_TOT'].ix[crpp['POD'] == p].values.ravel()[0])
            if per.size > 1:
                per = per[np.where(per == max(per))[0]]
                ps.append(per[0])
            else:
                ps.append(per[0])
        else:
            next
    
    wm = np.sum(np.array(ps) * np.array(cons)/1000)/np.sum(np.array(cons)/1000)
    
    return wm
####################################################################################################
def EstimateUnknownCurve(rical, db, p, crpp, d):
    
    perdite = pd.read_excel('C:/Users/utente/Documents/Sbilanciamento/perdite.xlsx')       
    per = perdite['PERDITE'].ix[perdite['POD'] == p].values.ravel()
    
    if per.size > 1:
        per = per[np.where(per == max(per))[0]]
    
    pred = [p, d]
    dm = d.month
    strm = str(dm) if len(str(dm)) > 1 else "0" + str(dm)
    if p in list(set(db['POD'].values.ravel())):
        #dbn = db.drop(db.ix[db['POD'] == p].index)
        npl = list(set(db['POD'].values.ravel()).difference(p))
        npl = list(set(npl).intersection(set(rical.columns)))
    else:
        npl = list(set(crpp['POD'].ix[crpp['Trattamento_' + strm] == 'O']).difference(set(db['POD'].values.ravel()).union(p)))
        npl = list(set(npl).intersection(set(rical.columns)))
    ricals = rical[npl].sum(axis = 1)
    #shape = ricals/np.trapz(ricals.values.ravel())
    TOT = ricals.sum()
    hourly_perc = ricals.values.ravel()/TOT
    HP = pd.DataFrame({'perc': hourly_perc})
    HP = HP.set_index(pd.date_range('2017-01-01','2018-12-31', freq = 'H')[:ricals.shape[0]])
    #SHAPE = pd.DataFrame({'shape': shape}).set_index(pd.date_range('2017-01-01','2017-12-31', freq = 'H')[:shape.size])
    prop_perc = HP.ix[HP.index.date == d]
    #prop_shape = SHAPE.ix[SHAPE.index.date == d]
    tot = crpp['CONSUMO_TOT'].ix[crpp['POD'] == p].values.ravel()
    prop = tot/1000 * prop_perc.values.ravel()*per
    
    pred.extend(prop.tolist())
    return pred
####################################################################################################
def setRicalIndex(rical):
    dl = []
    for i in range(rical.shape[0]):
        timestring = str(rical['Giorno'].ix[i])[:10] 
        dt = datetime.datetime.strptime(timestring, '%Y-%m-%d')
        dt = dt.replace(hour = int(rical['Ora'].ix[i]))
        dl.append(dt)
    return dl
####################################################################################################
def CheckMissingData(pod, rical, d):
    if not pod in rical.columns:
        return True
    dl = setRicalIndex(rical)
    val = rical[pod].values.ravel()
    RP = pd.DataFrame({'X': val}).set_index(pd.date_range(dl[0].date(), dl[-1].date() + datetime.timedelta(days = 1), freq = 'H')[:len(dl)])
    if np.isnan(RP.ix[RP.index.date == d].values.ravel()).all():
        check = (not np.isnan(RP.ix[RP.index.date == d - datetime.timedelta(days = 1)].values.ravel()).all()) and (not np.isnan(RP.ix[RP.index.date == d + datetime.timedelta(days = 1)].values.ravel()).all())
        if check:
            return True
        else:
            return False
    else:
        return False
####################################################################################################
def TakerWithout(rical, pdo, sos, p, d):
###             
    ricd = GetRicDate(d)
    pred = [p, d]
    pdop = pdo.ix[pdo['POD'] == p]
    sosp = sos.ix[sos['Pod'] == p]
    if p in rical.columns:
        ricalp = rical[p].ix[rical['Giorno'] == d].values.ravel()
        if ricalp.size == 24:
            print 'taken from Rical'
            pred.extend(ricalp.tolist())
         
    elif not p in rical.columns and pdop.shape[0] > 0:
        print 'taken from PDO'
        pdopd = pdop.ix[pdo['Giorno'] == ricd]
        if pdopd.shape[0] > 0:
            val = pdopd[pdopd.columns[4:]].values.ravel()
            pred.extend(val.tolist())
                  
    elif not p in rical.columns and pdop.shape[0] == 0 and sosp.shape[0] > 0:
        print 'taken from SOS'                
        sospd = sosp.ix[sosp['Giorno'] == ricd]
        if sospd.shape[0] > 0:
            val = sospd[sospd.columns[2:]].values.ravel()
            pred.extend(val.tolist())
              
    else:
        print 'not found'                    

    return pred                
####################################################################################################
def Taker(rical, pdo, sos, p, d):
###             
    perdite = pd.read_excel('C:/Users/utente/Documents/Sbilanciamento/perdite.xlsx')       
    per = perdite['PERDITE'].ix[perdite['POD'] == p].values.ravel()
    
    if per.size > 1:
        per = per[np.where(per == max(per))[0]]
    
    ricd = GetRicDate(d)
    pred = [p, d]
    pdop = pdo.ix[pdo['POD'] == p]
    sosp = sos.ix[sos['Pod'] == p]
    if p in rical.columns:
        ricalp = rical[p].ix[rical['Giorno'] == d].values.ravel()*per
        if ricalp.size == 24:
            print 'taken from Rical'
            pred.extend(ricalp.tolist())
         
    elif not p in rical.columns and pdop.shape[0] > 0:
        print 'taken from PDO'
        pdopd = pdop.ix[pdo['Giorno'] == ricd]
        if pdopd.shape[0] > 0:
            val = pdopd[pdopd.columns[4:]].values.ravel()*np.repeat(per, 24)
            pred.extend(val.tolist())
                  
    elif not p in rical.columns and pdop.shape[0] == 0 and sosp.shape[0] > 0:
        print 'taken from SOS'                
        sospd = sosp.ix[sosp['Giorno'] == ricd]
        if sospd.shape[0] > 0:
            val = sospd[sospd.columns[2:]].values.ravel()*per
            pred.extend(val.tolist())
              
    else:
        print 'not found'                    

    return pred                
####################################################################################################
def TakerWithout2(rical, pdo, sos, p, d):
###             
    ricd = GetRicDate(d)
    pred = [p, d]
    pdop = pdo.ix[pdo['POD'] == p]
    sosp = sos.ix[sos['Pod'] == p]
    if p in rical.columns:
        CMD = CheckMissingData(p, rical, d)
        if not CMD:
            ricalp = rical[p].ix[rical['Giorno'] == d].values.ravel()
            if ricalp.size == 24:
             print 'taken from Rical'
             pred.extend(ricalp.tolist())
         
    elif (not p in rical.columns and pdop.shape[0] > 0) or (CMD):
        print 'taken from PDO'
        pdopd = pdop.ix[pdo['Giorno'] == ricd.date()]
        if pdopd.shape[0] > 0:
            val = pdopd[pdopd.columns[4:]].values.ravel()
            pred.extend(val.tolist())
                  
    elif not p in rical.columns and pdop.shape[0] == 0 and sosp.shape[0] > 0:
        print 'taken from SOS'                
        sospd = sosp.ix[sosp['Giorno'] == ricd.date()]
        if sospd.shape[0] > 0:
            val = sospd[sospd.columns[2:]].values.ravel()
            pred.extend(val.tolist())
              
    else:
        print 'not found'                    

    return pred                
####################################################################################################
def Taker2(rical, pdo, sos, p, d):
###             
    perdite = pd.read_excel('C:/Users/utente/Documents/Sbilanciamento/perdite.xlsx')       
    per = perdite['PERDITE'].ix[perdite['POD'] == p].values.ravel()
    
    if per.size > 1:
        per = per[np.where(per == max(per))[0]]
    
    ricd = GetRicDate(d)
    pred = [p, d]
    pdop = pdo.ix[pdo['POD'] == p]
    sosp = sos.ix[sos['Pod'] == p]
    if p in rical.columns:
        CMD = CheckMissingData(p, rical, d)
        if not CMD:
            ricalp = rical[p].ix[rical['Giorno'] == d].values.ravel()*per
            if ricalp.size == 24:
                print 'taken from Rical'
                pred.extend(ricalp.tolist())
         
    elif (not p in rical.columns and pdop.shape[0] > 0) or (CMD):
        print 'taken from PDO'
        pdopd = pdop.ix[pdo['Giorno'] == ricd.date()]
        if pdopd.shape[0] > 0:
            val = pdopd[pdopd.columns[4:]].values.ravel()*per
            pred.extend(val.tolist())
                  
    elif not p in rical.columns and pdop.shape[0] == 0 and sosp.shape[0] > 0:
        print 'taken from SOS'                
        sospd = sosp.ix[sosp['Giorno'] == ricd.date()]
        if sospd.shape[0] > 0:
            val = sospd[sospd.columns[2:]].values.ravel()*per
            pred.extend(val.tolist())
              
    else:
        print 'not found'                    

    return pred                
####################################################################################################
def PredictOOS_Base(pdo, sos, rical, db, zona, dts):
    ### @PARAM: crpp is the "classic" crpp and dts could be a vector of dates
    if not isinstance(dts, list):
        dts = [dts]
    #sos = pd.read_hdf("C:/Users/utente/Documents/Sbilanciamento/sos_elaborati_finiti.h5")
#    pdo = pd.read_hdf("C:/Users/utente/Documents/Sbilanciamento/DB_misure.h5")
#    sos = pd.read_excel("C:/Users/utente/Documents/Sbilanciamento/sos_elaborati_finiti.xlsx")
#    pdo = pd.read_excel("C:/Users/utente/Documents/Sbilanciamento/DB_misure.xlsx")    
#    rical = pd.read_excel("C:/Users/utente/Documents/Sbilanciamento/Rical.xlsx", sheetname = zona) 
    
    sos['Giorno'] = pd.to_datetime(sos['Giorno'].values.ravel().tolist()).date   
    pdo['Giorno'] = pd.to_datetime(pdo['Giorno'].values.ravel().tolist()).date

    dm = scipy.stats.mode(map(lambda date: date.month, dts))[0][0]
    strm = str(dm) if len(str(dm)) > 1 else "0" + str(dm)
    crpp = pd.read_excel('H:/Energy Management/02. EDM/01. MISURE/4. CRPP/2017/' + strm + '-2017/_All_CRPP_' + strm + '_2017.xlsx')
    
    db = db.ix[db['Area'] == zona]
    crpp = crpp.ix[crpp['ZONA'] == zona]
    PRED = OrderedDict()     
    
    missing_counter = 0
    counter = 0
    missing_pod = []
    for d in dts:
        strm = str(d.month) if len(str(d.month)) > 1 else "0" + str(d.month)
        crpp_oos = crpp.ix[crpp['Trattamento_' + strm] == 'O']
        podlist = list(set(crpp_oos['POD'].values.ravel().tolist()).difference(set(db['POD'].ix[db['Giorno'] == max(db['Giorno'])])))
        for p in podlist:
            print p in rical.columns
            pred = Taker(rical, pdo, sos, p, d)
            
            if len(pred) < 26:                
                print '#pod not found: {}'.format(missing_counter)
                missing_pod.append((p,d))
                pred = EstimateUnknownCurve(rical, db, p, crpp, d)
            
            PRED[counter] = pred
            counter += 1
    
    PRED = pd.DataFrame.from_dict(PRED, orient = 'index')
    PRED = PRED[PRED.columns[:26]]
    PRED.columns = ['POD', 'Giorno', '1','2','3', '4', '5', '6', '7', '8', '9', '10',
                '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24']    
    Pred = PRED.fillna(0)
    return Pred
####################################################################################################
def PredictS_Base(pdo, sos, rical, db, zona, dts):
    ### @PARAM: crpp is the "classic" crpp and dts could be a vector of dates
    if not isinstance(dts, list):
        dts = [dts]
#    sos = pd.read_hdf("C:/Users/utente/Documents/Sbilanciamento/sos_elaborati_finiti.h5")
#    pdo = pd.read_hdf("C:/Users/utente/Documents/Sbilanciamento/DB_misure.h5")
#    rical = pd.read_excel("C:/Users/utente/Documents/Sbilanciamento/Rical.xlsx", sheetname = zona)    
    
#    sos['Giorno'] = pd.to_datetime(sos['Giorno'].apply(lambda date: datetime.datetime.strptime(date, '%Y-%m-%d')).values.ravel().tolist()).date   
#    pdo['Giorno'] = pd.to_datetime(pdo['Giorno'].apply(lambda date: datetime.datetime.strptime(date, '%Y-%m-%d')).values.ravel().tolist()).date
    sos['Giorno'] = pd.to_datetime(sos['Giorno'].values.ravel().tolist()).date   
    pdo['Giorno'] = pd.to_datetime(pdo['Giorno'].values.ravel().tolist()).date

    dm = scipy.stats.mode(map(lambda date: date.month, dts))[0][0]
    strm = str(dm) if len(str(dm)) > 1 else "0" + str(dm)
    crpp = pd.read_excel('H:/Energy Management/02. EDM/01. MISURE/4. CRPP/2017/' + strm + '-2017/_All_CRPP_' + strm + '_2017.xlsx')
    podcrpp = set(crpp['POD'].ix[crpp['ZONA'] == zona])
    
    db = db.ix[db['Area'] == zona]
    PRED = OrderedDict()        
    
    missing_counter = 0
    counter = 0
    missing_pod = []
    for d in dts:
        podlist = list(set(db['POD'].ix[db['Giorno'] == max(db['Giorno'])]).intersection(podcrpp))
        for p in podlist:
            pred = Taker(rical, pdo, sos, p, d)
            if len(pred) < 26:                
                print '#pod not found: {}'.format(missing_counter)
                missing_pod.append((p,d))
                pred = EstimateUnknownCurve(rical, db, p, crpp, d)
            
            PRED[counter] = pred
            counter += 1
    print '#pod not found: {}'.format(missing_counter)
    
    ### for the pods not found (e.g. we didn't receive the sample), look for the last "useful" day of measurement
    if len(missing_pod) > 0:    
        for mp in missing_pod:
            dbp = db.ix[db['POD'] == mp]
            for d in dts:
                pred = [d]
                dow = d.weekday()
                if dow < 5:
                    useful_days = map(lambda date: date.weekday(), dbp['Giorno'].values.ravel())
                    for i in range(len(useful_days)-1, 0, -1):
                        if useful_days[i] < 5:
                            pred.extend(dbp[dbp.columns[3:]].ix[i].values.ravel().tolist())
                            PRED[mp] = pred
                            break
                else:
                    ud = [x for x in dbp['Giorno'].values.ravel() if x.weekday() == dow]
                    pred.extend(dbp[dbp.columns[3:]].ix[dbp['Giorno'] == ud[-1]].values.ravel().tolist())
                    PRED[mp] = pred
                    break
    
    PRED = pd.DataFrame.from_dict(PRED, orient = 'index')    
    PRED.columns = ['POD','Giorno', '1','2','3', '4', '5', '6', '7', '8', '9', '10',
                '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24']
                
    PRED = PRED.fillna(0)
    return PRED
####################################################################################################
def PredictS_BaseWithout(pdo, sos, rical, db, zona, dts):
    ### @PARAM: crpp is the "classic" crpp and dts could be a vector of dates
    if not isinstance(dts, list):
        dts = [dts]
#    sos = pd.read_hdf("C:/Users/utente/Documents/Sbilanciamento/sos_elaborati_finiti.h5")
#    pdo = pd.read_hdf("C:/Users/utente/Documents/Sbilanciamento/DB_misure.h5")
#    rical = pd.read_excel("C:/Users/utente/Documents/Sbilanciamento/Rical.xlsx", sheetname = zona)    
    
#    sos['Giorno'] = pd.to_datetime(sos['Giorno'].apply(lambda date: datetime.datetime.strptime(date, '%Y-%m-%d')).values.ravel().tolist()).date   
#    pdo['Giorno'] = pd.to_datetime(pdo['Giorno'].apply(lambda date: datetime.datetime.strptime(date, '%Y-%m-%d')).values.ravel().tolist()).date
    sos['Giorno'] = pd.to_datetime(sos['Giorno'].values.ravel().tolist()).date   
    pdo['Giorno'] = pd.to_datetime(pdo['Giorno'].values.ravel().tolist()).date

    dm = scipy.stats.mode(map(lambda date: date.month, dts))[0][0]
    strm = str(dm) if len(str(dm)) > 1 else "0" + str(dm)
    crpp = pd.read_excel('H:/Energy Management/02. EDM/01. MISURE/4. CRPP/2017/' + strm + '-2017/_All_CRPP_' + strm + '_2017.xlsx')
    podcrpp = set(crpp['POD'].ix[crpp['ZONA'] == zona])
    
    db = db.ix[db['Area'] == zona]
    PREDW = OrderedDict()        
    
    missing_counter = 0
    counter = 0
    missing_pod = []
    for d in dts:
        podlist = list(set(db['POD'].ix[db['Giorno'] == max(db['Giorno'])]).intersection(podcrpp))
        for p in podlist:
            pred = TakerWithout(rical, pdo, sos, p, d)
            if len(pred) < 26:                
                print '#pod not found: {}'.format(missing_counter)
                missing_pod.append((p,d))
                pred = EstimateUnknownCurve(rical, db, p, crpp, d)
            
            PREDW[counter] = pred
            counter += 1
    print '#pod not found: {}'.format(missing_counter)
    
#    ### for the pods not found (e.g. we didn't receive the sample), look for the last "useful" day of measurement
#    if len(missing_pod) > 0:    
#        for mp in missing_pod:
#            dbp = db.ix[db['POD'] == mp]
#            for d in dts:
#                pred = [d]
#                dow = d.weekday()
#                if dow < 5:
#                    useful_days = map(lambda date: date.weekday(), dbp['Giorno'].values.ravel())
#                    for i in range(len(useful_days)-1, 0, -1):
#                        if useful_days[i] < 5:
#                            pred.extend(dbp[dbp.columns[3:]].ix[i].values.ravel().tolist())
#                            PREDW[mp] = pred
#                            break
#                else:
#                    ud = [x for x in dbp['Giorno'].values.ravel() if x.weekday() == dow]
#                    pred.extend(dbp[dbp.columns[3:]].ix[dbp['Giorno'] == ud[-1]].values.ravel().tolist())
#                    PREDW[mp] = pred
#                    break
    
    PREDW = pd.DataFrame.from_dict(PREDW, orient = 'index')    
    PREDW.columns = ['POD','Giorno', '1','2','3', '4', '5', '6', '7', '8', '9', '10',
                '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24']
                
    PREDW = PREDW.fillna(0)
    return PREDW
####################################################################################################
def Get_Cluster(db, zona):
    
    db = db.ix[db['Area'] == zona]
    db = db.set_index(db['Giorno'])
    counts = db['POD'].resample('D').count()
    dbagg = (0.001)*(db.resample('D').sum())
    dbagg['DOW'] = map(lambda date: date.weekday(), dbagg.index)
    return dbagg, counts
####################################################################################################
def MTCorrection(PREDW, db, dtf, zona, Sample):
#    If today's Tuesday and yestersay's Monday: --> use this function
#    So it's Monday to Wednesday

#    
#    sampled = sample.ix[sample.index.date == md].values.ravel()
#    Sampled = Sample.ix[Sample.index.date == md].values.ravel()
    
#    Sample = Get_SampleAsTS(db, zona)
    md = max(Sample.index.date)
    Sampled = Sample.ix[Sample.index.date == md].values.ravel()
    
    db2tt, n = Get_Cluster(db, zona)
    
    dtw = dtf.weekday()
    MW = (dtw == 2)
    if MW and (AddHolidaysDate(dtf) == 0) and (AddHolidaysDate(dtf - datetime.timedelta(days = 2)) == 0):
        db2tt = db2tt.ix[db2tt['DOW'] <= 2]
        db2tt = db2tt.ix[db2tt['DOW'] != 1]
        holindex = list(map(AddHolidaysDate, db2tt.index))
        if not sum(holindex[-10:]) == 0:
            tbr = []
            for i in range(len(holindex)):
                if holindex[i] == 1 and db2tt.DOW.ix[i] == 0:
                    tbr.append(i)
                    tbr.append(i + 1)
                elif holindex[i] == 1 and db2tt.DOW.ix[i] == 2:
                    tbr.append(i - 1)
                    tbr.append(i)
                else:
                    next
            tbr = list(set(tbr))
        
            db2tt = db2tt.drop(db2tt.index[tbr])
            db2tt = db2tt.ix[-10:] ### approximately a month back
        else:
            db2tt = db2tt.ix[-10:]
        
        db2mon = db2tt.ix[db2tt.DOW == 0]        
        db2tue = db2tt.ix[db2tt.DOW == 2]        
        
        if not np.array(db2mon.std().values.ravel() > 10).any() and not np.array(db2tue.std().values.ravel() > 10).any():
            #if db2mon.shape[0] == db2tue.shape[0]:
            db2mon = db2mon.reset_index(drop = True)             
            db2tue = db2tue.reset_index(drop = True)             
            SB = (db2tue[db2tue.columns[:24]] - db2mon[db2mon.columns[:24]])/db2mon[db2mon.columns[:24]]
            EV = SB.mean().values.ravel()
            SD = SB.std().values.ravel()
                
            pMAE = (-PREDW.sum().values.ravel()/1000 + Sampled*(1 + EV))/(PREDW.sum().values.ravel()/1000)
            alpha = np.repeat(0.0, 7)
            for i in range(7):
                threshold = min([abs(SD[i]), 0.15])
                if pMAE[i] > threshold:
                    alpha[i] += -1 + Sampled[i]*(1 + EV[i])/((1 + threshold)*(PREDW.sum().values.ravel()[i]/1000))
                elif pMAE[i] < -threshold:
                    alpha[i] += 1 - Sampled[i]*(1 + EV[i])/((1 - threshold)*(PREDW.sum().values.ravel()[i]/1000))
                else:
                    alpha[i] += 0.0    
            proposal = np.concatenate((1.0 + alpha, np.repeat(1.0,17))) * PREDW.sum().values.ravel()/1000         
                
            Hsm = np.sqrt(np.mean(np.diff(PREDW.sum().values.ravel()/1000)/np.trapz(np.diff(PREDW.sum().values.ravel()/1000)) - np.diff(proposal/1000)/np.trapz(np.diff(proposal/1000)))**2)
            print 'Sobolev semi-norm: {}'.format(Hsm)
            if Hsm > 0.25:
                print 'the shape of the proposed forecast is quite different from the one deduced from the data'
            
            return proposal
        else:
            return PREDW.sum().values.ravel()/1000
        #    EV = (db2tue[db2tue.columns[:24]].mean() - db2mon[db2mon.columns[:24]].mean())/db2mon[db2mon.columns[:24]].mean()
            # return base prediction
####################################################################################################
def Adjust_MPrediction(PREDW, db, dtf, zona):
### @BRIEF: adjust the martingale part base prediction. Given the sample, compare it with the last days measurement
### if the behaviour is sufficiently different, modify it to follow the last trend   
#    PREDW = PredictS_BaseWithout(db, zona, dtf)    
    tau = 0.15 ### max level of tolerated sbilanciamento
    tau2 = 1 ### 'tolerance multiplier'
    #m = scipy.stats.mode(PREDW['Giorno'].values.ravel())[0][0]
    Sample = Get_SampleAsTS(db, zona)
    sample =  pd.read_excel('C:/Users/utente/Documents/Sbilanciamento/forecast_campione_' + zona + '.xlsx')
    
    ### CHECK: variation of the process
    #### if I need to modify, the "L2" part identifies by how much the process needs to be modified,
    #### the "H1" part identifies where the process needs to be modified

    md = min([max(sample.index.date), max(Sample.index.date)])
    
    #daysdifference = (dtf - max(Sample.index.date)).days
    
#    sampled = sample.ix[sample.index.date == md].values.ravel()    
    sampled = sample.ix[sample.index.date == md].values.ravel()
    Sampled = Sample.ix[Sample.index.date == md].values.ravel()
    
    error = Sampled - sampled
    perror = error/sampled
    
    db2tt, n = Get_Cluster(db, zona)
    
#    if isinstance(dtf, list):
#        dtf = dtf[0].date()
    
    dtw = dtf.weekday()
    MW = 0 < dtw < 5
    #### if dtf is a holiday --> just correct based on weather
    if AddHolidaysDate(dtf) == 0 and MW:
        db2tt = db2tt[db2tt.columns[:24]].ix[db2tt['DOW'] == dtw]
        db2tt = db2tt.ix[-7:] ### 7 days back from the peak of autocorrelation
        if not np.array(db2tt.std().values.ravel() > 10).any():
            fcons = db2tt[db2tt.columns[:24]].mean().values.ravel() 
            fdiff = np.diff(fcons)
            fstd = db2tt[db2tt.columns[:24]].std().values.ravel()
            sbil_mean = fstd/fcons ### mean sbil of a standard deviation from the mean
            ### How is the measured sbil compared to sbil_mean?
            alpha = np.repeat(0.0, 24)
    #        for i in range(24):
    #            TAU = min(abs(sbil_mean[i]), tau)
    #            if perror[i] > TAU:
    #                print tau2*float((1 + TAU)*Sampled[i]/sampled[i])
    #                alpha[i] += tau2*float((1 + TAU)*Sampled[i]/sampled[i])
    #            elif perror[i] < -TAU:
    #                print tau2*float((1 - TAU)*Sampled[i]/sampled[i])
    #                alpha[i] += tau2*float((1 - TAU)*Sampled[i]/sampled[i])
    #            else:
    #                alpha[i] += tau2
            for i in range(24):
                TAU = min(abs(sbil_mean[i]), tau)
                if perror[i] > TAU:
                    print tau2*(Sampled[i]/float((1 + TAU)*sampled[i]) - 1)
                    alpha[i] += tau2*Sampled[i]/float((1 + TAU)*sampled[i]) - 1
                elif perror[i] < -TAU:
                    print tau2*float((1 - TAU)*Sampled[i]/sampled[i])
                    alpha[i] += tau2*(1 - (Sampled[i]/float((1 - TAU)*sampled[i])))   
                else:
                    alpha[i] += 0
                    
    #        proposal = PREDW.sum().values.ravel() * alpha #* min( [1 - (daysdifference - 1)*0.2, 1])                                     
            proposal = PREDW.sum().values.ravel() * (1.0 + alpha) #* min( [1 - (daysdifference - 1)*0.2, 1])                                     
            Hsm = np.sqrt(np.mean(fdiff/np.trapz(fdiff) - np.diff(proposal/1000)/np.trapz(np.diff(proposal/1000)))**2)
            print 'Sobolev semi-norm: {}'.format(Hsm)
            if Hsm > 0.25:
                print 'the shape of the proposed forecast is quite different from the one deduced from the data'
            
            return proposal/1000
        else:
            return PREDW.sum().values.ravel()/1000
    else:
        return PREDW.sum().values.ravel()/1000        
####################################################################################################
def Adjust_Peaks(proposal, db, dtf, zona):
### mettere rical e cercare i pod nuovi - se ce ne sono - in pdo o sos
### guarda note

    dbz = db.ix[db.Area == zona]
    
    sample_pod = list(set(dbz.POD.ix[dbz.Giorno == max(dbz.Giorno)]))
    il = [x for x in dbz.index if dbz.POD[x] in sample_pod]    
    
    DBZ = dbz.ix[il]
    DBZ = DBZ.set_index(DBZ.Giorno)
    dl = list(set([incom for incom in DBZ.index.date.tolist() if str(incom) != 'nan']))
    DBZm = DBZ.ix[DBZ.index.date > max(dl) - datetime.timedelta(days = 36)]
    DBZm = DBZm.resample('D').sum()/1000
    DBZm['DOW'] = map(lambda date: date.weekday(), DBZm.index)
    
    DBZmd = DBZm.ix[DBZm.DOW == dtf.weekday()]
    DBZmd = DBZmd.drop('DOW', axis = 1)    
    
    EV = DBZmd.mean().values.ravel()
    SD = DBZmd.std().values.ravel()
    SK = DBZmd.skew().values.ravel()
    coefs = [(-1)**np.int((SK[i] < 0)) for i in range(24)]
    ref = (EV - np.array(coefs)*SD)

    expected_error = (-proposal + ref)/(proposal)
    PKed = np.array([0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0]) * expected_error
    alpha = np.repeat(0.0, 24)
    for h in range(24):
        if  PKed[h] >= 0.15:
            alpha[h] += (ref[h]/(1.15*proposal[h])) - 1
        elif PKed[h] <= -0.15:
            alpha[h] += 1 - (ref[h]/(0.85*proposal[h]))
        else:
            alpha += 0.0
    
    if np.sum(alpha) > 0:
        print alpha
    proposal2 = proposal * (1.0 + alpha) #* min( [1 - (daysdifference - 1)*0.2, 1])                                     
    Hsm = np.sqrt(np.mean(np.diff(proposal2)/np.trapz(np.diff(proposal2)) - np.diff(proposal)/np.trapz(np.diff(proposal)))**2)
    print 'Sobolev semi-norm: {}'.format(Hsm)
    if Hsm > 0.25:
        print 'the shape of the proposed forecast is quite different from the one deduced from the data'
    
    return proposal2
####################################################################################################
def WeatherAdaptedProcess(dtf, proposal, meteo, zona, what = 'OOS'):
    
    dtf = pd.date_range(dtf, dtf + datetime.timedelta(days = 1), freq = 'H').to_pydatetime()[:24]
    vtf = DependencyWithTemperature(dtf, meteo)
    
    if what == 'SAMPLE':
        model = joblib.load('C:/Users/utente/Documents/Sbilanciamento/model_weather_SAMPLE_' + zona + '.pkl')
    else:
        model = joblib.load('C:/Users/utente/Documents/Sbilanciamento/model_weather_OOS_' + zona + '.pkl')
    
    predicted_diff = model.predict(vtf)
    proposed_mean = np.mean(proposal)
    
    if (proposed_mean + predicted_diff) <= (1.15)*proposed_mean: ### TO MODEL ###
        print 'proposed correction due to temperature: {}'.format(predicted_diff - proposed_mean)
        proposal += predicted_diff + proposed_mean
    
    return proposal
####################################################################################################
def Get_CompletePrediction(db, dtf, meteo, zona):
        
    if not isinstance(dtf, list):
        dtf = [dtf]
        
    for d in dtf:
    
        OOS = PredictOOS_Base(db, zona, d)
        S = Adjust_MPrediction(db, d, zona)  
        
        oos = OOS.sum().values.ravel()/1000
        s = S[S.columns[1:]].sum().values.ravel()/1000      
        
        oosh = WeatherAdaptedProcess(d, oos, meteo, zona, 'OOS') 
        sh = WeatherAdaptedProcess(d, s, meteo, zona, 'S')
        
        podlist = list(set(db['POD'].ix[db['Giorno'] == max(db['Giorno'])]))
        wm = computePerdita(d, podlist)        
        
        sh = pd.DataFrame({zona: sh}).set_index(pd.date_range(d, d + datetime.timedelta(days = 1), freq = 'H')[:24])
        yf = pd.DataFrame({zona: oosh + np.repeat(wm,24) * sh.values.ravel()}).set_index(pd.date_range(d, d + datetime.timedelta(days = 1), freq = 'H')[:24])
        
        if os.path.exists('C:/Users/utente/Documents/Sbilanciamento/forecast_' + zona + '.h5' ):
            forecast = pd.read_excel('C:/Users/utente/Documents/Sbilanciamento/forecast_' + zona + '.xlsx')
            forecast = forecast.append(yf)
            forecast.to_excel('C:/Users/utente/Documents/Sbilanciamento/forecast_' + zona + '.xlsx')
            sample = pd.read_excel('C:/Users/utente/Documents/Sbilanciamento/forecast_campione_' + zona + '.xlsx')
            sample = sample.append(sh)
            sample.to_excel('C:/Users/utente/Documents/Sbilanciamento/forecast_campione_' + zona + '.xlsx')
        else:
            yf.to_excel('C:/Users/utente/Documents/Sbilanciamento/forecast_' + zona + '.xlsx')
            sh.to_excel('C:/Users/utente/Documents/Sbilanciamento/forecast_campione_' + zona + '.xlsx')
        
        yf.to_excel('C:/Users/utente/Documents/Sbilanciamento/forecast_' + zona + '_' + str(d) + '.xlsx')
        sh.to_excel('C:/Users/utente/Documents/Sbilanciamento/forecast_campione_' + zona + '_' + str(d) + '.xlsx')   
    return 1
####################################################################################################
def SingleSaver(Pred, proposal, zona, dtf, db, crpp):
    
    podcrpp = set(crpp['POD'].ix[crpp['ZONA'] == zona])
    
    db = db.ix[db['Area'] == zona]
    podlist = list(set(db['POD'].ix[db['Giorno'] == max(db['Giorno'])]).intersection(podcrpp))    
    
    wm = computePerditaCRPP(dtf, podlist)    
    
    Pred = Pred.set_index(pd.to_datetime(Pred['Giorno']))
    Pred2 = Pred.resample('D').sum()
    oosh = Pred2.ix[dtf].values.ravel()/1000
    shw = proposal
    shw = pd.DataFrame({zona: shw}).set_index(pd.date_range(dtf, dtf + datetime.timedelta(days = np.floor(shw.size/24)), freq = 'H')[:shw.size])    
    
    shw = shw.ix[shw.index.date == dtf]
    oosh = pd.DataFrame({zona: oosh}).set_index(pd.date_range(min(Pred2.index.date), max(Pred2.index.date) + datetime.timedelta(days = 1), freq = 'H')[:oosh.size])    
    yf = pd.DataFrame({zona: oosh.ix[oosh.index.date == dtf].values.ravel() + wm*shw.values.ravel()}).set_index(pd.date_range(dtf, dtf + datetime.timedelta(days = 1), freq = 'H')[:24])
    
#    plt.figure();shw.plot()
#    plt.figure();yf.plot()
    shw.to_excel('H:/Energy Management/20. Strutture blocchi forecast/Dati per MO/Forecast davide/forecast_campione_SP_' + zona + '_' + str(dtf) + '.xlsx')   
        
    yf.to_excel('H:/Energy Management/20. Strutture blocchi forecast/Dati per MO/Forecast davide/forecast_' + zona + '_' + str(dtf) + '.xlsx')
    
    if os.path.exists('C:/Users/utente/Documents/Sbilanciamento/forecast_' + zona + '.xlsx' ):        
        sample = pd.read_excel('C:/Users/utente/Documents/Sbilanciamento/forecast_campione_' + zona + '.xlsx')
        sample = sample.append(shw)
        sample.to_excel('C:/Users/utente/Documents/Sbilanciamento/forecast_campione_' + zona + '.xlsx')
        forecast = pd.read_excel('C:/Users/utente/Documents/Sbilanciamento/forecast_' + zona + '.xlsx')
        forecast = forecast.append(yf)
        forecast.to_excel('C:/Users/utente/Documents/Sbilanciamento/forecast_' + zona + '.xlsx')
    else:
        shw.to_excel('C:/Users/utente/Documents/Sbilanciamento/forecast_campione_' + zona + '.xlsx')
        yf.to_excel('C:/Users/utente/Documents/Sbilanciamento/forecast_' + zona + '.xlsx')
####################################################################################################
def Saver(Pred, proposal, zona, dtf, db):
    
    if not isinstance(dtf, list):
        dm = dtf.month
        print dm
        strm = str(dm) if len(str(dm)) > 1 else "0" + str(dm)
        print strm
        crpp = pd.read_excel('H:/Energy Management/02. EDM/01. MISURE/4. CRPP/2017/' + strm + '-2017/_All_CRPP_' + strm + '_2017.xlsx')
        SingleSaver(Pred, proposal, zona, dtf, db, crpp)
    else:
        if np.std(map(lambda date: date.month, dtf)) == 0:
            dm = dtf[0].month
            strm = str(dm) if len(str(dm)) > 1 else "0" + str(dm)
            crpp = pd.read_excel('H:/Energy Management/02. EDM/01. MISURE/4. CRPP/2017/' + strm + '-2017/_All_CRPP_' + strm + '_2017.xlsx')
            for d in dtf:
                SingleSaver(Pred, proposal, zona, d, db, crpp)
        else:
            for d in dtf:                
                dm = d.month
                strm = str(dm) if len(str(dm)) > 1 else "0" + str(dm)
                crpp = pd.read_excel('H:/Energy Management/02. EDM/01. MISURE/4. CRPP/2017/' + strm + '-2017/_All_CRPP_' + strm + '_2017.xlsx')
                SingleSaver(Pred, proposal, zona, d, db, crpp)
####################################################################################################
def PODremover(df, list_of_pod):
    if len(list_of_pod) > 0:
        for p in list_of_pod:
            if p in df['POD'].values.ravel().tolist():
                df = df.drop(df.ix[df['POD'] == p].index)
            else:
                next
        return df
    else:
        return df
####################################################################################################
def Activator(db, dtf, zona):

    dbz = db.ix[db.Area == zona]
    activate = False
    activateMW = False
    wd = dtf.weekday()
    hol = AddHolidaysDate(dtf)
    yesterday = dtf - datetime.timedelta(days = 2)             
    
    if max(dbz.Giorno).date() == yesterday:
        if wd in [3,4] and AddHolidaysDate(yesterday) == 0 and hol == 0:
            activate = True
        if wd == 2 and AddHolidaysDate(yesterday) == 0 and hol == 0:
            activateMW = True
        
    return activate, activateMW
####################################################################################################