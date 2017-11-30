# -*- coding: utf-8 -*-
"""
Created on Thu May 04 09:24:18 2017

@author: utente

Sbilanciamento 3 -- BACKTESTING --
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
####################################################################################################
def percentageConsumption(db, zona, di, df):
    dr = pd.date_range(di, df, freq = 'D')
    All1 = pd.read_hdf("C:/Users/utente/Documents/Sbilanciamento/CRPP_2016.h5")
    All2 = pd.read_hdf("C:/Users/utente/Documents/Sbilanciamento/CRPP_2017.h5")  
    diz = OrderedDict()
    dbz = db.ix[db["Area"] == zona]
    for d in dr:
        drm = d.month
        strm = str(drm) if len(str(drm)) > 1 else "0" + str(drm)  
        dry = d.year
        if dry == 2017 and drm <= 3:
            strm = '03'
        
        if dry == 2016:
            All = All1
        else:
            All = All2
        pods = dbz["POD"].ix[dbz["Giorno"] == d].values.ravel().tolist()
        AllX = All.ix[All["Trattamento_"+ strm] == 1]
        totd = np.sum(np.nan_to_num([AllX["CONSUMO_TOT_" + strm].ix[y] for y in AllX.index if AllX["POD"].ix[y] in pods]))/1000
        #totd = All2["CONSUMO_TOT"].ix[All2["POD"].values.ravel() in pods].sum()
        tot = AllX["CONSUMO_TOT_" + strm].sum()/1000
        p = totd/tot
        diz[d] = [p]
    diz = pd.DataFrame.from_dict(diz, orient = 'index')
    return diz
####################################################################################################
def percentageConsumption2(db, zona, di, df):
    dr = pd.date_range(di, df, freq = 'D')
    All2016 = pd.read_hdf("C:/Users/utente/Documents/Sbilanciamento/CRPP2016_artigianale.h5")
    All1 = pd.read_hdf("C:/Users/utente/Documents/Sbilanciamento/CRPP_Jan_2017_artigianale.h5")  
    All2 = pd.read_hdf("C:/Users/utente/Documents/Sbilanciamento/CRPP_Feb_2017_artigianale.h5")  
    All3 = pd.read_hdf("C:/Users/utente/Documents/Sbilanciamento/CRPP_Mar_2017_artigianale.h5")  
    All4 = pd.read_hdf("C:/Users/utente/Documents/Sbilanciamento/CRPP_Apr_2017_artigianale.h5")  
    All5 = pd.read_hdf("C:/Users/utente/Documents/Sbilanciamento/CRPP_May_2017_artigianale.h5")  
    All6 = pd.read_hdf("C:/Users/utente/Documents/Sbilanciamento/CRPP_Jun_2017_artigianale.h5")  
        
    diz = OrderedDict()
    dbz = db.ix[db["Area"] == zona]
    for d in dr:
        drm = d.month
        strm = str(drm) if len(str(drm)) > 1 else "0" + str(drm)  
        dry = d.year
        if dry == 2017 and drm == 1:
            All = All1
            strm = '03'
        elif dry == 2017 and drm == 2:
            All = All2
            strm = '03'
        elif dry == 2017 and drm == 3:
            All = All3
        elif dry == 2017 and drm == 4:
            All = All4
        elif dry == 2017 and drm == 5:
            All = All5
        elif dry == 2017 and drm == 6:
            All = All6
        elif dry == 2016:
            All = All2016
        else:
            All = All6
        pods = list(set(dbz["POD"].ix[dbz["Giorno"] == d].values.ravel().tolist()))
        AllX = All.ix[All["Trattamento_"+ strm] == 1]
        AllX = AllX.ix[AllX["zona"] == zona]
        totd = np.sum(np.nan_to_num([AllX["Consumo_" + strm].ix[y] for y in range(AllX.shape[0]) if AllX["pod"].ix[y] in pods]))
        tot = AllX["Consumo_" + strm].sum()
        p = totd/tot
        diz[d] = [p]
    diz = pd.DataFrame.from_dict(diz, orient = 'index')
    return diz
####################################################################################################
def MakeExtendedDatasetWithSampleCurve(df, db, meteo, zona):
#### @PARAM: df is the dataset from Terna, db, All zona those for computing the perc consumption
#### and the sample curve
#### @BRIEF: extended version of the quasi-omonimous function in Sbilanciamento.py
#### every day will have a dummy variable representing it
    #wdays = ['Lun', 'Mar', 'Mer', 'Gio', 'Ven', 'Sab', 'Dom']
    final = max(df.index.date)
    strm = str(final.month) if len(str(final.month)) > 1 else "0" + str(final.month)
    strd = str(final.day) if len(str(final.day)) > 1 else "0" + str(final.day)
    final_date = str(final.year) + '-' + strm + '-' + strd
    psample = percentageConsumption2(db, zona, '2016-01-01', final_date)
    psample = psample.set_index(pd.date_range('2016-01-01', final_date, freq = 'D')[:psample.shape[0]])
    dts = OrderedDict()
    df = df.ix[df.index.date >= datetime.date(2016,1,3)]
    indices = pd.date_range('2016-01-03', final_date, freq = 'H')
    for i in indices:
        bri = Bridge(i.date())
        dls = StartsDaylightSaving(i.date())
        edls = EndsDaylightSaving(i.date())
        ll = []        
        hvector = np.repeat(0, 24)
        dvector = np.repeat(0, 7)
        mvector = np.repeat(0,12)
        wd = i.weekday()        
        td = 2
        if wd == 0:
            td = 3
        cmym = db[db.columns[3:]].ix[db["Giorno"] == (i.date()- datetime.timedelta(days = td))].sum(axis = 0).values.ravel()/1000
        dvector[wd] = 1
        h = i.hour
        hvector[h] = 1
        mvector[(i.month-1)] = 1
        dy = i.timetuple().tm_yday
        Tmax = meteo['Tmax'].ix[meteo['DATA'] == i.date()].values.ravel()[0]
        rain = meteo['PIOGGIA'].ix[meteo['DATA'] == i.date()].values.ravel()[0]
        wind = meteo['VENTOMEDIA'].ix[meteo['DATA'] == i.date()].values.ravel()[0]
        hol = AddHolidaysDate(i.date())
        ps = psample.ix[psample.index.date == (i.date() - datetime.timedelta(days = td))]
        ll.extend(dvector.tolist())
        ll.extend(mvector.tolist())
        ll.extend(hvector.tolist())        
        ll.extend([dy, Tmax, rain, wind, hol, ps[0].values[0], bri, dls, edls])
        ll.extend(cmym.tolist())
        if np.where(df.index == i)[0].size > 1:
            y = df['MO [MWh]'].ix[i].sum()
        elif np.where(df.index == i)[0].size == 0:
            print "ends daylight saving"
            y = 0
        else:
            y = df['MO [MWh]'].ix[i]
        ll.extend([y])
        dts[i] =  ll
    dts = pd.DataFrame.from_dict(dts, orient = 'index')
    dts.columns = [['Lun', 'Mar', 'Mer', 'Gio', 'Ven', 'Sab', 'Dom','Jan','Feb','March','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec',
    't0','t1','t2','t3','t4','t5','t6','t7','t8','t9','t10','t11','t12','t13','t14','t15','t16','t17','t18','t19','t20','t21','t22','t23',
    'pday','tmax','pioggia','vento','holiday','perc','ponte','daylightsaving','endsdaylightsaving',
    'r0','r1','r2','r3','r4','r5','r6','r7','r8','r9','r10','r11','r12','r13','r14','r15','r16','r17','r18','r19','r20','r21','r22','r23','y']]
    
    return dts
####################################################################################################
def CorrectionDataset(test, yhat_test, db, meteo, zona, short = False):

    start = min(test.index.date) 
    strm = str(start.month) if len(str(start.month)) > 1 else "0" + str(start.month)
    strd = str(start.day) if len(str(start.day)) > 1 else "0" + str(start.day)
    start_date = str(start.year) + '-' + strm + '-' + strd
    
    start2 = min(test.index.date) + datetime.timedelta(days = 7)
    strm = str(start2.month) if len(str(start2.month)) > 1 else "0" + str(start2.month)
    strd = str(start2.day) if len(str(start2.day)) > 1 else "0" + str(start2.day)
    start2_date = str(start2.year) + '-' + strm + '-' + strd    
    
    final = max(test.index.date)
    strm = str(final.month) if len(str(final.month)) > 1 else "0" + str(final.month)
    strd = str(final.day) if len(str(final.day)) > 1 else "0" + str(final.day)
    final_date = str(final.year) + '-' + strm + '-' + strd

    final2 = max(test.index.date) + datetime.timedelta(days = 1)
    strm = str(final2.month) if len(str(final2.month)) > 1 else "0" + str(final2.month)
    strd = str(final2.day) if len(str(final2.day)) > 1 else "0" + str(final2.day)
    final2_date = str(final2.year) + '-' + strm + '-' + strd


    sad = Get_SampleAsTS_AtDay(db, zona, start_date, final2_date)
    pc2 = percentageConsumption2(db, zona, start_date, final_date)
    yht = pd.DataFrame({'yhat': yhat_test}).set_index(test.index)
    est_sample = []
    for i in pc2.index:
        yhtd = yht.ix[yht.index.date == pd.to_datetime(i).date()].values.ravel()
        res = (yhtd * pc2.ix[i].values).tolist()
        est_sample.extend(res)
    
    ES = pd.DataFrame({"sam_hat": est_sample})
    ES = ES.set_index(test.index)
    
    TE = pd.DataFrame({'y': test['y'].values.ravel().tolist()}).set_index(test.index)
    TE = TE['y'].sort_index()
    
    DFE = pd.DataFrame({"error": sad.values.ravel()[:ES.values.ravel().size] - ES.values.ravel(), "yy": TE})
    
    if short:
        TE2 = TE.ix[TE.index.date >= start2]
        DFE = pd.DataFrame({"error": (sad.values.ravel()[:ES.values.ravel().size] - ES.values.ravel())[:TE2.shape[0]], "yy": TE2})
        return DFE
    else:    
        dts = OrderedDict()
        for i in pd.date_range(start2_date, final_date, freq = "H"):
            bri = Bridge(i.date())
            dls = StartsDaylightSaving(i.date())
            edls = EndsDaylightSaving(i.date())
            ll = []        
            hvector = np.repeat(0, 24)
            dvector = np.repeat(0, 7)
            mvector = np.repeat(0,12)
            wd = i.weekday()        
            td = 7
#            if wd == 0:
#                td = 4
            #cmym = DFE["error"].ix[DFE.index.date == (i.date()- datetime.timedelta(days = td))].values.ravel()
            cmym = DFE["error"].ix[DFE.index == (i - datetime.timedelta(days = td))].values.ravel()
            dvector[wd] = 1
            h = i.hour
            hvector[h] = 1
            mvector[(i.month-1)] = 1
            dy = i.timetuple().tm_yday
            Tmax = meteo['Tmedia'].ix[meteo['DATA'] == i.date()].values.ravel()[0] - meteo['Tmedia'].ix[meteo['DATA'] == (i - datetime.timedelta(days = td)).date()].values.ravel()[0]
            rain = meteo['PIOGGIA'].ix[meteo['DATA'] == i.date()].values.ravel()[0]
            wind = meteo['VENTOMEDIA'].ix[meteo['DATA'] == i.date()].values.ravel()[0]
            hol = AddHolidaysDate(i.date())
            ll.extend(dvector.tolist())
            ll.extend(mvector.tolist())
            ll.extend(hvector.tolist())        
            ll.extend([dy, Tmax, rain, wind, hol, bri, dls, edls])
            ll.extend(cmym.tolist())
            if DFE.ix[i].shape[0] > 1:
                y = DFE['yy'].ix[i].sum()
            elif DFE.ix[i].shape[0] == 0:
                y = 0
            else:
                y = DFE['yy'].ix[i]
            ll.extend([y])
            dts[i] =  ll
        dts = pd.DataFrame.from_dict(dts, orient = 'index')
#        dts.columns = [['Lun', 'Mar', 'Mer', 'Gio', 'Ven', 'Sab', 'Dom','Jan','Feb','March','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec',
#        't0','t1','t2','t3','t4','t5','t6','t7','t8','t9','t10','t11','t12','t13','t14','t15','t16','t17','t18','t19','t20','t21','t22','t23',
#        'pday','tmax','pioggia','vento','holiday','ponte','daylightsaving','endsdaylightsaving',
#        'r0','r1','r2','r3','r4','r5','r6','r7','r8','r9','r10','r11','r12','r13','r14','r15','r16','r17','r18','r19','r20','r21','r22','r23','y']]
        dts.columns = [['Lun', 'Mar', 'Mer', 'Gio', 'Ven', 'Sab', 'Dom','Jan','Feb','March','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec',
        't0','t1','t2','t3','t4','t5','t6','t7','t8','t9','t10','t11','t12','t13','t14','t15','t16','t17','t18','t19','t20','t21','t22','t23',
        'pday','tmax','pioggia','vento','holiday','ponte','daylightsaving','endsdaylightsaving','r','y']]                
        return dts
####################################################################################################
def CorrectionDatasetForecast(test, yhat_test, db, meteo, zona, short = False):

    start = min(test.index.date) 
    strm = str(start.month) if len(str(start.month)) > 1 else "0" + str(start.month)
    strd = str(start.day) if len(str(start.day)) > 1 else "0" + str(start.day)
    start_date = str(start.year) + '-' + strm + '-' + strd
    
    start2 = min(test.index.date) + datetime.timedelta(days = 3)
    strm = str(start2.month) if len(str(start2.month)) > 1 else "0" + str(start2.month)
    strd = str(start2.day) if len(str(start2.day)) > 1 else "0" + str(start2.day)
    start2_date = str(start2.year) + '-' + strm + '-' + strd    
    
    final = max(test.index.date)
    strm = str(final.month) if len(str(final.month)) > 1 else "0" + str(final.month)
    strd = str(final.day) if len(str(final.day)) > 1 else "0" + str(final.day)
    final_date = str(final.year) + '-' + strm + '-' + strd

    final2 = max(test.index.date) + datetime.timedelta(days = 1)
    strm = str(final2.month) if len(str(final2.month)) > 1 else "0" + str(final2.month)
    strd = str(final2.day) if len(str(final2.day)) > 1 else "0" + str(final2.day)
    final2_date = str(final2.year) + '-' + strm + '-' + strd


    sad = Get_SampleAsTS_AtDay(db, zona, start_date, final2_date)
    pc2 = percentageConsumption2(db, zona, start_date, final_date)
    yht = pd.DataFrame({'yhat': yhat_test}).set_index(test.index)
    est_sample = []
    for i in pc2.index:
        yhtd = yht.ix[yht.index.date == pd.to_datetime(i).date()].values.ravel()
        res = (yhtd * pc2.ix[i].values).tolist()
        est_sample.extend(res)
    
    ES = pd.DataFrame({"sam_hat": est_sample})
    ES = ES.set_index(test.index)
    
    sad = sad.ix[sad.index.date <= ES.index.date[-1]]
    DFE = pd.DataFrame({"error": sad.values.ravel() - ES.values.ravel()})
    DFE = DFE.set_index(ES.index)
    if short:
        TE2 = TE.ix[TE.index.date >= start2]
        DFE = pd.DataFrame({"error": (sad.values.ravel()[:ES.values.ravel().size] - ES.values.ravel())[:TE2.shape[0]]})
        return DFE
    else:    
        dts = OrderedDict()
        for i in pd.date_range(start2_date, final_date, freq = "H"):
            bri = Bridge(i.date())
            dls = StartsDaylightSaving(i.date())
            edls = EndsDaylightSaving(i.date())
            ll = []        
            hvector = np.repeat(0, 24)
            dvector = np.repeat(0, 7)
            mvector = np.repeat(0,12)
            wd = i.weekday()        
            td = 3
            if wd == 0:
                td = 4
            #cmym = DFE["error"].ix[DFE.index.date == (i.date()- datetime.timedelta(days = td))].values.ravel()
            cmym = DFE["error"].ix[DFE.index == (i - datetime.timedelta(days = td))].values.ravel()
            dvector[wd] = 1
            h = i.hour
            hvector[h] = 1
            mvector[(i.month-1)] = 1
            dy = i.timetuple().tm_yday
            Tmax = meteo['Tmax'].ix[meteo.index.date == i.date()].values.ravel()[0]
            rain = meteo['pioggia'].ix[meteo.index.date == i.date()].values.ravel()[0]
            wind = meteo['vento'].ix[meteo.index.date == i.date()].values.ravel()[0]
            hol = AddHolidaysDate(i.date())
            ll.extend(dvector.tolist())
            ll.extend(mvector.tolist())
            ll.extend(hvector.tolist())        
            ll.extend([dy, Tmax, rain, wind, hol, bri, dls, edls])
            ll.extend(cmym.tolist())
            dts[i] =  ll
        dts = pd.DataFrame.from_dict(dts, orient = 'index')
#        dts.columns = [['Lun', 'Mar', 'Mer', 'Gio', 'Ven', 'Sab', 'Dom','Jan','Feb','March','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec',
#        't0','t1','t2','t3','t4','t5','t6','t7','t8','t9','t10','t11','t12','t13','t14','t15','t16','t17','t18','t19','t20','t21','t22','t23',
#        'pday','tmax','pioggia','vento','holiday','ponte','daylightsaving','endsdaylightsaving',
#        'r0','r1','r2','r3','r4','r5','r6','r7','r8','r9','r10','r11','r12','r13','r14','r15','r16','r17','r18','r19','r20','r21','r22','r23','y']]
        dts.columns = [['Lun', 'Mar', 'Mer', 'Gio', 'Ven', 'Sab', 'Dom','Jan','Feb','March','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec',
        't0','t1','t2','t3','t4','t5','t6','t7','t8','t9','t10','t11','t12','t13','t14','t15','t16','t17','t18','t19','t20','t21','t22','t23',
        'pday','tmax','pioggia','vento','holiday','ponte','daylightsaving','endsdaylightsaving','r']]                
        return dts
####################################################################################################
def DifferentialDataset(df, db, meteo, zona):
    DFF = OrderedDict()
    db = db.ix[db["Area"] == zona]
    dr = pd.date_range('2016-01-08', '2017-05-31', freq = 'H')
    counter = 0
    for i in dr:
        counter += 1
        if counter % 100 == 0:
            print 'avanzamento = {}'.format(counter/dr.size)
        ll = []
        bri = Bridge(i.date())
        dls = StartsDaylightSaving(i.date())
        edls = EndsDaylightSaving(i.date())
        bri7 = Bridge(i.date() - datetime.timedelta(days = 7))
        dls7 = StartsDaylightSaving(i.date() - datetime.timedelta(days = 7))
        edls7 = EndsDaylightSaving(i.date() - datetime.timedelta(days = 7))
        hvector = np.repeat(0, 24)
        h = i.hour
        hvector[h] = 1
        ll.extend([bri, dls, edls, bri7, dls7, edls7])
        ll.extend([AddHolidaysDate(i.date()), AddHolidaysDate(i.date() - datetime.timedelta(days = 7))])
        sam_d = db[db.columns[3:]].ix[db["Giorno"] == i.date()].sum(axis = 0).values.ravel()/1000
        sam_w = db[db.columns[3:]].ix[db["Giorno"] == (i.date() - datetime.timedelta(days = 7))].sum(axis = 0).values.ravel()/1000
        ll.extend(hvector.tolist())
        ll.append(sam_d[h] - sam_w[h])
#        ll.append(meteo['Tmax'].ix[meteo.index.date == i.date()].values[0] - meteo['Tmax'].ix[meteo.index.date == (i.date() - datetime.timedelta(days = 365))].values[0])
#        ll.append(df['y'].ix[DBB.index.date == i.date()].mean() - DBB['y'].ix[DBB.index.date == (i.date() - datetime.timedelta(days = 365))].mean())
        ll.append(meteo['Tmax'].ix[meteo.index.date == i.date()].mean() - meteo['Tmax'].ix[meteo.index.date == (i.date() - datetime.timedelta(days = 7))].mean())

        if df['MO [MWh]'].ix[df.index == i].shape[0] == 0:
            y1 = 0
        elif df['MO [MWh]'].ix[df.index == i].shape[0] > 1:
            y1 = df['MO [MWh]'].ix[df.index == i].sum()
        else:
            y1 = df['MO [MWh]'].ix[df.index == i].values[0]
        
        if df['MO [MWh]'].ix[df.index == (i - datetime.timedelta(days = 7))].shape[0] == 0:
            y7 = 0
        elif df['MO [MWh]'].ix[df.index == (i - datetime.timedelta(days = 7))].shape[0] > 1:
            y7 = df['MO [MWh]'].ix[df.index == (i - datetime.timedelta(days = 7))].sum()
        else:
            y7 = df['MO [MWh]'].ix[df.index == (i - datetime.timedelta(days = 7))].values[0]
            
        Y = y1 - y7
        ll.append(Y)
        DFF[i] = ll
    
    DFF = pd.DataFrame.from_dict(DFF, orient = 'index')
    DFF.columns = [['ponte','daylightsaving','endsdaylightsaving','ponte7','daylightsaving7','endsdaylightsaving7',
                    'hol', 'hol7','t0','t1','t2','t3','t4','t5','t6','t7','t8','t9','t10','t11','t12','t13','t14',
                    't15','t16','t17','t18','t19','t20','t21','t22','t23','diff_sample','diff_tmax','diff_zona']] 
    
    month = ["Jan", "Feb", "March","Apr", "May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    for m in month:
        im = month.index(m)
        M = DFF.index.month == (im+1)        
        atm = [int(x) for x in M]
        DFF[m] = atm
    
    DFF = DFF[["Jan", "Feb", "March","Apr", "May","Jun","Jul","Aug","Sep","Oct","Nov","Dec",
               'ponte','daylightsaving','endsdaylightsaving','ponte7','daylightsaving7','endsdaylightsaving7',
               'hol', 'hol7','t0','t1','t2','t3','t4','t5','t6','t7','t8','t9','t10','t11','t12','t13','t14',
               't15','t16','t17','t18','t19','t20','t21','t22','t23','diff_tmax','diff_sample','diff_zona']]
    
    return DFF   
####################################################################################################
def DifferentialDatasetForecast(db, meteo, zona, di, df):
    DFF = OrderedDict()
    db = db.ix[db["Area"] == zona]
    dr = pd.date_range(di, df, freq = 'H')
    counter = 0
    for i in dr:
        counter += 1
        if counter % 100 == 0:
            print 'avanzamento = {}'.format(counter/dr.size)
        ll = []
        bri = Bridge(i.date())
        dls = StartsDaylightSaving(i.date())
        edls = EndsDaylightSaving(i.date())
        bri7 = Bridge(i.date() - datetime.timedelta(days = 7))
        dls7 = StartsDaylightSaving(i.date() - datetime.timedelta(days = 7))
        edls7 = EndsDaylightSaving(i.date() - datetime.timedelta(days = 7))
        hvector = np.repeat(0, 24)
        h = i.hour
        hvector[h] = 1
        ll.extend([bri, dls, edls, bri7, dls7, edls7])
        ll.extend([AddHolidaysDate(i.date()), AddHolidaysDate(i.date() - datetime.timedelta(days = 7))])
        sam_d = db[db.columns[3:]].ix[db["Giorno"] == i.date()].sum(axis = 0).values.ravel()/1000
        sam_w = db[db.columns[3:]].ix[db["Giorno"] == (i.date() - datetime.timedelta(days = 7))].sum(axis = 0).values.ravel()/1000
        ll.extend(hvector.tolist())
        ll.append(sam_d[h] - sam_w[h])
#        ll.append(meteo['Tmax'].ix[meteo.index.date == i.date()].values[0] - meteo['Tmax'].ix[meteo.index.date == (i.date() - datetime.timedelta(days = 365))].values[0])
#        ll.append(df['y'].ix[DBB.index.date == i.date()].mean() - DBB['y'].ix[DBB.index.date == (i.date() - datetime.timedelta(days = 365))].mean())
        ll.append(meteo['Tmax'].ix[meteo.index.date == i.date()].mean() - meteo['Tmax'].ix[meteo.index.date == (i.date() - datetime.timedelta(days = 7))].mean())


        DFF[i] = ll
    
    DFF = pd.DataFrame.from_dict(DFF, orient = 'index')
    DFF.columns = [['ponte','daylightsaving','endsdaylightsaving','ponte7','daylightsaving7','endsdaylightsaving7',
                    'hol', 'hol7','t0','t1','t2','t3','t4','t5','t6','t7','t8','t9','t10','t11','t12','t13','t14',
                    't15','t16','t17','t18','t19','t20','t21','t22','t23','diff_sample','diff_tmax']] 
    
    month = ["Jan", "Feb", "March","Apr", "May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    for m in month:
        im = month.index(m)
        M = DFF.index.month == (im+1)        
        atm = [int(x) for x in M]
        DFF[m] = atm
    
    DFF = DFF[["Jan", "Feb", "March","Apr", "May","Jun","Jul","Aug","Sep","Oct","Nov","Dec",
               'ponte','daylightsaving','endsdaylightsaving','ponte7','daylightsaving7','endsdaylightsaving7',
               'hol', 'hol7','t0','t1','t2','t3','t4','t5','t6','t7','t8','t9','t10','t11','t12','t13','t14',
               't15','t16','t17','t18','t19','t20','t21','t22','t23','diff_tmax','diff_sample']]
    
    return DFF   
####################################################################################################
def MakeExtendedDatasetGivenPOD(db, meteo, pod, end):
#### @PARAM: df is the dataset from Terna, db, All zona those for computing the perc consumption
#### and the sample curve
#### @BRIEF: extended version of the quasi-omonimous function in Sbilanciamento.py
#### every day will have a dummy variable representing it
    #wdays = ['Lun', 'Mar', 'Mer', 'Gio', 'Ven', 'Sab', 'Dom']
    db = db.ix[db["POD"] == pod].reset_index(drop = True)
    dts = OrderedDict()
    start = str(db["Giorno"].ix[0])[:10]
#    end =  str(db["Giorno"].ix[db.shape[0]-1])[:10]
#    if db["Giorno"].ix[db.shape[0]-1] > datetime.datetime(2017,3,31):
#        end = '2017-04-02'
    dr = pd.date_range(start, end, freq = 'H')
    for i in dr:
        #print i
        bri = Bridge(i.date())
        dls = StartsDaylightSaving(i.date())
        edls = EndsDaylightSaving(i.date())
        ll = []        
        hvector = np.repeat(0, 24)
        dvector = np.repeat(0, 7)
        mvector = np.repeat(0,12)
        wd = i.weekday()        
        td = 2
        if wd == 0:
            td = 3
        cmym = db[db.columns[3:]].ix[db["Giorno"] == (i.date()- datetime.timedelta(days = td))].sum(axis = 0).values.ravel()/1000
        dvector[wd] = 1
        h = i.hour
        hvector[h] = 1
        mvector[(i.month-1)] = 1
        #dy = i.timetuple().tm_yday
        Tmax = meteo['Tmax'].ix[meteo['DATA'] == i.date()].values.ravel()[0]
        rain = meteo['PIOGGIA'].ix[meteo['DATA'] == i.date()].values.ravel()[0]
        wind = meteo['VENTOMEDIA'].ix[meteo['DATA'] == i.date()].values.ravel()[0]
        hol = AddHolidaysDate(i.date())
        ll.extend(dvector.tolist())
        ll.extend(mvector.tolist())
        ll.extend(hvector.tolist())        
        ll.extend([Tmax, rain, wind, hol, bri, dls, edls])
        ll.extend(cmym.tolist())
        if db[str(h + 1)].ix[db["Giorno"] == i.date()].shape[0] > 0:
            ll.extend([db[str(h + 1)].ix[db["Giorno"] == i.date()].values[0]])
            dts[i] =  ll
        else:
            ll.extend([0])
    dts = pd.DataFrame.from_dict(dts, orient = 'index')
    dts.columns = [['Lun','Mar','Mer','Gio','Ven','Sab','Dom','Jan','Feb','March','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec',
    't0','t1','t2','t3','t4','t5','t6','t7','t8','t9','t10','t11','t12','t13','t14','t15','t16','t17','t18','t19','t20','t21','t22','t23',
    'tmax','pioggia','vento','holiday','ponte','daylightsaving','endsdaylightsaving',
    'r0','r1','r2','r3','r4','r5','r6','r7','r8','r9','r10','r11','r12','r13','r14','r15','r16','r17','r18','r19','r20','r21','r22','r23','y']]
    return dts
####################################################################################################
def MakeForecastDataset(db, meteo, zona, time_delta = 1):
#### @PARAM: df is the dataset from Terna, db, All zona those for computing the perc consumption
#### and the sample curve
#### @BRIEF: extended version of the quasi-omonimous function in Sbilanciamento.py
#### every day will have a dummy variable representing it
    #wdays = ['Lun', 'Mar', 'Mer', 'Gio', 'Ven', 'Sab', 'Dom']
    future = datetime.datetime.now() + datetime.timedelta(days = time_delta + 1)
    strm = str(future.month) if len(str(future.month)) > 1 else "0" + str(future.month)
    strd = str(future.day) if len(str(future.day)) > 1 else "0" + str(future.day)
    final_date = str(future.year) + '-' + strm + '-' + strd
    psample = percentageConsumption2(db, zona, '2017-07-07',final_date)
    psample = psample.set_index(pd.date_range('2017-07-07', final_date, freq = 'D')[:psample.shape[0]])
    dts = OrderedDict()
    dr = pd.date_range('2017-07-07', final_date, freq = 'H')
    for i in dr[2*24:dr.size-1]:
        bri = Bridge(i.date())
        dls = StartsDaylightSaving(i.date())
        edls = EndsDaylightSaving(i.date())
        ll = []        
        hvector = np.repeat(0, 24)
        dvector = np.repeat(0, 7)
        mvector = np.repeat(0,12)
        wd = i.weekday()        
        td = 2
        if wd == 0:
            td = 3
        cmym = db[db.columns[3:]].ix[db["Giorno"] == (i.date()- datetime.timedelta(days = td))].sum(axis = 0).values.ravel()/1000
        dvector[wd] = 1
        h = i.hour
        hvector[h] = 1
        mvector[(i.month-1)] = 1
        dy = i.timetuple().tm_yday
        Tmax = meteo['Tmax'].ix[meteo.index.date == i.date()].values.ravel()[0]
        rain = meteo['pioggia'].ix[meteo.index.date == i.date()].values.ravel()[0]
        wind = meteo['vento'].ix[meteo.index.date == i.date()].values.ravel()[0]
        hol = AddHolidaysDate(i.date())
        ps = psample.ix[psample.index.date == (i.date() - datetime.timedelta(days = td))]
        ll.extend(dvector.tolist())
        ll.extend(mvector.tolist())
        ll.extend(hvector.tolist())        
        ll.extend([dy, Tmax, rain, wind, hol, ps[0].values[0], bri, dls, edls])
        ll.extend(cmym.tolist())
        dts[i] =  ll
    dts = pd.DataFrame.from_dict(dts, orient = 'index')
    dts.columns = [['Lun', 'Mar', 'Mer', 'Gio', 'Ven', 'Sab', 'Dom','Jan','Feb','March','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec',
    't0','t1','t2','t3','t4','t5','t6','t7','t8','t9','t10','t11','t12','t13','t14','t15','t16','t17','t18','t19','t20','t21','t22','t23',
    'pday','tmax','pioggia','vento','holiday','perc','ponte','daylightsaving','endsdaylightsaving',
    'r0','r1','r2','r3','r4','r5','r6','r7','r8','r9','r10','r11','r12','r13','r14','r15','r16','r17','r18','r19','r20','r21','r22','r23']]
    return dts
####################################################################################################
def MakeForecastPerPOD(db, meteo, pod, time_delta = 1):
#### @PARAM: df is the dataset from Terna, db, All zona those for computing the perc consumption
#### and the sample curve
#### @BRIEF: extended version of the quasi-omonimous function in Sbilanciamento.py
#### every day will have a dummy variable representing it
    #wdays = ['Lun', 'Mar', 'Mer', 'Gio', 'Ven', 'Sab', 'Dom']
    db = db.ix[db['POD'] == pod]
    future = datetime.datetime.now() + datetime.timedelta(days = time_delta)
    strm = str(future.month) if len(str(future.month)) > 1 else "0" + str(future.month)
    strd = str(future.day) if len(str(future.day)) > 1 else "0" + str(future.day)
    final_date = str(future.year) + '-' + strm + '-' + strd
    dts = OrderedDict()
    dr = pd.date_range('2017-04-01', final_date, freq = 'H')
    for i in dr:
        ll = []        
        hvector = np.repeat(0, 24)
        dvector = np.repeat(0, 7)
        wd = i.weekday()        
        td = 2
        if wd == 0:
            td = 3
        cmym = db[db.columns[3:]].ix[db["Giorno"] == (i.date()- datetime.timedelta(days = td))].sum(axis = 0).values.ravel()/1000
        dvector[wd] = 1
        h = i.hour
        hvector[h] = 1
        dy = i.timetuple().tm_yday
        Tmax = meteo['Tmax'].ix[meteo.index.date == i.date()].values.ravel()[0]
        rain = meteo['pioggia'].ix[meteo.index.date == i.date()].values.ravel()[0]
        wind = meteo['vento'].ix[meteo.index.date == i.date()].values.ravel()[0]
        hol = AddHolidaysDate(i.date())
        ll.extend(dvector.tolist())
        ll.extend(hvector.tolist())        
        ll.extend([dy, Tmax, rain, wind, hol])
        ll.extend(cmym.tolist())
        dts[i] =  ll
    dts = pd.DataFrame.from_dict(dts, orient = 'index')
    dts.columns = [['Lun', 'Mar', 'Mer', 'Gio', 'Ven', 'Sab', 'Dom',
    't0','t1','t2','t3','t4','t5','t6','t7','t8','t9','t10','t11','t12','t13','t14','t15','t16','t17','t18','t19','t20','t21','t22','t23',
    'pday','tmax','pioggia','vento','holiday',
    'r0','r1','r2','r3','r4','r5','r6','r7','r8','r9','r10','r11','r12','r13','r14','r15','r16','r17','r18','r19','r20','r21','r22','r23']]
    return dts
####################################################################################################
def MakeDatasetLongTerm(df, db, meteo, zona, td):
#### @PARAM: df is the dataset from Terna, db, All zona those for computing the perc consumption
#### and the sample curve
#### @BRIEF: extended version of the quasi-omonimous function in Sbilanciamento.py
#### every day will have a dummy variable representing it
    #wdays = ['Lun', 'Mar', 'Mer', 'Gio', 'Ven', 'Sab', 'Dom']
    final = max(df.index.date)
    strm = str(final.month) if len(str(final.month)) > 1 else "0" + str(final.month)
    strd = str(final.day) if len(str(final.day)) > 1 else "0" + str(final.day)
    final_date = str(final.year) + '-' + strm + '-' + strd
    psample = percentageConsumption(db, zona, '2016-01-01', final_date)
    psample = psample.set_index(pd.date_range('2016-01-01', final_date, freq = 'D')[:psample.shape[0]])
    dts = OrderedDict()
    df = df.ix[df.index.date >= (datetime.date(2016,1,1) + datetime.timedelta(days = td))]
    for i in df.index.tolist():
        dls = StartsDaylightSaving(i.date())
        edls = EndsDaylightSaving(i.date())
        ll = []        
        hvector = np.repeat(0, 24)
        dvector = np.repeat(0, 7)
        wd = i.weekday()        
        cmym = db[db.columns[3:]].ix[db["Giorno"] == (i.date()- datetime.timedelta(days = td))].sum(axis = 0).values.ravel()/1000
        dvector[wd] = 1
        h = i.hour
        hvector[h] = 1
        dy = i.timetuple().tm_yday
        Tmax = meteo['Tmax'].ix[meteo['DATA'] == i.date()].values.ravel()[0]
        rain = meteo['PIOGGIA'].ix[meteo['DATA'] == i.date()].values.ravel()[0]
        wind = meteo['VENTOMEDIA'].ix[meteo['DATA'] == i.date()].values.ravel()[0]
        hol = AddHolidaysDate(i.date())
        ps = psample.ix[psample.index.date == (i.date() - datetime.timedelta(days = td))]
        ll.extend(dvector.tolist())
        ll.extend(hvector.tolist())        
        ll.extend([dy, Tmax, rain, wind, hol, ps[0].values[0], dls, edls])
        ll.extend(cmym.tolist())
        if df.ix[i].shape[0] > 1:
            y = df['MO [MWh]'].ix[i].sum()
        elif df.ix[i].shape[0] == 0:
            y = 0
        else:
            y = df['MO [MWh]'].ix[i]
        ll.extend([y])
        dts[i] =  ll
    dts = pd.DataFrame.from_dict(dts, orient = 'index')
    dts.columns = [['Lun', 'Mar', 'Mer', 'Gio', 'Ven', 'Sab', 'Dom',
    't0','t1','t2','t3','t4','t5','t6','t7','t8','t9','t10','t11','t12','t13','t14','t15','t16','t17','t18','t19','t20','t21','t22','t23',
    'pday','tmax','pioggia','vento','holiday','perc','daylightsaving','endsdaylightsaving',
    'r0','r1','r2','r3','r4','r5','r6','r7','r8','r9','r10','r11','r12','r13','r14','r15','r16','r17','r18','r19','r20','r21','r22','r23','y']]
    return dts
####################################################################################################
def MakeForecastLongTerm(db, meteo, zona, time_delta):
#### @PARAM: df is the dataset from Terna, db, All zona those for computing the perc consumption
#### and the sample curve
#### @BRIEF: extended version of the quasi-omonimous function in Sbilanciamento.py
#### every day will have a dummy variable representing it
    #wdays = ['Lun', 'Mar', 'Mer', 'Gio', 'Ven', 'Sab', 'Dom']
    future = datetime.datetime.now() + datetime.timedelta(days = time_delta + 1)
    strm = str(future.month) if len(str(future.month)) > 1 else "0" + str(future.month)
    strd = str(future.day) if len(str(future.day)) > 1 else "0" + str(future.day)
    final_date = str(future.year) + '-' + strm + '-' + strd
    psample = percentageConsumption(db, zona, '2017-05-20',final_date)
    psample = psample.set_index(pd.date_range('2017-05-20', final_date, freq = 'D')[:psample.shape[0]])
    dts = OrderedDict()
    dr = pd.date_range('2017-05-20', final_date, freq = 'H')
    for i in dr[time_delta*24:dr.size-1]:
        dls = StartsDaylightSaving(i.date())
        edls = EndsDaylightSaving(i.date())
        ll = []        
        hvector = np.repeat(0, 24)
        dvector = np.repeat(0, 7)
        wd = i.weekday()        
        td = time_delta
        cmym = db[db.columns[3:]].ix[db["Giorno"] == (i.date() - datetime.timedelta(days = td))].sum(axis = 0).values.ravel()/1000
        dvector[wd] = 1
        h = i.hour
        hvector[h] = 1
        dy = i.timetuple().tm_yday
        Tmax = meteo['Tmax'].ix[meteo.index.date == i.date()].values.ravel()[0]
        rain = meteo['pioggia'].ix[meteo.index.date == i.date()].values.ravel()[0]
        wind = meteo['vento'].ix[meteo.index.date == i.date()].values.ravel()[0]
        hol = AddHolidaysDate(i.date())
        ps = psample.ix[psample.index.date == (i.date() - datetime.timedelta(days = td))]
        ll.extend(dvector.tolist())
        ll.extend(hvector.tolist())        
        ll.extend([dy, Tmax, rain, wind, hol, ps[0].values[0], dls, edls])
        ll.extend(cmym.tolist())
        dts[i] =  ll
    dts = pd.DataFrame.from_dict(dts, orient = 'index')
    dts.columns = [['Lun', 'Mar', 'Mer', 'Gio', 'Ven', 'Sab', 'Dom',
    't0','t1','t2','t3','t4','t5','t6','t7','t8','t9','t10','t11','t12','t13','t14','t15','t16','t17','t18','t19','t20','t21','t22','t23',
    'pday','tmax','pioggia','vento','holiday','perc','daylightsaving','endsdaylightsaving',
    'r0','r1','r2','r3','r4','r5','r6','r7','r8','r9','r10','r11','r12','r13','r14','r15','r16','r17','r18','r19','r20','r21','r22','r23']]
    return dts
####################################################################################################
def getImbalance(error, prediction):
    ### error becomes a DataFrame with same index set as test
    tau = error.values.ravel()/prediction
    ERR = pd.DataFrame.from_dict({"Sbil": error.values.ravel().tolist(), "Sbil_perc": tau.tolist()})
    ERR = ERR.set_index(error.index)
    return ERR
####################################################################################################    
def EvaluateImbalance(val, imb, zona, pun):
    diz = OrderedDict()
    cd = convertDates(val['DATA RIFERIMENTO CORRISPETTIVO'])
    val = val.set_index(cd.values)
    valzona = val.ix[val["CODICE RUC"] == "UC_DP1608_" + zona]
    #valzona = valzona.set_index(pd.date_range("2015-01-01", "2018-01-02", freq = "H")[:valzona.shape[0]])
    for i in imb.index:
        if i.date() == datetime.date(2017,3,26) and i.hour == 2:
            next
        else:
            valsbil = 0
            valdiff = 0
            if i in pun.index:
                i = i.to_pydatetime()
                dp = pun[zona].ix[i]
                ### dentro la banda:
                if np.abs(imb["Sbil_perc"].ix[i]) <= 0.15: 
                    if valzona["SEGNO SBILANCIAMENTO AGGREGATO ZONALE"].ix[i] > 0 and imb["Sbil"].ix[i] > 0:
                        valsbil = np.abs(imb["Sbil"].ix[i]) * min(valzona["Pmgp_ven"].ix[i],valzona["PmedioMSD_acq"].ix[i])
                        valdiff = -np.abs(imb["Sbil"].ix[i]) * (dp - min(valzona["Pmgp_ven"].ix[i],valzona["PmedioMSD_acq"].ix[i]))
                    elif valzona["SEGNO SBILANCIAMENTO AGGREGATO ZONALE"].ix[i] < 0 and imb["Sbil"].ix[i] > 0:
                        valsbil = np.abs(imb["Sbil"].ix[i]) * max(valzona["Pmgp_ven"].ix[i],valzona["PmedioMSD_ven"].ix[i])
                        valdiff = -np.abs(imb["Sbil"].ix[i]) * (dp - max(valzona["Pmgp_ven"].ix[i],valzona["PmedioMSD_ven"].ix[i]))
                    elif valzona["SEGNO SBILANCIAMENTO AGGREGATO ZONALE"].ix[i] > 0 and imb["Sbil"].ix[i] < 0:
                        valsbil = -np.abs(imb["Sbil"].ix[i]) * min(valzona["Pmgp_acq"].ix[i],valzona["PmedioMSD_acq"].ix[i])
                        valdiff = np.abs(imb["Sbil"].ix[i]) * (dp - min(valzona["Pmgp_acq"].ix[i],valzona["PmedioMSD_acq"].ix[i]))
                    else:
                        valsbil = -np.abs(imb["Sbil"].ix[i]) * max(valzona["Pmgp_acq"].ix[i],valzona["PmedioMSD_ven"].ix[i])
                        valdiff = np.abs(imb["Sbil"].ix[i]) * (dp - max(valzona["Pmgp_acq"].ix[i],valzona["PmedioMSD_ven"].ix[i]))
                    diz[i] = [valsbil, valdiff]
                ### fuori banda:
                else:
                    PV = abs(-imb["Sbil"].ix[i]/(imb["Sbil_perc"].ix[i] - 1))
                    inside = PV * 0.15
                    outside = imb["Sbil"].ix[i] - inside
                    if valzona["SEGNO SBILANCIAMENTO AGGREGATO ZONALE"].ix[i] > 0 and imb["Sbil"].ix[i] > 0:
                        valsbil = np.abs(outside) * min(valzona["Pmgp_ven"].ix[i],valzona["PmedioMSD_acq"].ix[i]) +  np.abs(inside) * min(valzona["Pmgp_ven"].ix[i],valzona["PmedioMSD_acq"].ix[i])
                        valdiff = -np.abs(outside) * (dp - min(valzona["Pmgp_ven"].ix[i],valzona["PmedioMSD_acq"].ix[i])) + (-np.abs(inside) * (dp - min(valzona["Pmgp_ven"].ix[i],valzona["PmedioMSD_acq"].ix[i])))
                    elif valzona["SEGNO SBILANCIAMENTO AGGREGATO ZONALE"].ix[i] < 0 and imb["Sbil"].ix[i] > 0:
                        valsbil = np.abs(outside) * valzona["Pmgp_ven"].ix[i] + np.abs(inside) * max(valzona["Pmgp_ven"].ix[i],valzona["PmedioMSD_ven"].ix[i])
                        valdiff = -np.abs(outside) * (dp - valzona["Pmgp_ven"].ix[i]) + ( -np.abs(inside) * (dp - max(valzona["Pmgp_ven"].ix[i],valzona["PmedioMSD_ven"].ix[i])))
                    elif valzona["SEGNO SBILANCIAMENTO AGGREGATO ZONALE"].ix[i] > 0 and imb["Sbil"].ix[i] < 0:
                        valsbil = -np.abs(outside) * valzona["Pmgp_acq"].ix[i] + (-np.abs(inside) * min(valzona["Pmgp_acq"].ix[i],valzona["PmedioMSD_acq"].ix[i])) 
                        valdiff = np.abs(outside) * (dp - valzona["Pmgp_acq"].ix[i]) + np.abs(inside) * (dp - min(valzona["Pmgp_acq"].ix[i],valzona["PmedioMSD_acq"].ix[i]))
                    else:
                        valsbil = -np.abs(outside) * max(valzona["Pmgp_acq"].ix[i],valzona["PmedioMSD_ven"].ix[i]) + (-np.abs(inside) * max(valzona["Pmgp_acq"].ix[i],valzona["PmedioMSD_ven"].ix[i]))
                        valdiff = np.abs(outside) * (dp - max(valzona["Pmgp_acq"].ix[i],valzona["PmedioMSD_ven"].ix[i])) + np.abs(inside) * (dp - max(valzona["Pmgp_acq"].ix[i],valzona["PmedioMSD_ven"].ix[i]))
                    diz[i] = [valsbil, valdiff]
    diz = pd.DataFrame.from_dict(diz, orient = 'index')
    diz.columns = [["Val_sbil", "gain"]]
    return diz
####################################################################################################
def podwiseForecast(db, meteo, zona, end):
    removed = []
    fdf = OrderedDict()
    podlist = list(set(db["POD"].ix[db["Area"] == zona].values.ravel().tolist()))
    for pod in podlist:
        if db.ix[db["POD"] == pod].shape[0] > 366:
            print 'Avanzamento: {} %'.format((podlist.index(pod) + 1)/len(podlist))
            DBP = MakeExtendedDatasetGivenPOD(DB, meteo, pod, end)
            
            train2 = DBP.ix[DBP.index.date < datetime.date(2017, 1, 1)]
            train = DBP.sample(frac = 1)
            test = DBP.ix[DBP.index.date > datetime.date(2016, 12, 31)]
            #test = test.ix[test.index.date < datetime.date(2017, 4, 12)]
            if test['y'].sum() < 10e-6:
                removed.append(pod)
                continue
            
            brf = RandomForestRegressor(criterion = 'mse', max_depth = 48, n_estimators = 24, n_jobs = 1)
            
            brf.fit(train[train.columns[:60]], train[train.columns[60]])
            yhat_train = brf.predict(train2[train2.columns[:60]])
            
            rfR2 = 1 - (np.sum((train2[train2.columns[60]] - yhat_train)**2))/(np.sum((train2[train2.columns[60]] - np.mean(train2[train2.columns[60]]))**2))
            print 'R-squared per pod {} = {}'.format(pod, rfR2)
            
            
            yhat_test = brf.predict(test[test.columns[:60]])
            rfR2_test = 1 - (np.sum((test[test.columns[60]] - yhat_test)**2))/(np.sum((test[test.columns[60]] - np.mean(test[test.columns[60]]))**2))
            print 'R-squared test per pod {} = {}'.format(pod, rfR2_test)
            
            if yhat_test.size < 2185:
                cc = np.concatenate([yhat_test, np.repeat(0, 2185 - yhat_test.size)]).tolist()
            else:
                cc = yhat_test.tolist()
            
            fdf[pod] = cc
        else:
            removed.append(pod)
    
    fdf = pd.DataFrame.from_dict(fdf, orient = "columns")
    fdf = fdf.set_index(pd.date_range('2017-01-01', '2017-04-02', freq = 'H'))
    return fdf, removed
#################################################################################################### 
def transform_df(df):
    diz = OrderedDict()
    dr = pd.date_range('2016-01-01', '2017-04-01', freq = 'D')
    for i in dr:
        dfi = df['MO [MWh]'].ix[df.index.date == i.date()]
        if int(dfi.shape[0]) == 24:
            diz[i.date()] = dfi.values.ravel().tolist()
        elif int(dfi.shape[0]) == 23:
            vec = np.repeat(0.0, 24)
            for h in range(24):
                if h != 2:
                    vec[h] = dfi.ix[dfi.index.hour == h]
            diz[i.date()] = vec.tolist()
        else:
            vec = np.repeat(0.0, 24)
            for h in range(24):
                if h == 2:
                    vec[h] = dfi.ix[dfi.index.hour == h].sum()
                else:
                    vec[h] = dfi.ix[dfi.index.hour == h].sum()
            diz[i.date()] = vec.tolist()
    diz = pd.DataFrame.from_dict(diz, orient = 'index')
    diz.columns = [['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24']]
    return diz
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
def removePerdite(df):
    df2 = OrderedDict()
    for i in range(df.shape[0]):
        x = df.ix[i]
        df2[i] = [x[1],x[3],x[4],x[10]/x[36],x[11]/x[36],x[12]/x[36],x[13]/x[36],x[14]/x[36],
                  x[15]/x[36],x[16]/x[36],x[17]/x[36],x[18]/x[36],x[19]/x[36],x[20]/x[36],
                  x[21]/x[36],x[22]/x[36],x[23]/x[36],x[24]/x[36],x[25]/x[36],x[26]/x[36],x[27]/x[36],x[28]/x[36],
                  x[29]/x[36],x[30]/x[36],x[31]/x[36],x[32]/x[36],x[33]/x[36],x[34]/x[36]]
    df2 = pd.DataFrame.from_dict(df2, orient = 'index')
    df2.columns = [["POD", "Area", "Giorno",  "1","2","3","4","5","6","7","8","9","10","11","12","13","14","15",
         "16","17","18","19","20","21","22","23","24", "25"]]
    return df2
####################################################################################################
def reduceDB(db):
    ndb = OrderedDict()
    db24 = db[db.columns[:27]].ix[db["25"] == 0]
    db25 = db.ix[db["25"] > 0].reset_index(drop = True)
    for r in range(db25.shape[0]):
        nl = db25[db25.columns[:5]].ix[r].values.tolist()
        nl.append(db25[db25.columns[5:7]].ix[r].sum())
        nl.extend(db25[db25.columns[7:]].ix[r].values.tolist())
        ndb[r] = nl
    ndb = pd.DataFrame.from_dict(ndb, orient = 'index')
    ndb.columns = [["POD", "Area", "Giorno", "1","2","3","4","5","6","7","8","9","10","11","12","13","14","15",
         "16","17","18","19","20","21","22","23","24"]]
    NDB = db24.append(ndb, ignore_index = True)
    return NDB
####################################################################################################
def SampleAtDay(db, dtd, zona):
    db = db.ix[db["Area"] == zona]
    return list(set(db["POD"].ix[db["Giorno"] == dtd].tolist()))
####################################################################################################
def MakeSplittedDataset(df, db, meteo, zona):
    ### @PARAM: df --> misurato di Terna, db --> database misure orarie giornaliere    
    final = max(df.index.date)
    strm = str(final.month) if len(str(final.month)) > 1 else "0" + str(final.month)
    strd = str(final.day) if len(str(final.day)) > 1 else "0" + str(final.day)
    final_date = str(final.year) + '-' + strm + '-' + strd
    psample = percentageConsumption2(db, zona, '2016-01-01', final_date)
    psample = psample.set_index(pd.date_range('2016-01-01', final_date, freq = 'D')[:psample.shape[0]])
    df = df.ix[df.index.date >= datetime.date(2016,1,3)]
    OOS = Get_OutOfSample(df, db, zona)
    df_oos = MakeExtendedDatasetWithSampleCurve(OOS, db, meteo, zona)
    if os.path.exists("C:/Users/utente/Documents/Sbilanciamento/" + zona + "/Out_Of_Sample"):
        df_oos.to_hdf("C:/Users/utente/Documents/Sbilanciamento/" + zona + "/Out_Of_Sample/dataset")
    else:
        os.makedirs("C:/Users/utente/Documents/Sbilanciamento/" + zona + "/Out_Of_Sample")
    
#    crpp2017 = pd.read_hdf("C:/Users/utente/Documents/Sbilanciamento/CRPP_2017.h5")
    
    removed = []
    fdf = OrderedDict()
    podlist = list(set(db["POD"].ix[db["Area"] == zona].values.ravel().tolist()))
    for pod in podlist:
        if db.ix[db["POD"] == pod].shape[0] > 366:
            print 'Avanzamento: {} %'.format((podlist.index(pod) + 1)/len(podlist))
            DBP = MakeExtendedDatasetGivenPOD(DB, meteo, pod)
            
            train2 = DBP.ix[DBP.index.date < datetime.date(2017, 1, 1)]
            train = DBP.sample(frac = 1)
            test = DBP.ix[DBP.index.date > datetime.date(2016, 12, 31)]
            #test = test.ix[test.index.date < datetime.date(2017, 4, 12)]
            if test['y'].sum() < 10e-6:
                removed.append(pod)
                continue
            
            brf = RandomForestRegressor(criterion = 'mse', max_depth = 48, n_estimators = 24, n_jobs = 1)
            
            brf.fit(train[train.columns[:60]], train[train.columns[60]])
            yhat_train = brf.predict(train2[train2.columns[:60]])
            
            rfR2 = 1 - (np.sum((train2[train2.columns[60]] - yhat_train)**2))/(np.sum((train2[train2.columns[60]] - np.mean(train2[train2.columns[60]]))**2))
            print 'R-squared per pod {} = {}'.format(pod, rfR2)
            
            
            yhat_test = brf.predict(test[test.columns[:60]])
            rfR2_test = 1 - (np.sum((test[test.columns[60]] - yhat_test)**2))/(np.sum((test[test.columns[60]] - np.mean(test[test.columns[60]]))**2))
            print 'R-squared test per pod {} = {}'.format(pod, rfR2_test)
            
            if yhat_test.size < 2185:
                cc = np.concatenate([yhat_test, np.repeat(0, 2185 - yhat_test.size)]).tolist()
            else:
                cc = yhat_test.tolist()
            
            fdf[pod] = cc
        else:
            removed.append(pod)
    
    fdf = pd.DataFrame.from_dict(fdf, orient = "columns")
    fdf = fdf.set_index(pd.date_range('2017-01-01', '2017-12-31', freq = 'H')[:fdf.shape[0]])    
    
####################################################################################################
def LearnPodwiseModels(db, meteo, zona, end):
    if os.path.exists("C:/Users/utente/Documents/Sbilanciamento/" + zona + "/Podwise_models"):
        pass
    else:
        os.makedirs("C:/Users/utente/Documents/Sbilanciamento/" + zona + "/Podwise_models")
    removed = []
    podlist = list(set(db["POD"].ix[db["Area"] == zona].values.ravel().tolist()))
    for pod in podlist:
        print 'Avanzamento: {} %'.format((podlist.index(pod) + 1)/len(podlist))
        DBP = MakeExtendedDatasetGivenPOD(DB, meteo, pod, end)
            
        train2 = DBP.ix[DBP.index.date < datetime.date(2017, 1, 1)]
        train = DBP.sample(frac = 1)
        test = DBP.ix[DBP.index.date > datetime.date(2017, 1, 3)]
            
        if test['y'].sum() < 10e-6:
            removed.append(pod)
            continue
            
        brf = RandomForestRegressor(criterion = 'mse', max_depth = 48, n_estimators = 24, n_jobs = 1)
            
        brf.fit(train[train.columns[:74]], train[train.columns[74]])
        yhat_train = brf.predict(train2[train2.columns[:74]])
            
        rfR2 = 1 - (np.sum((train2[train2.columns[74]] - yhat_train)**2))/(np.sum((train2[train2.columns[74]] - np.mean(train2[train2.columns[74]]))**2))
        print 'R-squared per pod {} = {}'.format(pod, rfR2)
            
            
        yhat_test = brf.predict(test[test.columns[:74]])
        rfR2_test = 1 - (np.sum((test[test.columns[74]] - yhat_test)**2))/(np.sum((test[test.columns[74]] - np.mean(test[test.columns[74]]))**2))
        print 'R-squared test per pod {} = {}'.format(pod, rfR2_test)

        joblib.dump(brf, "C:/Users/utente/Documents/Sbilanciamento/" + zona + "/Podwise_models/" + pod + "_model.pkl")           
           
    
    return removed
####################################################################################################
db2016 = pd.read_excel("C:/Users/utente/Documents/Sbilanciamento/DB_2016.xlsm", sheetname = "DB_SI_perd")
db2 = removePerdite(db2016)
db2.to_hdf("C:/Users/utente/Documents/Sbilanciamento/DB_2016_originale.h5", "db2016")

db = pd.read_hdf("C:/Users/utente/Documents/Sbilanciamento/DB_2016_originale.h5")

ndb = reduceDB(db)
ndb.to_hdf("C:/Users/utente/Documents/Sbilanciamento/DB_2016_trattato.h5", "ndb")


k2e = pd.read_excel("C:/Users/utente/Documents/Sbilanciamento/Aggregato_copia.xlsx", sheetname = 'Forecast k2E', skiprows = [0,1,2])
k2e = k2e.set_index(pd.date_range('2017-01-01', '2018-01-02', freq = 'H')[:k2e.shape[0]])
#db = pd.read_excel("C:/Users/utente/Documents/Sbilanciamento/DB_2016_noperd.xlsx", converters = {'1': str, '2': str, '3': str,
#                                                                                                 '4': str, '5': str, '6': str,
#                '7': str, '8': str, '9': str, '10': str, '11': str, '12': str,
#                '13': str, '14': str, '15': str, '16': str, '17': str, '18': str,
#                '19': str, '20': str, '21': str, '22': str, '23': str, '24': str, '25': str} )
            
db = pd.read_hdf("C:/Users/utente/Documents/Sbilanciamento/DB_2016_noperd.h5")

db.columns = [["POD", "Area", "Giorno", "1","2","3","4","5","6","7","8","9","10","11","12","13","14","15",
         "16","17","18","19","20","21","22","23","24", "25"]]

db = db[["POD", "Area", "Giorno", "1","2","3","4","5","6","7","8","9","10","11","12","13","14","15",
         "16","17","18","19","20","21","22","23","24"]]


#dt = pd.read_excel("C:/Users/utente/Documents/Sbilanciamento/Aggregatore_orari - 17_03.xlsm",
#                   skiprows = [0], sheetname = "Consumi base 24")

db = pd.read_hdf("C:/Users/utente/Documents/Sbilanciamento/DB_2016_trattato.h5")


#dt = pd.read_excel("C:/Users/utente/Documents/Sbilanciamento/Aggregatore_orari-2017.xlsx")
dt = pd.read_excel("H:/Energy Management/02. EDM/01. MISURE/3. DISTRIBUTORI/ENEL Distribuzione S.p.A/2017/Aggregatore_orari-2017.xlsx")

dt.columns = [str(i) for i in dt.columns]

dt = dt[["POD", "Area", "Giorno", "1","2","3","4","5","6","7","8","9","10","11","12","13","14","15",
         "16","17","18","19","20","21","22","23","24"]]

dt = dt.drop_duplicates(subset = ['POD', 'Area', 'Giorno'], keep = 'last')

DB = db.append(dt, ignore_index = True) 

sbil = pd.read_excel('C:/Users/utente/Documents/misure/aggregato_sbilanciamento2.xlsx')



nord = sbil.ix[sbil['CODICE RUC'] == 'UC_DP1608_NORD']
nord.index = pd.date_range('2015-01-01', '2017-12-31', freq = 'H')[:nord.shape[0]]
mi6 = pd.read_excel('C:/Users/utente/Documents/PUN/Milano 2016.xlsx')
mi6 = mi6.ix[:365].set_index(pd.date_range('2016-01-01', '2016-12-31', freq = 'D'))
mi7 = pd.read_excel('C:/Users/utente/Documents/PUN/Milano 2017.xlsx')
mi7 = mi7.set_index(pd.date_range('2017-01-01', '2017-08-31', freq = 'D'))
mi = mi6.append(mi7)
###############################
cnord = sbil.ix[sbil['CODICE RUC'] == 'UC_DP1608_CNOR']
cnord.index = pd.date_range('2015-01-01', '2017-12-31', freq = 'H')[:cnord.shape[0]]
fi6 = pd.read_excel('C:/Users/utente/Documents/PUN/Firenze 2016.xlsx')
fi6 = fi6.ix[:365].set_index(pd.date_range('2016-01-01', '2016-12-31', freq = 'D'))
fi7 = pd.read_excel('C:/Users/utente/Documents/PUN/Firenze 2017.xlsx')
fi7 = fi7.set_index(pd.date_range('2017-01-01', '2017-06-30', freq = 'D'))
fi = fi6.append(fi7)
###############################
start = time.time()
oos = Get_OutOfSample(cnord, DB, "CNOR")
time.time() - start
start = time.time()
sam = Get_SampleAsTS(DB, "CNOR")
time.time() - start
cross = scipy.signal.correlate(sklearn.preprocessing.scale(sam.resample('M').mean().values), sklearn.preprocessing.scale(oos.resample('M').mean().values), mode = 'same')
npcross = np.convolve(sam.resample('M').mean().values.ravel(), oos.resample('M').mean().values.ravel(), mode = "same")
import statsmodels.tsa.stattools

ccf = statsmodels.tsa.stattools.ccf(sam.resample('D').sum().values.ravel(),oos.resample('D').sum().values.ravel())

plt.figure()
plt.plot(ccf, color = 'red')
v =  [1,2,3,4,5,6,7,8,9,10,11,12,13,14]
for vx in v:
    plt.axvline(x = vx, color = 'black')


plt.figure()
sam.resample('M').mean().plot()
oos.resample('M').mean().plot()


plt.figure()
plt.plot(cross.ravel())
v =  [744, 1440,2184,2904,3648,4368,5192,5856,6576,7320,8040,8784]
for vx in v:
    plt.axvline(x = vx, color = 'black')

plt.figure()    
plt.plot(sam.resample('M').sum().values.ravel()/(sam.resample('M').sum().values.ravel() + oos.resample('M').sum().values.ravel()))
###############################
csud = sbil.ix[sbil['CODICE RUC'] == 'UC_DP1608_CSUD']
csud.index = pd.date_range('2015-01-01', '2017-12-31', freq = 'H')[:csud.shape[0]]
ro6 = pd.read_excel('C:/Users/utente/Documents/PUN/Roma 2016.xlsx')
ro6 = ro6.ix[:365].set_index(pd.date_range('2016-01-01', '2016-12-31', freq = 'D'))
ro7 = pd.read_excel('C:/Users/utente/Documents/PUN/Fiumicino 2017.xlsx')
ro7 = ro7.set_index(pd.date_range('2017-01-01', '2017-06-30', freq = 'D'))
ro = ro6.append(ro7)
###############################
sud = sbil.ix[sbil['CODICE RUC'] == 'UC_DP1608_SUD']
sud.index = pd.date_range('2015-01-01', '2017-12-31', freq = 'H')[:sud.shape[0]]
rc6 = pd.read_excel('C:/Users/utente/Documents/PUN/Bari 2016.xlsx')
rc6 = rc6.ix[:365].set_index(pd.date_range('2016-01-01', '2016-12-31', freq = 'D'))
rc7 = pd.read_excel('C:/Users/utente/Documents/PUN/Bari 2017.xlsx')
rc7 = rc7.set_index(pd.date_range('2017-01-01', '2017-06-30', freq = 'D'))
rc = rc6.append(rc7)
###############################
sici = sbil.ix[sbil['CODICE RUC'] == 'UC_DP1608_SICI']
sici.index = pd.date_range('2015-01-01', '2017-12-31', freq = 'H')[:sici.shape[0]]
pa6 = pd.read_excel('C:/Users/utente/Documents/PUN/Palermo 2016.xlsx')
pa6 = pa6.ix[:365].set_index(pd.date_range('2016-01-01', '2016-12-31', freq = 'D'))
pa7 = pd.read_excel('C:/Users/utente/Documents/PUN/Palermo 2017.xlsx')
pa7 = pa7.set_index(pd.date_range('2017-01-01', '2017-06-30', freq = 'D'))
pa = pa6.append(pa7)
###############################
sard = sbil.ix[sbil['CODICE RUC'] == 'UC_DP1608_SARD']
sard.index = pd.date_range('2015-01-01', '2017-12-31', freq = 'H')[:sard.shape[0]]
ca6 = pd.read_excel('C:/Users/utente/Documents/PUN/Cagliari 2016.xlsx')
ca6 = ca6.ix[:365].set_index(pd.date_range('2016-01-01', '2016-12-31', freq = 'D'))
ca7 = pd.read_excel('C:/Users/utente/Documents/PUN/Cagliari 2017.xlsx')
ca7 = ca7.set_index(pd.date_range('2017-01-01', '2017-06-30', freq = 'D'))
ca = ca6.append(ca7)

######### mean behaviour per month ############                 ####################################
cd = convertDates(nord['DATA RIFERIMENTO CORRISPETTIVO'])
nord = nord.set_index(cd.values)

cd = convertDates(cnord['DATA RIFERIMENTO CORRISPETTIVO'])
cnord = cnord.set_index(cd.values)

cd = convertDates(csud['DATA RIFERIMENTO CORRISPETTIVO'])
csud = csud.set_index(cd.values)

cd = convertDates(sud['DATA RIFERIMENTO CORRISPETTIVO'])
sud = sud.set_index(cd.values)

cd = convertDates(sici['DATA RIFERIMENTO CORRISPETTIVO'])
sici = sici.set_index(cd.values)

cd = convertDates(sard['DATA RIFERIMENTO CORRISPETTIVO'])
sard = sard.set_index(cd.values)

df_t = transform_df(nord)
df_t = df_t.set_index(pd.date_range('2016-01-01', '2017-04-01', freq = 'D'))
df_t.resample('M').mean().T.plot()

####################################################################################################
####################################################################################################
print np.where(np.diff(nord['MO [MWh]']) >= 0)[0].size/np.diff(nord['MO [MWh]']).size
print np.where(np.diff(cnord['MO [MWh]']) >= 0)[0].size/np.diff(cnord['MO [MWh]']).size
print np.where(np.diff(csud['MO [MWh]']) >= 0)[0].size/np.diff(csud['MO [MWh]']).size
print np.where(np.diff(sud['MO [MWh]']) >= 0)[0].size/np.diff(sud['MO [MWh]']).size
print np.where(np.diff(sici['MO [MWh]']) >= 0)[0].size/np.diff(sici['MO [MWh]']).size
print np.where(np.diff(sard['MO [MWh]']) >= 0)[0].size/np.diff(sard['MO [MWh]']).size
###############################################                 ####################################
###############################################
DBB1 = MakeExtendedDatasetWithSampleCurve(nord, DB, mi, "NORD")
DBB2 = MakeExtendedDatasetWithSampleCurve(cnord, DB, fi, "CNOR")
DBB3 = MakeExtendedDatasetWithSampleCurve(csud, DB, ro, "CSUD")
DBB4 = MakeExtendedDatasetWithSampleCurve(sud, DB, rc, "SUD")
DBB5 = MakeExtendedDatasetWithSampleCurve(sici, DB, pa, "SICI")
DBB6 = MakeExtendedDatasetWithSampleCurve(sard, DB, ca, "SARD")

DBB1.to_hdf('C:/Users/utente/Documents/Sbilanciamento/nord.h5', 'nord')
DBB2.to_hdf('C:/Users/utente/Documents/Sbilanciamento/cnord.h5', 'cnord')
DBB3.to_hdf('C:/Users/utente/Documents/Sbilanciamento/csud.h5', 'csud')
DBB4.to_hdf('C:/Users/utente/Documents/Sbilanciamento/sud.h5', 'sud')
DBB5.to_hdf('C:/Users/utente/Documents/Sbilanciamento/sici.h5', 'sici')
DBB6.to_hdf('C:/Users/utente/Documents/Sbilanciamento/sard.h5', 'sard')
###############################################
###############################################
DBB = MakeDatasetLongTerm(nord, DB, mi, "NORD", 5)
DBB = MakeDatasetLongTerm(cnord, DB, fi, "CNOR",5)
DBB = MakeDatasetLongTerm(csud, DB, ro, "CSUD",5)
DBB = MakeDatasetLongTerm(sud, DB, rc, "SUD",5)
DBB = MakeDatasetLongTerm(sici, DB, pa, "SICI",5)
DBB = MakeDatasetLongTerm(sard, DB, ca, "SARD",5)

DBB.to_hdf('C:/Users/utente/Documents/Sbilanciamento/nord5.h5', 'nord')
DBB.to_hdf('C:/Users/utente/Documents/Sbilanciamento/cnord5.h5', 'cnord')
DBB.to_hdf('C:/Users/utente/Documents/Sbilanciamento/csud5.h5', 'csud')
DBB.to_hdf('C:/Users/utente/Documents/Sbilanciamento/sud5.h5', 'sud')
DBB.to_hdf('C:/Users/utente/Documents/Sbilanciamento/sici5.h5', 'sici')
DBB.to_hdf('C:/Users/utente/Documents/Sbilanciamento/sard5.h5', 'sard')
####################################################
####################################################

DBB = MakeExtendedDatasetWithSampleCurve(nord, DB, mi, "NORD")
DBB = MakeExtendedDatasetWithSampleCurve(cnord, DB, fi, "CNOR")
DBB = MakeExtendedDatasetWithSampleCurve(csud, DB, ro, "CSUD")
DBB = MakeExtendedDatasetWithSampleCurve(sud, DB, rc, "SUD")
DBB = MakeExtendedDatasetWithSampleCurve(sici, DB, pa, "SICI")
DBB = MakeExtendedDatasetWithSampleCurve(sard, DB, ca, "SARD")

DBB.to_hdf('C:/Users/utente/Documents/Sbilanciamento/nord.h5', 'nord')
DBB.to_hdf('C:/Users/utente/Documents/Sbilanciamento/cnord.h5', 'cnord')
DBB.to_hdf('C:/Users/utente/Documents/Sbilanciamento/csud.h5', 'csud')
DBB.to_hdf('C:/Users/utente/Documents/Sbilanciamento/sud.h5', 'sud')
DBB.to_hdf('C:/Users/utente/Documents/Sbilanciamento/sici.h5', 'sici')
DBB.to_hdf('C:/Users/utente/Documents/Sbilanciamento/sard.h5', 'sard')
#### check on holidays ####
feste = DBB.ix[DBB['holiday'] == 1].index
feste = set(feste.tolist())

### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
 ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
DBB = pd.read_hdf('C:/Users/utente/Documents/Sbilanciamento/nord.h5')
DBB = pd.read_hdf('C:/Users/utente/Documents/Sbilanciamento/cnord.h5')
DBB = pd.read_hdf('C:/Users/utente/Documents/Sbilanciamento/csud.h5')
DBB = pd.read_hdf('C:/Users/utente/Documents/Sbilanciamento/sud.h5')
DBB = pd.read_hdf('C:/Users/utente/Documents/Sbilanciamento/sici.h5')
DBB = pd.read_hdf('C:/Users/utente/Documents/Sbilanciamento/sard.h5')
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
 ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
DBB = DBB1
DBB = DBB2
DBB = DBB3
DBB = DBB4
DBB = DBB5
DBB = DBB6

train2 = DBB.ix[DBB.index.date < datetime.date(2017, 2, 1)]
train = DBB.sample(frac = 1)
test = DBB.ix[DBB.index.date > datetime.date(2017, 1, 31)]
#test = test.ix[test.index.date < datetime.date(2017, 4, 12)]



###### check to see if the initial error is due to the model or some other pecularity
###### happened in January 2017
#train2 = DBB.ix[DBB.index.date < datetime.date(2017, 2, 1)]
#train = DBB.sample(frac = 1)
#test = DBB.ix[DBB.index.date > datetime.date(2017, 1, 31)]
#test = test.ix[test.index.date < datetime.date(2017, 4, 12)]


### It seems the performances depend on the initial permutation of the trainingset
### Google: cross validation with random forest sklearn

#bre =  AdaBoostRegressor(RandomForestRegressor(criterion = 'mse', max_depth = 24, n_jobs = 1), n_estimators=3000)
bre = AdaBoostRegressor(DecisionTreeRegressor(criterion = 'mse', max_depth = 24), n_estimators=3000)

brf = RandomForestRegressor(criterion = 'mse', max_depth = 48, n_estimators = 24, n_jobs = 1)

brf.fit(train[train.columns[:76]], train[train.columns[76]])
yhat_train = brf.predict(train2[train2.columns[:76]])

rfR2 = 1 - (np.sum((train2[train2.columns[76]] - yhat_train)**2))/(np.sum((train2[train2.columns[76]] - np.mean(train2[train2.columns[76]]))**2))
print rfR2


yhat_test = brf.predict(test[test.columns[:76]])
rfR2_test = 1 - (np.sum((test[test.columns[76]] - yhat_test)**2))/(np.sum((test[test.columns[76]] - np.mean(test[test.columns[76]]))**2))
print rfR2_test
print r2_score(test[test.columns[76]], yhat_test)

zona = "SICI"
pd.DataFrame({'volumi_forecast': yhat_test}).set_index(test.index).to_excel("C:/Users/utente/Documents/Sbilanciamento/Backtesting/volumi" + zona + ".xlsx")
############################# TEST on error adjustment #############################################

sad = Get_SampleAsTS_AtDay(DB, "NORD", '2017-02-01', '2017-04-30')
pc2 = percentageConsumption2(DB, "NORD", '2017-02-01', '2017-04-30')
yht = pd.DataFrame({'yhat': yhat_test}).set_index(test.index)
est_sample = []
for i in pc2.index:
    yhtd = yht.ix[yht.index.date == pd.to_datetime(i).date()].values.ravel()
    res = (yhtd * pc2.ix[i].values).tolist()
    est_sample.extend(res)

ES = pd.DataFrame({"sam_hat": est_sample})
ES = ES.set_index(test.index)
add = pd.DataFrame({"sam_hat":np.repeat(0, 24)})
add = add.set_index(pd.date_range('2017-03-26', '2017-03-27', freq = 'H')[:add.shape[0]])
ES = ES.append(add.ix[2])
ES = ES.sort_index()

add2 = pd.DataFrame({"y":np.repeat(0, 24)})
add2 = add.set_index(pd.date_range('2017-03-26', '2017-03-27', freq = 'H')[:add.shape[0]])
TE = pd.DataFrame({'y': test['y'].values.ravel().tolist()}).set_index(test.index)
TE['y'].ix[datetime.datetime(2017,3,26,2)] = 0
TE = TE['y'].sort_index()

DFE = pd.DataFrame({"error": sad.values.ravel() - ES.values.ravel(), "yy": TE})


DTE = CorrectionDataset(test, yhat_test, DB, mi, "NORD")

print np.corrcoef(DTE['r'], DTE['y'])
plt.figure()
plt.plot(np.correlate(DTE['r'], DTE['y'], 'same'))


for h in range(24):
    DTEh = DTE.ix[DTE['t' + str(h)] == 1]
    plt.figure()
    plt.scatter(DTEh['r'], DTEh['y'])
    plt.title('error vs target_value at h = {}'.format(h))

plt.figure()
DTE[['r', 'y']].plot()

############ error by day ############
days = ['Lun', 'Mar', 'Mer', 'Gio', 'Ven', 'Sab', 'Dom']
for d in days:
    DTEd = DTE.ix[DTE[d] == 1]
    plt.figure()
    plt.hist(DTEd['r'], bins = 20)
    print DTEd['r'].describe()
############ error by hour ############
bar = []
std_bar = []
std_y = []
bar_T = []
T_std = []
for h in range(24):
    DTEh = DTE.ix[DTE['t' + str(h)] == 1]
    bar.append(np.corrcoef(DTEh['r'], DTEh['y'])[0,1])
    std_bar.append(DTEh['r'].std())
    std_y.append(DTEh['y'].std())
    bar_T.append(np.corrcoef(DTEh['tmax'], DTEh['y'])[0,1])
    T_std.append(DTEh['tmax'].std())
plt.figure()
plt.bar(np.arange(24),np.array(bar))
plt.figure()
plt.bar(np.arange(24),np.array(std_bar), color = 'yellow')
plt.figure()
plt.bar(np.arange(24),np.array(std_y), color = 'orange')
plt.figure()
plt.bar(np.arange(24),np.array(bar_T), color = 'coral')
plt.figure()
plt.bar(np.arange(24),np.array(T_std), color = 'red')

print np.corrcoef(np.array(std_bar), np.array(std_y))

params = {'n_estimators': 500, 'max_depth': 48, 'min_samples_split': 2,
          'learning_rate': 0.01, 'loss': 'ls', 'criterion': 'friedman_mse'}
gbm = GradientBoostingRegressor(**params)

##### NORMAL
gbm.fit(X = DTE[DTE.columns[:52]], y = DTE['y'])
mse = mean_squared_error(DTE['y'], gbm.predict(DTE[DTE.columns[:52]]))
print mse
R2 = r2_score(DTE["y"].reshape(DTE["y"].size, 1), gbm.predict(DTE[DTE.columns[:52]]))
print R2
yg_train_n = gbm.predict(DTE[DTE.columns[:52]])
##### SHORT = True
gbm.fit(X = DTE['error'].reshape(DTE["error"].size, 1), y = DTE['yy'])
mse = mean_squared_error(DTE['yy'], gbm.predict(DTE['error'].reshape(DTE["error"].size, 1)))
print mse
R2 = r2_score(DTE["yy"].reshape(DTE["yy"].size, 1), gbm.predict(DTE['error'].reshape(DTE["error"].size, 1)))
print R2
yg_train_n = gbm.predict(DTE['error'].reshape(DTE["error"].size, 1))

############# simpler model ################
bre = RandomForestRegressor(criterion = 'mse', max_depth = 48, n_estimators = 24, n_jobs = 1)

DTE2 = DTE
DTE = DTE.sample(frac = 1)

bre.fit(DTE[DTE.columns[:52]], DTE['y'])

ye = bre.predict(DTE[DTE.columns[:52]])
print r2_score(DTE['y'], ye)

plt.figure()
plt.plot(test['y'].ix[-yg_train_n.size:].values.ravel(), color = "navy", marker = "o", lw = 4)
plt.plot(yg_train_n, color = "orange", marker = "+")

samerror = TE.ix[-yg_train_n.size:].values.ravel() - yg_train_n

plt.figure()
plt.plot(samerror, color = "indigo", marker = 'o')

plt.figure()
plt.plot(test['y'].values.ravel(), color = "orange", marker = "*")
plt.plot(np.concatenate((yhat_test[:71].tolist(), yg_train_n.tolist()), axis = 0), color = "coral", marker = "+")

plt.figure()
plt.hist(TE.values.ravel() - yg_train_n, bins = 20)
plt.figure()
plt.plot(TE.values.ravel() - yg_train_n, color = "grey")

plt.figure()
plotting.autocorrelation_plot(TE.values.ravel() - yg_train_n)

gerror = test['y'] - np.concatenate((yhat_test[:72].tolist(), yg_train_n.tolist()), axis = 0)

plt.figure()
plt.plot(test['y'])
plt.figure()
plt.plot(np.concatenate((yhat_test[:71].tolist(), yg_train_n.tolist()), axis = 0), color = 'red')
plt.figure()
plt.hist(test['y'], bins = 20)
plt.figure()
plt.hist(np.concatenate((yhat_test[:72].tolist(), yg_train_n.tolist()), axis = 0), bins = 20, color = 'red')

plt.figure();plt.plot(yhat_test, color = 'red');plt.plot(DTE['r'].values.ravel())

plt.figure()
plt.plot(gerror, color = 'grey')

print np.mean(gerror)
print np.median(gerror)
print np.std(gerror)
print np.min(gerror)
print np.max(gerror)

gMAE = gerror/test['y'].values.ravel()
gMAE[1274] = 0
gMAE = np.nan_to_num(gMAE)

print np.mean(gMAE)
print np.median(gMAE)
print np.std(gMAE)
print np.min(gMAE)
print np.max(gMAE)

plt.figure()
plt.plot(gMAE)
plt.axhline(y = max(gMAE))
plt.axhline(y = min(gMAE))
plt.figure()
plt.hist(gMAE, bins = 20, color = "yellow")

feature_importance = gbm.feature_importances_
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5
plt.figure()
plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.yticks(pos, DTE.columns[sorted_idx])
plt.xlabel('Relative Importance')
plt.title('Variable Importance')
plt.show()

from sklearn.ensemble.partial_dependence import plot_partial_dependence

features = [50, 51, (50, 51)]
plt.figure()
fig, axs = plot_partial_dependence(gbm, DTE[DTE.columns[:52]], [6]) 
####################################################################################################

plt.figure()
plt.plot(yhat_test, color = 'blue', marker = 'o')
plt.plot(test['y'].values.ravel(), color = 'red', marker = '+')

pd.DataFrame.from_dict({'prediction': yhat_test.tolist()}, orient = 'columns').set_index(test.index).to_excel('backtesting_SARD.xlsx')
#### graphical comparison with k2e
nk2e = k2e["NORD"]/1000
tnk2e = nk2e.ix[nk2e.index < datetime.datetime(2017,4,1)]
cnk2e = k2e["CNOR"]/1000
tcnk2e = cnk2e.ix[cnk2e.index < datetime.datetime(2017,4,1)]
csk2e = k2e["CSUD"]/1000
tcsk2e = csk2e.ix[csk2e.index < datetime.datetime(2017,4,1)]
sk2e = k2e["SUD"]/1000
tsk2e = sk2e.ix[sk2e.index < datetime.datetime(2017,4,1)]
sik2e = k2e["SICI"]/1000
tsik2e = sik2e.ix[sik2e.index < datetime.datetime(2017,4,1)]
sak2e = k2e["SARD"]/1000
tsak2e = sak2e.ix[sak2e.index < datetime.datetime(2017,4,1)]


plt.figure()
plt.plot(yhat_test, color = 'blue', marker = 'o', label = 'Axopower')
plt.plot(test[test.columns[63]].values.ravel(), color = 'red', marker = 'x', label = 'Terna')
plt.plot(tnk2e.values.ravel(), color = 'black', marker = '8', label = 'K2E')
plt.legend(loc = 'upper left')
plt.title("Forecast zona NORD")
##############################
plt.figure()
plt.plot(yhat_test, color = 'blue', marker = 'o', label = 'Axopower')
plt.plot(test[test.columns[63]].values.ravel(), color = 'red', marker = 'x', label = 'Terna')
plt.plot(tcnk2e.values.ravel(), color = 'black', marker = '8', label = 'K2E')
plt.legend(loc = 'upper left')
plt.title("Forecast zona CNORD")
##############################
plt.figure()
plt.plot(yhat_test, color = 'blue', marker = 'o', label = 'Axopower')
plt.plot(test[test.columns[63]].values.ravel(), color = 'red', marker = 'x', label = 'Terna')
plt.plot(tcsk2e.values.ravel(), color = 'black', marker = '8', label = 'K2E')
plt.legend(loc = 'upper left')
plt.title("Forecast zona CSUD")
##############################
plt.figure()
plt.plot(yhat_test, color = 'blue', marker = 'o', label = 'Axopower')
plt.plot(test[test.columns[63]].values.ravel(), color = 'red', marker = 'x', label = 'Terna')
plt.plot(tsk2e.values.ravel(), color = 'black', marker = '8', label = 'K2E')
plt.legend(loc = 'upper left')
plt.title("Forecast zona SUD")
##############################
plt.figure()
plt.plot(yhat_test, color = 'blue', marker = 'o', label = 'Axopower')
plt.plot(test[test.columns[63]].values.ravel(), color = 'red', marker = 'x', label = 'Terna')
plt.plot(tsik2e.values.ravel(), color = 'black', marker = '8', label = 'K2E')
plt.legend(loc = 'upper left')
plt.title("Forecast zona SICI")
##############################
plt.figure()
plt.plot(yhat_test, color = 'blue', marker = 'o', label = 'Axopower')
plt.plot(test[test.columns[63]].values.ravel(), color = 'red', marker = 'x', label = 'Terna')
plt.plot(tsak2e.values.ravel(), color = 'black', marker = '8', label = 'K2E')
plt.legend(loc = 'upper left')
plt.title("Forecast zona SARD")
##############################


error = test['y'].values.ravel() - yhat_test    


k2e_error = test[test.columns[63]].values.ravel() - tnk2e.values.ravel()[:tnk2e.values.ravel().size - 1] 
k2e_error = test[test.columns[63]].values.ravel() - tcnk2e.values.ravel()[:tcnk2e.values.ravel().size - 1]     
k2e_error = test[test.columns[63]].values.ravel() - tcsk2e.values.ravel()[:tcsk2e.values.ravel().size - 1]     
k2e_error = test[test.columns[63]].values.ravel() - tsk2e.values.ravel()[:tsk2e.values.ravel().size - 1]     
k2e_error = test[test.columns[63]].values.ravel() - tsik2e.values.ravel()[:tsik2e.values.ravel().size - 1]     
k2e_error = test[test.columns[63]].values.ravel() - tsak2e.values.ravel()[:tsak2e.values.ravel().size - 1]     

print np.mean(k2e_error)
print np.median(k2e_error)
print np.std(k2e_error)

print np.mean(error)
print np.median(error)
print np.std(error)

plt.figure() 
plt.hist(k2e_error, bins = 20, color = 'green')   
plt.figure() 
plt.hist(error, bins = 20)   

maek2 = k2e_error/test[test.columns[63]].values.ravel()
mae = error/test['y'].values.ravel()

mae[np.where(test['y'].values.ravel() == 0)[0]] = 0

print np.mean(maek2)
print np.median(maek2)
print np.std(maek2)
print np.max(maek2)
print np.min(maek2)

print np.mean(mae)
print np.median(mae)
print np.std(mae)
print np.max(mae)
print np.min(mae)

###### ABSOLUTE ERRORS ########
amaek2 = np.abs(k2e_error)/test[test.columns[63]].values.ravel()
amae = np.abs(error)/test['y'].values.ravel()

amae[np.where(test['y'].values.ravel() == 0)[0]] = 0


print np.mean(amaek2)
print np.median(amaek2)
print np.std(amaek2)
print np.max(amaek2)
print np.min(amaek2)

print np.mean(amae)
print np.median(amae)
print np.std(amae)
print np.max(amae)
print np.min(amae)
##############################

print scipy.stats.mstats.mquantiles(maek2, prob = [0.025, 0.975])
print scipy.stats.mstats.mquantiles(mae, prob = [0.025, 0.975])

zona = "SARD"

plt.figure()
plt.plot(maek2, color = 'green')
plt.axhline(y = 0.15)
plt.axhline(y = -0.15)
plt.title("Errore percentuale forecast K2E zona " + zona)
plt.figure()
plt.plot(mae, color = 'blue')
plt.axhline(y = 0.15)
plt.axhline(y = -0.15)
plt.title("Errore percentuale forecast AXO zona " + zona)


np.corrcoef(maek2, mae)

plt.figure()
plotting.autocorrelation_plot(maek2, color = 'green')
plt.title("Autocorrelazione errore percentuale forecast K2E zona " + zona)
plt.figure()
plotting.autocorrelation_plot(mae)
plt.title("Autocorrelazione errore percentuale forecast AXO zona " + zona)
plt.figure()
plotting.autocorrelation_plot(amae, color = 'orange')
plt.title("Autocorrelazione errore percentuale assoluto forecast AXO zona " + zona)


plt.figure() 
plt.hist(maek2, bins = 20, color = 'green')   
plt.figure() 
plt.hist(mae, bins = 20)   

print np.where(np.abs(maek2) >= 0.15)[0].size/maek2.size
print np.where(np.abs(mae) >= 0.15)[0].size/mae.size


k2e_error = pd.DataFrame.from_dict({"error":k2e_error}).set_index(test.index)
axo_error = pd.DataFrame.from_dict({"error":error}).set_index(test.index)

K2E = getImbalance(k2e_error, tnk2e.values.ravel()[:tnk2e.values.ravel().size - 1])
K2E = getImbalance(k2e_error, tcnk2e.values.ravel()[:tcnk2e.values.ravel().size - 1])
K2E = getImbalance(k2e_error, tcsk2e.values.ravel()[:tcsk2e.values.ravel().size - 1])
K2E = getImbalance(k2e_error, tsk2e.values.ravel()[:tsk2e.values.ravel().size - 1])
K2E = getImbalance(k2e_error, tsik2e.values.ravel()[:tsik2e.values.ravel().size - 1])
K2E = getImbalance(k2e_error, tsak2e.values.ravel()[:tsak2e.values.ravel().size - 1])


AXO = getImbalance(axo_error, yhat_test)

##### convert values to numbers and remove the round brackets from the excel file in the file below
val = pd.read_excel("C:/Users/utente/Documents/Sbilanciamento/valorizzazione_sbilanciamento.xlsx")
pun = pd.read_excel("H:/Energy Management/04. WHOLESALE/02. REPORT PORTAFOGLIO/2017/06. MI/DB_Borse_Elettriche_PER MI_17_conMacro_072017.xlsm", sheetname = "DB_Dati")
pun = pun.set_index(pd.date_range("2017-01-01", "2018-01-02", freq = "H")[:pun.shape[0]])
pun = pun.ix[pun.index.date <= today.date()]
pun = pun[pun.columns[[12,16,19,24,27,28,30]]]
pun.columns = [["PUN","CNOR","CSUD","NORD","SARD","SICI","SUD"]]

pun = pun.ix[pun.index >= axo_error.index[0]]
pun = pun.ix[pun.index <= axo_error.index[-1]]
##### Test for valorization ####
spn = nord["SBILANCIAMENTO FISICO [MWh]"].values.ravel()/nord["PV [MWh]"].values.ravel()
prova = pd.DataFrame.from_dict({"Sbil": nord["SBILANCIAMENTO FISICO [MWh]"].values.ravel().tolist(),
                                "Sbil_perc": spn.tolist()})
prova = prova.set_index(nord.index)                                
VS = EvaluateImbalance(val, prova, "NORD", pun)                                
#####                                

imb_k2e = EvaluateImbalance(val, K2E, "NORD", pun)
imb_k2e = EvaluateImbalance(val, K2E, "CNOR", pun)
imb_k2e = EvaluateImbalance(val, K2E, "CSUD", pun)
imb_k2e = EvaluateImbalance(val, K2E, "SUD", pun)
imb_k2e = EvaluateImbalance(val, K2E, "SICI", pun)
imb_k2e = EvaluateImbalance(val, K2E, "SARD", pun)


imb_k2e.mean()
imb_k2e.sum()
imb_k2e.to_excel("C:/Users/utente/Documents/Sbilanciamento/Backtesting/sbilanciamento_k2e_" + zona + ".xlsx")

imb_axo = EvaluateImbalance(val, AXO, "NORD", pun)
imb_axo = EvaluateImbalance(val, AXO, "CNOR", pun)
imb_axo = EvaluateImbalance(val, AXO, "CSUD", pun)
imb_axo = EvaluateImbalance(val, AXO, "SUD", pun)
imb_axo = EvaluateImbalance(val, AXO, "SICI", pun)
imb_axo = EvaluateImbalance(val, AXO, "SARD", pun)

zona = "SICI"
print imb_axo.mean()
print imb_axo.sum()
imb_axo.to_excel("C:/Users/utente/Documents/Sbilanciamento/Backtesting/sbilanciamento_axo_" + zona + ".xlsx")

######### Imbalance on MO and Sample new algorithm 
vs = pd.read_excel('C:/Users/utente/Documents/Sbilanciamento/valsbil.xlsx', sheetname = 'MO')

zone = ["CNOR","CSUD","NORD","SARD","SICI","SUD"]
for zona in zone:
    dz = sbil.ix[sbil['CODICE RUC'] == 'UC_DP1608_' + zona]
    dz.index = pd.date_range('2015-01-01', '2017-12-31', freq = 'H')[:dz.shape[0]]
    
    cd = convertDates(dz['DATA RIFERIMENTO CORRISPETTIVO'])
    dz = dz.set_index(cd.values)
    dz = dz.ix[dz.index.date > datetime.date(2016,12,31)]
    error = dz['MO [MWh]'].values.ravel() - vs[zona].values.ravel()
    error = pd.DataFrame({'error': error}).set_index(dz.index)
    imb = getImbalance(error, vs[zona].values.ravel())
    
    valimb = EvaluateImbalance(val, imb, zona, pun)
    aggvi = valimb.resample('M').sum()
    aggvol = dz['MO [MWh]'].resample('M').sum().values.ravel()
    aggvi['volume'] = aggvol
    aggvi['val_sbil_medio'] = aggvi['Val_sbil'].values.ravel()/aggvol
    aggvi['gain_medio'] = aggvi['gain'].values.ravel()/aggvol
    
    print aggvi
    aggvi.to_excel('MO_stat_sbil_' + zona + '.xlsx')

### SAMPLE 
vs = pd.read_excel('C:/Users/utente/Documents/Sbilanciamento/valsbil.xlsx', sheetname = 'SAMPLE')
vs = vs.set_index(pd.date_range('2017-05-20', '2017-12-31', freq = 'H')[:vs.shape[0]])

unareti = []
for i in db.index:
    pod = db['POD'].ix[i]
    if pod[:5] == 'IT006' or pod[:5] == 'IT012':
        unareti.append(i)

db_nouna = db.drop(unareti)

#zona = 'NORD'
for zona in zone:
    dz = Get_SampleAsTS(db_nouna, zona)
    
    error = dz.ix[vs.index].values.ravel() - vs[zona].values.ravel()
    error = pd.DataFrame({'error': error}).set_index(vs.index)
    imb = getImbalance(error, vs[zona].values.ravel())
    
    valimb = EvaluateImbalance(val, imb, zona, pun)
    aggvi = valimb.resample('M').sum()
    aggvol = dz.ix[vs.index].resample('M').sum().values.ravel()
    aggvi['volume'] = aggvol
    aggvi['val_sbil_unitario'] = aggvi['Val_sbil'].values.ravel()/aggvol
    aggvi['gain_unitario'] = aggvi['gain'].values.ravel()/aggvol
    print aggvi

    aggvi.to_excel('campione_stat_sbil_' + zona + '.xlsx')


#########


###### podwise forecast

DBP = MakeExtendedDatasetGivenPOD(DB, mi, "IT001E00123781")

train2 = DBP.ix[DBP.index.date < datetime.date(2017, 1, 1)]
train = DBP.sample(frac = 1)
test = DBP.ix[DBP.index.date > datetime.date(2016, 12, 31)]
test = test.ix[test.index.date < datetime.date(2017, 4, 12)]


### It seems the performances depend on the initial permutatio of the trainingset
### Google: cross validation with random forest sklearn

#ffregr = AdaBoostRegressor(DecisionTreeRegressor(criterion = 'mse', max_depth = 24), n_estimators=3000)
ffregr =  AdaBoostRegressor(RandomForestRegressor(criterion = 'mse', max_depth = 24, n_jobs = 1), n_estimators=3000)

brf = RandomForestRegressor(criterion = 'mse', max_depth = 48, n_estimators = 24, n_jobs = 1)

brf.fit(train[train.columns[:60]], train[train.columns[60]])
yhat_train = brf.predict(train2[train2.columns[:60]])

rfR2 = 1 - (np.sum((train2[train2.columns[60]] - yhat_train)**2))/(np.sum((train2[train2.columns[60]] - np.mean(train2[train2.columns[60]]))**2))
print rfR2


yhat_test = brf.predict(test[test.columns[:60]])
rfR2_test = 1 - (np.sum((test[test.columns[60]] - yhat_test)**2))/(np.sum((test[test.columns[60]] - np.mean(test[test.columns[60]]))**2))
print rfR2_test

plt.figure()
plt.plot(yhat_test, color = 'blue', marker = 'o')
plt.plot(test[test.columns[60]].values.ravel(), color = 'red', marker = '+')

errpod = test[test.columns[60]].values.ravel() - yhat_test
MAE = errpod/test[test.columns[60]].values.ravel()

plt.figure()
plt.plot(errpod, color = 'orange')
plt.figure()
plt.plot(MAE, color = 'navy')
plt.axhline(y = 0.15, color = 'red')
plt.axhline(y = -0.15, color = 'red')


forecast_per_pod = podwiseForecast(DB, mi, "NORD")
forecast_per_pod = podwiseForecast(DB, fi, "CNOR")
forecast_per_pod, rem = podwiseForecast(DB, ro, "CSUD")
fdf = podwiseForecast(DB, rc, "SUD")



fdf = fdf[0]
rem = fdf[1]

csud_ = DB.ix[DB["Area"] == "CSUD"]
CSUD = pd.DataFrame()
for pod in forecast_per_pod.columns:
    CSUD = CSUD.append(DB.ix[DB["POD"] == pod])

csud_gg = []
for d in pd.date_range('2017-01-01', '2017-03-31', freq = 'D'):
    #print d.month
    #print CSUD.ix[CSUD["Giorno"] == d].shape
    csud_gg.extend(CSUD.ix[CSUD["Giorno"] == d].sum())


err = forecast_per_pod.sum(axis = 1).values[:2160] - np.array(csud_gg)

csud_gg[2018] = 1
err[2018] = 1

R2_glob = 1 - (np.sum((err)**2))/(np.sum((np.array(csud_gg) - np.mean(np.array(csud_gg)))**2))
print R2_glob

plt.figure()
plt.plot(np.array(csud_gg), marker = '8', color = 'green')
plt.plot(forecast_per_pod.sum(axis = 1).values, marker = 'o', color = 'magenta')

errmae = err/np.array(csud_gg)
errmae[2018] = 0

print np.mean(err)
print np.median(err)
print np.std(err)
print np.mean(errmae)
print np.median(errmae)
print np.std(errmae)
print np.max(errmae)
print np.min(errmae)

plt.figure()
plt.plot(errmae)
plt.axhline(y = 0.15)
plt.axhline(y = -0.15)

################################
######## Test on May 2017 ######
################################

DBB = pd.read_hdf('C:/Users/utente/Documents/Sbilanciamento/nord.h5')
DBB = pd.read_hdf('C:/Users/utente/Documents/Sbilanciamento/cnord.h5')
DBB = pd.read_hdf('C:/Users/utente/Documents/Sbilanciamento/csud.h5')
DBB = pd.read_hdf('C:/Users/utente/Documents/Sbilanciamento/sud.h5')
DBB = pd.read_hdf('C:/Users/utente/Documents/Sbilanciamento/sici.h5')
DBB = pd.read_hdf('C:/Users/utente/Documents/Sbilanciamento/sard.h5')


mi2017 = pd.read_excel('C:/Users/utente/Documents/meteo/Milano.xlsx')
mi2017 = mi2017[["Tmin", "Tmax", "Tmedia", "vento", "pioggia"]]
fi2017 = pd.read_excel('C:/Users/utente/Documents/meteo/Firenze.xlsx')
fi2017 = fi2017[["Tmin", "Tmax", "Tmedia", "vento", "pioggia"]]
ro2017 = pd.read_excel('C:/Users/utente/Documents/meteo/Roma.xlsx')
ro2017 = ro2017[["Tmin", "Tmax", "Tmedia", "vento", "pioggia"]]
ba2017 = pd.read_excel('C:/Users/utente/Documents/meteo/Bari.xlsx')
ba2017 = ba2017[["Tmin", "Tmax", "Tmedia", "vento", "pioggia"]]
pa2017 = pd.read_excel('C:/Users/utente/Documents/meteo/Palermo.xlsx')
pa2017 = pa2017[["Tmin", "Tmax", "Tmedia", "vento", "pioggia"]]
ca2017 = pd.read_excel('C:/Users/utente/Documents/meteo/Cagliari.xlsx')
ca2017 = ca2017[["Tmin", "Tmax", "Tmedia", "vento", "pioggia"]]

############################
mi52017 = pd.read_excel('C:/Users/utente/Documents/meteo/5-giorni/Milano.xlsx')
mi52017 = mi52017[["Tmin", "Tmax", "Tmedia", "vento", "pioggia"]]
fi52017 = pd.read_excel('C:/Users/utente/Documents/meteo/5-giorni/Firenze.xlsx')
fi52017 = fi52017[["Tmin", "Tmax", "Tmedia", "vento", "pioggia"]]
ro52017 = pd.read_excel('C:/Users/utente/Documents/meteo/5-giorni/Roma.xlsx')
ro52017 = ro52017[["Tmin", "Tmax", "Tmedia", "vento", "pioggia"]]
ba52017 = pd.read_excel('C:/Users/utente/Documents/meteo/5-giorni/Bari.xlsx')
ba52017 = ba52017[["Tmin", "Tmax", "Tmedia", "vento", "pioggia"]]
pa52017 = pd.read_excel('C:/Users/utente/Documents/meteo/5-giorni/Palermo.xlsx')
pa52017 = pa52017[["Tmin", "Tmax", "Tmedia", "vento", "pioggia"]]
ca52017 = pd.read_excel('C:/Users/utente/Documents/meteo/5-giorni/Cagliari.xlsx')
ca52017 = ca52017[["Tmin", "Tmax", "Tmedia", "vento", "pioggia"]]

DBT = MakeForecastLongTerm(DB, mi52017, "NORD",5)
DBT = MakeForecastLongTerm(DB, fi52017, "CNOR",5)
DBT = MakeForecastLongTerm(DB, ro52017, "CSUD",5)
DBT = MakeForecastLongTerm(DB, ba52017, "SUD",5)
DBT = MakeForecastLongTerm(DB, pa52017, "SICI",5)
DBT = MakeForecastLongTerm(DB, ca52017, "SARD",5)
###########################


DBH = DBB.ix[DBB['holiday'] == 1]
DBN = DBB.ix[DBB['holiday'] == 0]

DBtrain = DBB.sample(frac = 1)

DBHtrain = DBH.sample(frac = 1)
DBNtrain = DBN.sample(frac = 1)


DBT = MakeForecastDataset(DB, mi2017, "NORD")
DBT = MakeForecastDataset(DB, fi2017, "CNOR")
DBT = MakeForecastDataset(DB, ro2017, "CSUD")
DBT = MakeForecastDataset(DB, ba2017, "SUD")
DBT = MakeForecastDataset(DB, pa2017, "SICI")
DBT = MakeForecastDataset(DB, ca2017, "SARD")



DBTH =  DBT.ix[DBT['holiday'] == 1]
DBTN =  DBT.ix[DBT['holiday'] == 0]

################################ Separation holiday/non-holidays ###################################
brfh = RandomForestRegressor(criterion = 'mse', max_depth = 48, n_estimators = 24, n_jobs = 1)

brfh.fit(DBHtrain[DBHtrain.columns[:76]], DBHtrain[DBHtrain.columns[76]])
yhat_train_h = brfh.predict(DBHtrain[DBHtrain.columns[:76]])

rfR2H = 1 - (np.sum((DBHtrain[DBHtrain.columns[76]] - yhat_train_h)**2))/(np.sum((DBHtrain[DBHtrain.columns[76]] - np.mean(DBHtrain[DBHtrain.columns[76]]))**2))
print rfR2H

yhat_test_h = brfh.predict(DBTH)

yth = pd.DataFrame.from_dict({'pred': yhat_test_h.tolist()})
yth = yth.set_index(DBTH.index)
yth.plot(color = 'orchid')

yth.to_excel("NORD_previsione_2017-06-02.xlsx")
##########################
brfn = RandomForestRegressor(criterion = 'mse', max_depth = 48, n_estimators = 24, n_jobs = 1)

params = {'n_estimators': 500, 'max_depth': 48, 'min_samples_split': 2,
          'learning_rate': 0.01, 'loss': 'ls', 'criterion': 'friedman_mse'}
gbm = GradientBoostingRegressor(**params)

gbm.fit(DBNtrain[DBNtrain.columns[:76]], DBNtrain['y'])
mse = mean_squared_error(DBNtrain[DBNtrain.columns[76]], gbm.predict(DBNtrain[DBNtrain.columns[:76]]))
yg_train_n = gbm.predict(DBNtrain[DBNtrain.columns[:76]])


brfn.fit(DBNtrain[DBNtrain.columns[:76]], DBNtrain['y'])
yhat_train_n = brfn.predict(DBNtrain[DBNtrain.columns[:76]])

rfR2n = 1 - (np.sum((DBNtrain[DBNtrain.columns[76]] - yhat_train_n)**2))/(np.sum((DBNtrain[DBNtrain.columns[76]] - np.mean(DBNtrain[DBNtrain.columns[76]]))**2))
print rfR2n

yhat_test_n = brfn.predict(DBTN)

eps = brfn.predict(DBN[DBN.columns[:76]]) - DBN['y']
eps5 = eps.ix[eps.index.month == 5]
plt.figure()
eps5.ix[eps5.index.year == 2017].plot(color = 'black')
plt.figure()
plotting.autocorrelation_plot(eps5)

ea = eps/DBN['y']
ea5 = ea.ix[ea.index.month == 5]
plt.figure()
ea5.ix[ea5.index.year == 2017].plot(color = 'black')
plt.figure()
plotting.autocorrelation_plot(ea5)


for h in range(24):
    eps5h = eps5.ix[eps5.index.hour == h]
    print eps5h.describe()
    plt.figure()
    plt.hist(eps5h, color = 'dimgrey', bins = 20)
    plt.title('error may hour = {}'.format(h))
    
for h in range(24):
    ea5h = ea5.ix[ea5.index.hour == h]
    print ea5h.describe()
    plt.figure()
    plt.hist(ea5h, color = 'dimgrey', bins = 20)
    plt.title('error may hour = {}'.format(h))

plt.figure()
scipy.stats.probplot(eps.values.ravel(), dist="norm", plot=plt)
plt.show()

model = AR(eps.values.ravel())
model_fit = model.fit()
window = model_fit.k_ar
coef = model_fit.params

yhat_test = brfn.predict(DBN[DBN.columns[:76]])

history = eps.values.ravel()[eps.values.ravel().size -window:]
history = [history[i] for i in range(history.size)]
predictions = list()
for t in range(DBN['y'].values.ravel().size):
	# persistence
	yhat = yhat_test[t]
	error = yhat - DBN['y'].values.ravel()[t]
	# predict error
	length = len(history)
	lag = [history[i] for i in range(length-window,length)]
	pred_error = coef[0]
	for d in range(window):
		pred_error += coef[d+1] * lag[window-d-1]
	# correct the prediction
	yhat = yhat - pred_error
	predictions.append(yhat)
	history.append(error)
	print('predicted=%f, expected=%f' % (yhat, DBN['y'].values.ravel()[t]))
# error
mse = mean_squared_error(DBN['y'].values.ravel(), predictions)
print('Test MSE: %.3f' % mse)
# plot predicted error
plt.figure()
plt.plot(DBN['y'].values.ravel())
plt.plot(predictions, color='red')

jun = []
history = eps5.values.ravel()[eps5.values.ravel().size - window:]
history = [history[i] for i in range(history.size)]
for x in range(yhat_test_n.size):
    yhat = yhat_test_n[x]
    length = len(history)
    lag = [history[i] for i in range(length - window,length)]
    pred_error = coef[0]
    for d in range(window):
    		pred_error += coef[d+1] * lag[window-d-1]
    jun.append(yhat - pred_error)
    
plt.figure()
plt.plot(yhat_test_n, color = 'grey')
plt.plot(np.array(jun), color = 'orange')
plt.figure()
plt.plot(np.array(jun) - yhat_test_n)


days = ['Lun', 'Mar', 'Mer', 'Gio', 'Ven', 'Sab', 'Dom']
for d in days:
    epsd = eps.ix[DBN[d] == 1]
    print epsd.describe()
    plt.figure()
    plt.plot(epsd.values.ravel()) 
    plt.title(d)       
    
EPS = []
for h in range(24):
    epsd = eps.ix[DBN['Mer'] == 1]
    eps5h = epsd.ix[epsd.index.hour == h]
    EPS.append(eps5h.values.ravel())    
    print eps5h.describe()
plt.figure()
plt.boxplot(EPS)
plt.title('Mer')

plt.figure()
plt.scatter(DBB['tmax'].resample('D').mean().values.ravel(), eps.resample('D').mean().values.ravel())
plt.figure()
plt.scatter(DBB['tmax'].ix[DBB.index.year == 2017].resample('D').mean().values.ravel(), DBB['y'].ix[DBB.index.year == 2017].resample('D').max().values.ravel())
np.corrcoef(DBB['tmax'].resample('D').mean().values.ravel(), DBB['y'].resample('D').mean().values.ravel())
plt.figure()
plt.scatter(eps.ix[DBN.index.year == 2017].resample('D').mean().values.ravel(), DBN['y'].ix[DBN.index.year == 2017].resample('D').max().values.ravel())


DFC = pd.DataFrame({'T': DBN['tmax'].ix[DBN.index.year == 2017].resample('D').mean().values.ravel(),
                    'T2': DBN['tmax'].ix[DBN.index.year == 2017].resample('D').mean().values.ravel()**2,
                    'T3': DBN['tmax'].ix[DBN.index.year == 2017].resample('D').mean().values.ravel()**3,
                    'Lun': DBN['Lun'].ix[DBN.index.year == 2017].resample('D').mean().values.ravel(),
                    'Mar': DBN['Mar'].ix[DBN.index.year == 2017].resample('D').mean().values.ravel(),
                    'Mer': DBN['Mer'].ix[DBN.index.year == 2017].resample('D').mean().values.ravel(),
                    'Gio': DBN['Gio'].ix[DBN.index.year == 2017].resample('D').mean().values.ravel(),
                    'Ven': DBN['Ven'].ix[DBN.index.year == 2017].resample('D').mean().values.ravel(),
                    'Sab': DBN['Sab'].ix[DBN.index.year == 2017].resample('D').mean().values.ravel(),
                    'Dom': DBN['Dom'].ix[DBN.index.year == 2017].resample('D').mean().values.ravel(),
                    'y': DBN['y'].ix[DBN.index.year == 2017].resample('D').max().values.ravel()})
DFC = DFC.dropna()

from sklearn.linear_model import LinearRegression

lm = LinearRegression(fit_intercept = True)
lm.fit(X = DFC[DFC.columns[:10]], y = DFC['y'])

print r2_score(DFC['y'], lm.predict(DFC[DFC.columns[:10]]))

DBTNC = pd.DataFrame({'T': DBTN['tmax'].resample('D').mean().values.ravel(),
                    'T2': DBTN['tmax'].resample('D').mean().values.ravel()**2,
                    'T3': DBTN['tmax'].resample('D').mean().values.ravel()**3,
                    'Lun': DBTN['Lun'].resample('D').mean().values.ravel(),
                    'Mar': DBTN['Mar'].resample('D').mean().values.ravel(),
                    'Mer': DBTN['Mer'].resample('D').mean().values.ravel(),
                    'Gio': DBTN['Gio'].resample('D').mean().values.ravel(),
                    'Ven': DBTN['Ven'].resample('D').mean().values.ravel(),
                    'Sab': DBTN['Sab'].resample('D').mean().values.ravel(),
                    'Dom': DBTN['Dom'].resample('D').mean().values.ravel()})

DBTNC = DBTNC.dropna()

diff = lm.predict(DBTNC)
print lm.coef_

DFF = OrderedDict()
dr = pd.date_range('2017-01-03', '2017-05-31', freq = 'D')
for i in dr:
    ll = []
    ll.append(DBB['tmax'].ix[DBB.index.date == i.date()].values[0] - DBB['tmax'].ix[DBB.index.date == (i.date() - datetime.timedelta(days = 365))].values[0])
    ll.append(DBB['tmax'].ix[DBB.index.date == i.date()].mean() - DBB['tmax'].ix[DBB.index.date == (i.date() - datetime.timedelta(days = 7))].mean())
    ll.append(DBB['y'].ix[DBB.index.date == i.date()].mean() - DBB['y'].ix[DBB.index.date == (i.date() - datetime.timedelta(days = 365))].mean())
    ll.append(DBB['y'].ix[DBB.index.date == i.date()].mean() - DBB['y'].ix[DBB.index.date == (i.date() - datetime.timedelta(days = 7))].mean())
    DFF[i.date()] = ll
    
DFF = pd.DataFrame.from_dict(DFF, orient = 'index')

start = time.time()
DFF = DifferentialDataset(DB, mi, "NORD")
print time.time() - start


mi = mi.append(mi2017)
mi = mi[['Tmedia', 'Tmin', 'Tmax','VENTOMEDIA','PIOGGIA']]
mi.columns = [[u'Tmin', u'Tmax', u'Tmedia', u'vento', u'pioggia']]
mi = mi.ix[mi.index.date < datetime.date(2017,3,30)]
mi = mi.append(mi2017)

DFFN = DifferentialDatasetForecast(DB, mi, "NORD", '2017-07-09','2017-07-14')

#DFF = DFF.ix[DFF.index.date > datetime.date(2016,1,7)]

col = ['blue', 'yellow', 'green', 'black', 'orange','red','purple', 'pink', 'indigo', 'orchid', 'salmon','purple']
plt.figure()
for m in range(1,13,1): 
    #plt.figure()
    plt.scatter(DFF['diff_tmax'].ix[DFF.index.month == m].values.ravel(),DFF['diff_sample'].ix[DFF.index.month == m].values.ravel(), color = col[m-1])
    print 'corr month {} = {}'.format(m, DFF[['diff_zona', 'diff_sample']].ix[DFF.index.month == m].corr())
    

drf = RandomForestRegressor(criterion = 'mse', max_depth = 48, n_estimators = 24, n_jobs = 1)

DFFT = DFF.ix[DFF.index.date < datetime.date(2017,4,1)]
Te = DFF.ix[DFF.index.date > datetime.date(2017,3,31)]
DFFT2 = DFFT.sample(frac = 1)

drf.fit(DFFT2[DFFT2.columns[:53]], DFFT2['diff_sample'])

print r2_score(DFFT2['diff_sample'], drf.predict(DFFT2[DFFT2.columns[:53]]))
print mean_squared_error(Te['diff_sample'], drf.predict(Te[Te.columns[:53]]))
print 1 - (np.sum((Te['diff_sample'] - drf.predict(Te[Te.columns[:53]])**2)))/(np.sum((Te['diff_sample'] - np.mean(Te['diff_sample']))**2))

plt.figure()
plt.plot(Te['diff_sample'].values.ravel())
plt.plot(drf.predict(Te[Te.columns[:53]]), color = 'green')

ds = drf.predict(DFFN[DFFN.columns[:53]])

plt.figure()
plt.plot(ds)

plt.figure()
plt.plot(DFF['diff_sample'].ix[DFF.index.month == 6].values.ravel())
plt.plot(DFF['diff_zona'].ix[DFF.index.month == 6].values.ravel())

D = DFF['diff_zona'].ix[DFF.index.month == 6].values.ravel() - DFF['diff_sample'].ix[DFF.index.month == 6].values.ravel()

DFF.to_hdf("C:/Users/utente/Documents/Sbilanciamento/differential_dataset.h5", "DFD")

plt.figure()
plt.plot(D)

DFS = Get_SampleAsTS_AtDay(DB, "NORD",'2017-01-01', '2017-05-31')

plt.figure()
plt.plot(nord['MO [MWh]'].ix[nord.index.date > datetime.date(2016,12,31)].values.ravel())
plt.plot(DFS.values.ravel())
plt.figure()
plt.scatter(nord['MO [MWh]'].ix[DFS.index].values.ravel()[:DFS.shape[0]], DFS.values.ravel())
oos = nord['MO [MWh]'].ix[DFS.index].values.ravel() - DFS.values.ravel()
oos[np.where(np.isnan((nord['MO [MWh]'].ix[DFS.index].values.ravel() - DFS.values.ravel())))[0]] = 0
plt.figure()
plt.plot(oos)

nd = nord['MO [MWh]'].ix[DFS.index].values.ravel()[:DFS.shape[0]]
nd[2018] = 0


from sklearn.linear_model import RANSACRegressor
lm = RANSACRegressor(LinearRegression())
lm.fit(DFS.values.ravel().reshape(-1,1), nd)

plt.figure()
plt.scatter(DFS.values.ravel(), nd)
plt.plot(lm.predict(np.arange(120).reshape(-1,1)))
#plt.plot(DFS.values.ravel()*lm.coef_ + lm.intercept_)
###### Try with error on sample
yhat_test_n = brf.predict(DBTN)

DTEf = CorrectionDatasetForecast(DBTN, yhat_test_n, DB, mi2017, "NORD")

yghat = gbm.predict(DTEf)
yehat = bre.predict(DTEf)


sad = Get_SampleAsTS_AtDay(DB, "NORD", '2017-07-02', '2017-07-03')
pc2 = percentageConsumption2(DB, "NORD", '2017-07-02', '2017-07-03')
yht = pd.DataFrame({'yhat': yhat_test_n}).set_index(DBTN.index)
est_sample = []
for i in pc2.index:
    yhtd = yht.ix[yht.index.date == pd.to_datetime(i).date()].values.ravel()
    res = (yhtd * pc2.ix[i].values).tolist()
    est_sample.extend(res)

ES = pd.DataFrame({"sam_hat": est_sample})
ES = ES.set_index(sad.index)

DFE = pd.DataFrame({"error": sad.values.ravel() - ES.values.ravel(), "yy": TE})

yghat = gbm.predict((sad.values.ravel() - ES.values.ravel()).reshape(sad.values.size,1))

pred_error = yhat_test_n[48:] -  yghat

plt.figure()
plt.plot(pred_error)
plt.figure()
plt.plot(yhat_test_n[48:], color = "red", marker = 'o')
plt.plot(yehat, color = "orange", marker = '*')
plt.plot(yghat, color = "orange", marker = '*')


ytn = pd.DataFrame.from_dict({'pred': yhat_test_n.tolist()})
ytn = ytn.set_index(DBTN.index)
ytn.plot(color = 'dimgrey')

DD = DifferentialDatasetForecast(DB, mi2017, "NORD", '2017-07-07','2017-07-11')

YH = DD['diff_sample'].ix[DD.index.date == datetime.date(2017,7,9)].values.ravel()*1.98 + yht.ix[yht.index.date == datetime.date(2017,7,12)].values.ravel()
YH = pd.DataFrame({'NORD':YH}).set_index(pd.date_range('2017-07-12', '2017-07-13', freq = 'H')[:24])
YH.to_excel("NORD_previsione_2017-07-12.xlsx")
## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##

bfr =  AdaBoostRegressor(RandomForestRegressor(criterion = 'mse', max_depth = 24, n_jobs = 1), n_estimators=3000)

brf = RandomForestRegressor(criterion = 'mse', max_depth = 48, n_estimators = 24, n_jobs = 1)

brf.fit(DBtrain[DBtrain.columns[:61]], DBtrain[DBtrain.columns[61]])
yhat_train = brf.predict(DBtrain[DBtrain.columns[:61]])

rfR2 = 1 - (np.sum((DBtrain[DBtrain.columns[61]] - yhat_train)**2))/(np.sum((DBtrain[DBtrain.columns[61]] - np.mean(DBtrain[DBtrain.columns[61]]))**2))
print rfR2

yhat_test = brf.predict(DBT)

##### Feature importance ####
importance = brf.feature_importances_
importance = pd.DataFrame(importance, index=DBtrain.columns[:61], columns=["Importance"])

importance["Std"] = np.std([tree.feature_importances_ for tree in brf.estimators_], axis=0)

x = range(importance.shape[0])
y = importance.ix[:, 0]
yerr = importance.ix[:, 1]

fig, ax = plt.subplots()
Ax = ax.bar(x, y, yerr=yerr, align="center")
Ax.set_xticklabels(DBtrain.columns)
####################################


plt.figure()
plt.plot(yhat_test, marker = '8', color = 'red')
plt.figure()
plt.plot(yhat_test, marker = '8', color = 'magenta')
plt.title('AdaBoost + RF')
plt.figure()
plt.plot(DBB['y'].values.ravel())


DBT['holiday'].ix[datetime.date(2017,4,16)]

pd.DataFrame.from_dict({'previsione': yhat_test[-24:].tolist()}).to_excel('C:/Users/utente/Documents/previsione_2017-05-17.xlsx')

Y = DBB['y'].astype(float)#.values.ravel()
U = np.repeat(0.0, Y.size)
U = U + Y
Y = []
for i in range(DBB.shape[0]):
    if DBB['y'].ix[i].size == 1:
        Y.append(float(DBB['y'].ix[i]))
    else:
        Y.append(float(DBB['y'].ix[i].sum()))

plt.figure()
plt.plot(Y[2831:3576])
plt.axvline(x = 2831, color = 'black',  linewidth=2.0)
plt.axvline(x = 3575, color = 'black',  linewidth=2.0)

zone = ["NORD", "CNOR", "CSUD", "SUD", "SICI", "SARD"]
for z in zone:
    pc2 = percentageConsumption2(DB, z, '2017-01-01', '2017-05-31')
    plt.figure()
    pc2.plot()

pc2 = percentageConsumption2(DB, "CNOR", '2016-01-01', '2017-05-31')
pc = percentageConsumption(DB, "CNOR", '2017-01-01', '2017-05-31')
   
rem = LearnPodwiseModels(DB, rc, "SUD", end = '2017-05-31')

sos = pd.read_excel("C:/Users/utente/Documents/Sbilanciamento/sos_aggregati.xlsx")

sos.to_hdf("C:/Users/utente/Documents/Sbilanciamento/sos_aggregati.h5", "sos")

H = pd.read_excel("C:/Users/utente/Documents/Sbilanciamento/CRPP2016_artigianale.xlsx")
H.to_hdf("C:/Users/utente/Documents/Sbilanciamento/CRPP2016_artigianale.h5", "CRPP2016")
    
####################################################################################################    
sample = Get_SampleAsTS(DB, "NORD")
oos = Get_OutOfSample(nord, DB, "NORD")

sample = sample.ix[sample.index.year == 2017]
oos = oos.ix[oos.index.year == 2017]
oos = oos.ix[sample.index]

bar = []
std_bar_sam = []
std_bar_oos = []

plt.figure()
for h in range(24):
    bar.append(np.corrcoef(sample.ix[sample.index.hour == h].values.ravel(), oos.ix[oos.index.hour == h].values.ravel())[0,1])
    std_bar_sam.append(sample.ix[sample.index.hour == h].std())
    std_bar_oos.append(oos.ix[oos.index.hour == h].std())
    
plt.figure()
plt.bar(np.arange(24),np.array(bar))
plt.figure()
plt.bar(np.arange(24),np.array(std_bar_sam), color = 'green')
plt.figure()
plt.bar(np.arange(24),np.array(std_bar_oos), color = 'yellow')

for h in range(24):
    plt.figure()    
    plt.scatter(sample.ix[sample.index.hour == h].values.ravel() ,oos.ix[oos.index.hour == h].values.ravel())
    plt.title(str(h))
####################################################################################################
    
mi2017 = pd.read_excel('C:/Users/utente/Documents/meteo/Milano.xlsx')
mi2017 = mi2017[["Tmin", "Tmax", "Tmedia", "vento", "pioggia"]]

DBB = pd.read_hdf('C:/Users/utente/Documents/Sbilanciamento/nord.h5')

DB2 = DB.drop_duplicates(['POD', 'Area', 'Giorno'], keep = 'last')    
    
#DBB = MakeExtendedDatasetWithSampleCurve2(nord, DB2, mi, "NORD", '2017-01-01','2017-05-31')
#    
#DBB[['ysample', 'yoos']].plot()    
#DBB[['ysample', 'yoos']].corr()    
#    
#DBtrain = DBB.sample(frac = 1)
#TStrain = TrainSample.sample(frac = 1)
#
#brs = RandomForestRegressor(criterion = 'mse', max_depth = 48, n_estimators = 24, n_jobs = 1)
#
#brs.fit(TStrain[TStrain.columns[:79]], TStrain['ysample'])
#yhat_train = brs.predict(TStrain[TStrain.columns[:79]])
#
##rfR2 = 1 - (np.sum((DBtrain[DBtrain.columns[61]] - yhat_train)**2))/(np.sum((DBtrain[DBtrain.columns[61]] - np.mean(DBtrain[DBtrain.columns[61]]))**2))
##print rfR2
#
#print r2_score(TrainSample['ysample'], brs.predict(TrainSample[TrainSample.columns[:79]]))
#
#bro = RandomForestRegressor(criterion = 'mse', max_depth = 48, n_estimators = 24, n_jobs = 1)
#
#bro.fit(DBtrain[DBtrain.columns[:81]], DBtrain['yoos'])
#yhat_train = bro.predict(DBtrain[DBtrain.columns[:81]])
#
##rfR2 = 1 - (np.sum((DBtrain[DBtrain.columns[61]] - yhat_train)**2))/(np.sum((DBtrain[DBtrain.columns[61]] - np.mean(DBtrain[DBtrain.columns[61]]))**2))
##print rfR2
#
#print r2_score(DBB['yoos'], bro.predict(DBB[DBB.columns[:81]]))
#
#plt.figure()
#plt.plot(DBB['yoos'].values.ravel())
#plt.plot(bro.predict(DBB[DBB.columns[:80]]))
#   
##DFT = MakeForecastDataset2(DB2, mi2017, "NORD", '2017-06-01')   
#   
#yos = bro.predict(TT)   
#
#Nhat = TT['ysample'] + yos   
#
#plt.figure()
#plt.plot(Nhat)
#   
#yf_sample = brs.predict(TF[TF.columns[:79]])   
#
#pd.DataFrame({'campione': yf_sample}).set_index(TF.index).to_excel('campione_previsto_per_2017-07-15.xlsx')
#
#
#TF2 = TF
#TF2['ysample'] = np.concatenate((yf_sample[:24], p(np.arange(24))))
#TF2['ysample'] = yf_sample
#yf_oos = bro.predict(TF2)   
#
#yf = yf_sample + yf_oos
#yf = pd.DataFrame({'prediction': yf}).set_index(TF.index)
#
#yf.ix[yf.index.date > datetime.date(2017,7,14)].plot()
#   
#DFT['ysample'] = trues    
#   
#DBB2 = DBB[DBB.columns[:81]]
#   
#DBB = DBB2.append(DFT.dropna())   
#
#
#brs = RandomForestRegressor(criterion = 'mse', max_depth = 48, n_estimators = 24, n_jobs = 1)
#
#brs.fit(DBB[DBB.columns[:80]], DBB['ysample'])
#yhat_train = brs.predict(DBB[DBB.columns[:80]])
#
#   
#tsample = brs.predict(DFT[DFT.columns[:80]])   
#toos= bro.predict(DFT)   
#
#trues = Get_SampleAsTS(DB, "NORD")   
#trues = trues.ix[DFT.index]
#
#plt.figure()
#plt.plot(trues.values.ravel())
#plt.plot(tsample, color = 'red')
#
#ty = tsample + toos
#
#plt.figure()
#plt.plot(ty)
#    
#datetime.date(2017,6,3) in pd.to_datetime(DB["Giorno"].values.ravel().tolist())

#####& 21/07/2017 ##################################################################################    
#for h in range(24):
#    print 'mean error hour {}: {}'.format(h, eo.ix[eo.index.hour == h].mean().values)
#    print 'median error hour {}: {}'.format(h, eo.ix[eo.index.hour == h].median())
#    print 'std error hour {}: {}'.format(h, eo.ix[eo.index.hour == h].std())
##    plt.figure()
##    eo.ix[eo.index.hour == h].plot(kind = 'hist')    
#    plt.figure()
#    eo.ix[eo.index.hour == h].boxplot()
#    
#feo = fourierExtrapolation(eo.values.ravel(), 50)
#
#plt.figure()
#plt.plot(eo.values.ravel(),color = 'black')
#plt.plot(feo, color = 'indigo')
#
#eo['Giorno'] = np.array([eo.index.date[i].weekday() for i in range(eo.shape[0])])
#
#diz = OrderedDict()
#for i in list(set(eo.index.date)):
#    if eo['Giorno'].ix[i] < 5:
#        diz[i] = eo['err'].ix[eo.index.date == i].values.ravel().tolist()
#
#diz = pd.DataFrame.from_dict(diz, orient = 'index')
#
#diz.to_excel('C:/Users/utente/Documents/Sbilanciamento/errore_feriali_OOS.xlsx')
#
#plt.figure()
#diz.plot(kind = 'box')
#
#
#diz.std(axis = 0)
#
#diz2 = OrderedDict()
#for i in list(set(eo.index.date)):
#    if eo['Giorno'].ix[i] >= 5:
#        diz2[i] = eo['err'].ix[eo.index.date == i].values.ravel().tolist()
#
#diz2 = pd.DataFrame.from_dict(diz2, orient = 'index')
#
#diz2.to_excel('C:/Users/utente/Documents/Sbilanciamento/errore_festivi_OOS.xlsx')
#
#
#plt.figure()
#diz2.plot(kind = 'box')
#
#
#diz2.std(axis = 0)
#
#yoc2 = yf_oos2[-24:]
#yoc = []
#for h in range(24):
#     if diz2.median().values.ravel()[h] > 0:
#         yoc.append(yoc2[h] - diz2.median().values.ravel()[h])
#     else:
#        yoc.append(yoc2[h] + diz2.median().values.ravel()[h])
#
#plt.figure()
#plt.plot(yoc2 - diz2.median().values.ravel(), color = 'cadetblue')
####################################################################################################
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)

gp.fit(TStrain[TStrain.columns[:79]], TStrain['ysample'])

# Make the prediction on the meshed x-axis (ask for MSE as well)
y_pred, sigma = gp.predict(TF[TF.columns[:79]], return_std=True)

####################################################################################################
jun = pd.read_excel('C:/Users/utente/Documents/Sbilanciamento/Consuntivo_giu.xlsx')
sam = Get_SampleAsTS(DB2, 'NORD')
samj = sam.ix[sam.index.month == 6]
oosj = jun['NORD'].values.ravel() - samj.values.ravel()

oosj = pd.DataFrame({'oos': oosj}).set_index(samj.index)
oos = Get_OutOfSample(nord, DB2, 'NORD')

oos.columns = ['oos']

oos = oos.append(oosj)

oos.to_hdf('C:/Users/utente/Documents/Sbilanciamento/storico_oos_' + zona + '.h5', 'oos_' + zona)

oos = pd.read_hdf('C:/Users/utente/Documents/Sbilanciamento/storico_oos_' + zona + '.h5')


plt.figure()
plt.plot(oosj)


DDoos = DifferentialDatasetOOS(oos, mi, 'NORD')
DDoos = DDoos.dropna()

plt.figure()
plt.scatter(DDoos['diff_tmax'].values.ravel(), DDoos['diff_sample'].values.ravel())

np.corrcoef(DDoos['diff_tmax'].values.ravel(), DDoos['diff_sample'].values.ravel())

oos.resample('D').mean()
mi7 = mi['Tmax'].ix[mi.index.date > datetime.date(2016,12,31)]
mi7 = mi7.ix[mi7.index.date < datetime.date(2017,7,1)]
mi7.columns = ['Tmax']

plt.figure()
plt.scatter(mi7.values.ravel(),oos.resample('D').mean().values.ravel(), color = 'green')

dow = map(lambda date: date.weekday(), mi7.index.date)
dow2 = map(lambda date: date.weekday(), oos.index.date)

plt.figure()
plt.scatter(mi7.ix[np.where(np.array(dow) < 5)[0]].values.ravel(),oos.ix[np.where(np.array(dow2) < 5)[0]].resample('D').mean().dropna().values.ravel())

mi7nh = mi7.ix[np.where(np.array(dow) < 5)[0]]
oosnh = oos.ix[np.where(np.array(dow2) < 5)[0]].resample('D').mean().dropna()

mi_1 = mi7nh.ix[np.where(oosnh.values.ravel() < 40)[0]]
oos_1 = oosnh.ix[np.where(oosnh.values.ravel() < 40)[0]]
mi_2 = mi_1.ix[np.where(oos_1.values.ravel() > 24)[0]]
oos_2 = oos_1.ix[np.where(oos_1.values.ravel() > 24)[0]]


plt.figure()
plt.scatter(mi_2.values.ravel(), oos_2.values.ravel(), color = 'salmon')


dat = pd.DataFrame({'y': oos_2.values.ravel(), 'T': mi_2})
dat = dat.sort_values(by = 'y')


from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression

rf = RandomForestRegressor(criterion = 'mse', max_depth = 1000, n_estimators = 5000, n_jobs = 1)
rf.fit(dat['T'].reshape(-1,1), dat['y'])

model = make_pipeline(PolynomialFeatures(3), LinearRegression(fit_intercept = True))
model.fit(dat['T'].reshape(-1,1), dat['y'])

y_plot = model.predict(dat['T'].reshape(-1,1))
y_plot = rf.predict(dat['T'].reshape(-1,1))

print r2_score(dat['y'], y_plot)
print mean_squared_error(dat['y'], y_plot)


plt.figure()
plt.scatter(mi_2.values.ravel(), oos_2.values.ravel(), color = 'salmon')
plt.plot(np.arange(0,35, 0.05), rf.predict(np.arange(0,35, 0.05).reshape(-1,1)), color = 'navy')

residuals = dat['y'] - y_plot

plt.figure()
plt.hist(residuals, bins = 20)

plt.figure()
plt.scatter(y_plot, residuals)

joblib.dump(model, 'C:/Users/utente/Documents/Sbilanciamento/Tmax_load_OOS_not_holiday.pkl')

np.mean(residuals)
np.std(residuals)


mi_h = mi7.ix[np.where(oos.resample('D').mean().values.ravel() < 24)[0]]
oos_h = oos.resample('D').mean().ix[np.where(oos.resample('D').mean().values.ravel() < 24)[0]]


plt.figure()
plt.scatter(mi_h.values.ravel(), oos_h.values.ravel(), color = 'grey')

np.corrcoef(mi_h.values.ravel(), oos_h.values.ravel())


dath = pd.DataFrame({'y': oos_h.values.ravel(), 'T': mi_h})
dath = dath.sort_values(by = 'y')

modelh = make_pipeline(PolynomialFeatures(3), LinearRegression(fit_intercept = True))
modelh.fit(dath['T'].reshape(-1,1), dath['y'])

y_ploth = modelh.predict(dath['T'].reshape(-1,1))
print r2_score(dath['y'], y_ploth)

plt.figure()
plt.scatter(mi_h.values.ravel(), oos_h.values.ravel(), color = 'grey')
plt.plot(np.arange(0,35, 0.05), modelh.predict(np.arange(0,35, 0.05).reshape(-1,1)), color = 'black')

joblib.dump(modelh, 'C:/Users/utente/Documents/Sbilanciamento/Tmax_load_OOS_holiday.pkl')

nord.columns = ['MO [MWh]']
nn = Get_Z(nord, 'NORD')
dft = Get_DependencyWithTemperature(nn, mi)

vals = dft['y'].values.ravel().tolist()
X = np.repeat(0, len(vals))
for j in range(len(vals)):
    if str(type(vals[j])) == 'pandas.core.series.Series':
        X[j] = sum(vals[j].values.ravel())
    else:
        X[j] = vals[j]

plt.figure()
plt.scatter(dft['tmax'].values.ravel(),  dft['y'].diff().values.ravel(), color = 'lime')

plt.figure()
dft['y'].diff().hist(bins = 20)
dft['y'].mean()
dft['y'].std()

dftt = dft.ix[dft.index.year == 2016]
dfte = dft.ix[dft.index.year == 2017]


rf = RandomForestRegressor(criterion = 'mse', max_depth = 1000, n_estimators = 5000, n_jobs = 1)
rf.fit(dftt[dftt.columns[:-1]], dftt['y'])


y_plot = rf.predict(dfte[dfte.columns[:-1]])

print r2_score(dfte['y'], y_plot)
print mean_squared_error(dfte['y'], y_plot)

plt.figure()
plt.plot(y_plot)

plt.figure()
plt.plot(dft['y'].diff(),color = 'violet')

DB2 = dt.drop_duplicates(['POD', 'Area', 'Giorno'], keep = 'last')    

crpp = pd.read_excel('H:/Energy Management/02. EDM/01. MISURE/4. CRPP/2017/03-2017/_All_CRPP_03_2017.xlsx')
podlist = list(set(crpp['POD'].ix[crpp['ZONA'] == zona]))

dts = pd.date_range('2017-03-01', '2017-03-31', freq = 'D')
sos = pd.read_hdf("C:/Users/utente/Documents/Sbilanciamento/sos_elaborati_finiti.h5")
pdo = pd.read_hdf("C:/Users/utente/Documents/Sbilanciamento/DB_misure.h5")
    
sos['Giorno'] = pd.to_datetime(sos['Giorno'].values.ravel().tolist()).date   
pdo['Giorno'] = pd.to_datetime(pdo['Giorno'].values.ravel().tolist()).date
    
db = DB2.ix[DB2['Area'] == zona]
PRED = OrderedDict()        
    
missing_counter = 0
missing_pod = []
for d in dts:
    ricd = GetRicDate(d).date()
    #podlist = list(set(db['POD'].ix[db['Giorno'] == max(db['Giorno'])]))
    for p in podlist:
        print 'avanzamento: {}'.format(podlist.index(p)/len(podlist))
        pdop = pdo.ix[pdo['POD'] == p]
        sosp = sos.ix[sos['Pod'] == p]
        if pdop.shape[0] > 0:
            #print 'taken from PDO'
            pdopd = pdop.ix[pdop['Giorno'] == ricd]
            if pdopd.shape[0] > 0:
                pred = pdopd[pdopd.columns[4:]].values.ravel()
                PRED[p] = pred.tolist()
            else:
               #print 'not found'                    
               missing_counter += 1
               missing_pod.append((p,d))
        else:
           #print 'taken from SOS'                
           sospd = sosp.ix[sosp['Giorno'] == ricd]
           if sospd.shape[0] > 0:
               pred = sospd[sospd.columns[2:]].values.ravel()
               PRED[p] = pred.tolist()
           else:
               #print 'not found'                    
               missing_counter += 1
               missing_pod.append((p,d))
            
        
print '#pod not found: {}'.format(missing_counter)
    
PRED = pd.DataFrame.from_dict(PRED, orient = 'index')    
PRED.to_excel('campione_da_storico_PDO.xlsx')
    

Sam = Get_SampleAsTS(dt, 'NORD')
Sam = Sam.ix[Sam.index.year == 2017]
Samd = Sam.resample('D').mean()
Samdd = Samd.diff()

Err = Samdd.ix[1:].values.ravel()/Samd.ix[1:].values.ravel()

dow = map(lambda date: date.weekday(), Samd.ix[1:].index)

plt.figure()
plt.scatter(Samd.ix[1:].values.ravel()[np.where(np.array(dow) < 5)[0]], Err[np.where(np.array(dow) < 5)[0]])
plt.scatter(Samd.ix[1:].values.ravel()[np.where(np.array(dow) == 5)[0]], Err[np.where(np.array(dow) == 5)[0]], color = 'red')
plt.scatter(Samd.ix[1:].values.ravel()[np.where(np.array(dow) == 6)[0]], Err[np.where(np.array(dow) == 6)[0]], color = 'green')

##### LOCAL TREND: Fri on Sat and Sun
plt.figure()
plt.scatter(Samd.ix[1:].values.ravel()[np.where(np.array(dow) == 4)[0]], Samd.ix[1:].values.ravel()[np.where(np.array(dow) == 5)[0]], color = 'blue')
plt.scatter(Samd.ix[1:].values.ravel()[np.where(np.array(dow) == 4)[0]], Samd.ix[1:].values.ravel()[np.where(np.array(dow) == 6)[0]], color = 'red', marker = '*')
plt.figure()
plt.scatter(Samd.ix[1:].values.ravel()[np.where(np.array(dow) == 5)[0]], Samd.ix[1:].values.ravel()[np.where(np.array(dow) == 6)[0]], color = 'black')
plt.figure()
plt.scatter(Samd.values.ravel()[np.where(np.array(dow) == 4)[0]], Samd.values.ravel()[np.where(np.array(dow) == 0)[0][:-1]], color = 'purple')
plt.figure()
plt.scatter(Samd.values.ravel()[np.where(np.array(dow) == 0)[0]], Samd.values.ravel()[np.where(np.array(dow) == 1)[0]], color = 'steelblue')


#### LOCAL TREND: Mon on Wed
hbar = []
sdow =  map(lambda date: date.weekday(), Sam.index)
for h in range(24):
    hl = Sam.ix[np.where(np.array(sdow) == 0)[0][:-24]]
    hhll = hl.ix[hl.index.hour == h]
    hm = Sam.ix[np.where(np.array(sdow) == 2)[0]]
    hhmm = hm.ix[hm.index.hour == h]
    hbar.append(np.corrcoef(hhll.values.ravel(),hhmm.values.ravel())[0,1])

plt.figure()
plt.bar(np.arange(24),hbar,color = 'skyblue')

pbar = []
vpbar = []
BXLM = OrderedDict() 
LonM = OrderedDict() 
for h in range(24):
    hl = Sam.ix[np.where(np.array(sdow) == 0)[0][:-24]]
    hhll = hl.ix[hl.index.hour == h]
    hm = Sam.ix[np.where(np.array(sdow) == 2)[0]]
    hhmm = hm.ix[hm.index.hour == h]
    P = (hhmm.values.ravel() - hhll.values.ravel())/hhll.values.ravel()
    BXLM[str(h)] = P
    pbar.append(np.mean(P))
    vpbar.append(np.std(P))
    if h < 7:
        print np.mean(P)
        S = (-(1 + np.mean(P))*hhll.values.ravel() + hhmm.values.ravel())/((1 + np.mean(P))*hhll.values.ravel())
        LonM[str(h)] = S


plt.figure()
plt.bar(np.arange(24),pbar,color = 'skyblue')
plt.figure()
plt.bar(np.arange(24),vpbar,color = 'indigo')

BXLM = pd.DataFrame.from_dict(BXLM, orient = 'columns')
plt.figure()
BXLM.boxplot()
LonM = pd.DataFrame.from_dict(LonM, orient = 'columns')
plt.figure()
LonM.boxplot()

#### Friday to Saturday
pbar = []
vpbar = []
BXVS = OrderedDict() 
VonS = OrderedDict() 
for h in range(24):
    hl = Sam.ix[np.where(np.array(sdow) == 4)[0]]
    hhll = hl.ix[hl.index.hour == h]
    hm = Sam.ix[np.where(np.array(sdow) == 5)[0]]
    hhmm = hm.ix[hm.index.hour == h]
    P = (hhll.values.ravel() - hhmm.values.ravel())/hhll.values.ravel()
    BXVS[str(h)] = P
    pbar.append(np.mean(P))
    vpbar.append(np.std(P))
    S = ((-1 + np.mean(P))*hhll.values.ravel() + hhmm.values.ravel())/hhmm.values.ravel()
    VonS[str(h)] = S

plt.figure()
plt.bar(np.arange(24),pbar,color = 'skyblue')
plt.figure()
plt.bar(np.arange(24),vpbar,color = 'indigo')

BXVS = pd.DataFrame.from_dict(BXVS, orient = 'columns')
plt.figure()
BXVS.boxplot()
VonS = pd.DataFrame.from_dict(VonS, orient = 'columns')
plt.figure()
VonS.boxplot()
plt.axhline(y = 0.15, color = 'black')
plt.axhline(y = -0.15, color = 'black')
plt.axhline(y = -0.30, color = 'red')
plt.axhline(y = 0.30, color = 'red')


giorni = ['Lun','Mar','Mer','Gio','Ven','Sab','Dom']
dow2 = map(lambda date: date.weekday(), Samd.index)
for day in range(7):
    plt.figure()
    plotting.autocorrelation_plot(Samd.ix[np.where(np.array(dow2) == day)[0]])
    plt.title('autocorrelation {}'.format(giorni[day]))


plt.figure()
plt.scatter(Samdd.ix[1:].values.ravel()[np.where(np.array(dow) == 0)[0]], Err[np.where(np.array(dow) == 0)[0]], color = 'yellow')
plt.figure()
plt.scatter(Samdd.ix[1:].values.ravel()[np.where(np.array(dow) < 5)[0]], Err[np.where(np.array(dow) < 5)[0]], color = 'pink')
plt.figure()
plt.scatter(Samdd.ix[1:].values.ravel()[np.where(np.array(dow) == 5)[0]], Err[np.where(np.array(dow) == 5)[0]], color = 'grey')
plt.figure()
plt.scatter(Samdd.ix[1:].values.ravel()[np.where(np.array(dow) == 6)[0]], Err[np.where(np.array(dow) == 6)[0]], color = 'black')

fer = Samdd.ix[1:].ix[np.where(np.array(dow) > 0)[0]]
fErr = Err[np.where(np.array(dow) > 0)[0]]
hol = map(lambda date: AddHolidaysDate(date), fer.index)

plt.figure()
plt.scatter(fer.values.ravel(), fErr, color = 'steelblue')

Fer = fer.ix[np.where(np.array(hol) == 0)[0]]
FErr = fErr[np.where(np.array(hol) == 0)[0]]

plt.figure()
plt.scatter(Fer.values.ravel(), FErr, color = 'darkslateblue')
plt.figure()
plt.hist(FErr, bins = 20)

scipy.stats.mstats.mquantiles(FErr, prob = [0.025, 0.975])

plt.figure()
plt.scatter(Samdd.ix[1:].values.ravel(), np.exp(Err), color = 'midnightblue')

mi7 = pd.read_excel('C:/Users/utente/Documents/PUN/Milano 2017.xlsx')
mi7 = mi7.set_index(pd.date_range('2017-01-01', '2017-10-31', freq = 'D'))
mi7 = mi7[['Tmin', 'Tmax','Tmedia', 'VENTOMEDIA', 'PIOGGIA']]
mi7 = mi7.set_index(np.array(list(map(lambda date: date.date(), pd.to_datetime(mi7.index)))))
mi7.columns = ["Tmin", "Tmax", "Tmedia", "vento", "pioggia"]
mi2017 = pd.read_excel('C:/Users/utente/Documents/meteo/Milano.xlsx')
mi2017 = mi2017[["Tmin", "Tmax", "Tmedia", "vento", "pioggia"]]
mi = mi7.append(mi2017)
mi['index'] = mi.index.date
mi = mi.drop_duplicates(subset = 'index', keep = 'first')
mi = mi.set_index(mi['index'])

mi6 = pd.read_excel('C:/Users/utente/Documents/PUN/Milano 2016.xlsx')
mi6 = mi6.drop(59)
mi6 = mi6.ix[:365].set_index(pd.date_range('2017-01-01', '2017-12-31', freq = 'D'))
mi6 = mi6[['Tmin', 'Tmax','Tmedia', 'VENTOMEDIA', 'PIOGGIA']]
mi6 = mi6.set_index(np.array(list(map(lambda date: date.date(), pd.to_datetime(mi6.index)))))
mi6.columns = ["Tmin", "Tmax", "Tmedia", "vento", "pioggia"]

mi = mi7.append(mi6.ix[mi6.index.month > 9])
mi = mi.set_index(pd.date_range('2017-01-01', '2017-12-31', freq = 'D').date)

Sample = Get_SampleAsTS(db, 'NORD')
Oos = Get_OutOfSample(nord,db,'NORD')

RIC = RIC[RIC.columns[2:]].sum(axis = 1)/1000
RIC = pd.DataFrame({'x':RIC.values.ravel()}).set_index(pd.date_range('2017-01-01', '2018-12-31', freq = 'H')[:8760])
df2 = RIC

dec = statsmodels.api.tsa.seasonal_decompose(RIC, freq = 24)
dec.plot()
plt.figure()
plt.plot(dec.resid)
plt.figure()
dec.resid.fillna(0).resample('D').sum().plot()

DWT = Get_MeanDependencyWithTemperature(df2, mi6)

DWToos = Get_MeanDependencyWithTemperature(Oos, mi)

dl = 'Dom'
plt.figure()
plt.scatter(DWT['tmax'].ix[DWT[dl] == 1],DWT['y'].ix[DWT[dl] == 1], color = 'magenta')
plt.title(dl)
plt.figure()
plt.scatter(DWT['tmax'],DWT['y'], color = 'indigo')
plt.scatter( DWT['tmax'], rf.predict(DWT[DWT.columns[:27]]), color = 'salmon', marker = '*' )

DWTt = DWT.sample(frac = 1)

rf = RandomForestRegressor(criterion = 'mse', n_estimators = 3000, n_jobs = 1)
rf.fit(DWTt[DWTt.columns[:27]], DWTt['y'])

print r2_score(DWTt['y'], rf.predict(DWTt[DWTt.columns[:27]]))
print mean_squared_error(DWTt['y'], rf.predict(DWTt[DWTt.columns[:27]]))

DWToost = DWToos.sample(frac = 1)

rf = RandomForestRegressor(criterion = 'mse', n_estimators = 3000, n_jobs = 1)
rf.fit(DWToost[DWToost.columns[:27]], DWToost['y'])

print r2_score(DWToost['y'], rf.predict(DWToost[DWToost.columns[:27]]))
print mean_squared_error(DWToost['y'], rf.predict(DWToost[DWToost.columns[:27]]))


ES = Estimate_Sbil(db, mi, 'NORD')
PS = Predict_Sbil(db, mi, 'NORD', datetime.date(2017,9,13))

rf = RandomForestRegressor(criterion = 'mse', n_estimators = 3000, n_jobs = 1)
rf.fit(ES[ES.columns[:28]], ES['y'])

print r2_score(ES['y'], rf.predict(ES[ES.columns[:28]]))
print mean_squared_error(ES['y'], rf.predict(ES[ES.columns[:28]]))

feature_importance = rf.feature_importances_
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5
plt.figure()
plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.yticks(pos, ES.columns[sorted_idx])
plt.xlabel('Relative Importance')
plt.title('Variable Importance')
plt.show()

ESt = ES.ix[ES.index.date < datetime.date(2017,8,8)]
ESte = ES.ix[ES.index.date > datetime.date(2017,8,7)]

ESt = ESt.sample(frac = 1)

rf = RandomForestRegressor(criterion = 'mse', n_estimators = 3000, n_jobs = 1)
rf.fit(ESt[ESt.columns[:28]], ESt['y'])

print r2_score(ESt['y'], rf.predict(ESt[ESt.columns[:28]]))
print mean_squared_error(ESt['y'], rf.predict(ESt[ESt.columns[:28]]))

print r2_score(ESte['y'], rf.predict(ESte[ESte.columns[:28]]))

plt.figure()
plt.plot(ESte.index, ESte['y'].values.ravel(), marker = 'o')
plt.plot(ESte.index, rf.predict(ESte[ESte.columns[:28]]), marker = '*', color = 'coral')
print 'Sbilanciamento previsto: {}'.format(rf.predict(PS[PS.columns[:28]]))

from sklearn.model_selection import train_test_split

res = pd.DataFrame()
for b in range(100):
    X_train, X_test, y_train, y_test = train_test_split(ES[ES.columns[:28]], ES['y'], test_size=0.33)
    rf2 = RandomForestRegressor(criterion = 'mse', n_estimators = 3000, n_jobs = 1)
    rf2.fit(X_train, y_train)
    r2t = r2_score(y_train, rf2.predict(X_train))
    mset = mean_squared_error(y_train, rf.predict(X_train))
    r2te = r2_score(y_test, rf.predict(X_test))
    res = res.append(pd.DataFrame({'r2_train': [r2t], 'mse_train': [mset], 'r2_test':[r2te]}))
    
    
A2A = rical[list(set(rical.columns[2:]).intersection(set(a2a)))]
A2A = A2A.set_index(rical.index)
Sample = Sample.ix[Sample.index.year == 2017]
SA = A2A.sum(axis = 1)
SA = SA.ix[:5807]

np.corrcoef(Sample.values.ravel(), SA.values.ravel()/1000)

########################## LOCAL  TREND ############################################################ 

MU = Sample.resample('D').apply(np.trapz)
MU.plot()
NU = Oos.resample('D').apply(np.trapz)
NU.plot(color = 'red')
plt.figure()
plotting.autocorrelation_plot(MU)
plt.figure()
plotting.autocorrelation_plot(NU)

###############################################################################
def BestApproximatingPolynomial(vec):
    best_d = 0
    P1 = np.mean(vec)
    qerr1 = np.mean((vec - P1)**2)
    best_error = qerr1
    print('0-degree error: {}'.format(qerr1))
    for d in range(1, 11, 1):
        BP = np.poly1d(np.polyfit(np.linspace(1,vec.size,vec.size), vec, d))
        qerr = np.mean((vec - BP(np.linspace(1,vec.size,vec.size)))**2)
        print(str(d)+'-degree error: {}'.format(qerr))
        if qerr < best_error:
            best_d = d
            best_error = qerr
    return best_d
###############################################################################
def exponential_smoothing(series, alpha):
    result = [series[0]] # first value is same as series
    for n in range(1, len(series)):
        result.append(alpha * series[n] + (1 - alpha) * result[n-1])
    return result
###############################################################################
def deseasonalise(hts): ## TS with temporal timestamp
    dix = OrderedDict()
    for h in range(24):
        dix[h] = hts.ix[hts.index.hour == h].values.ravel()
    return pd.DataFrame.from_dict(dix, orient = 'columns')
###############################################################################
def deseasonalised(hts, seas):
    res = []
    for i in range(0, hts.shape[0]-24, 24):
        day = hts[i:i+24].values.ravel()
        res.append(day - seas)
    day = hts[hts.shape[0] - 24:].values.ravel()
    res.append(day - seas)
    return np.array(res).ravel()
###############################################################################
    
BP = np.poly1d(np.polyfit(np.linspace(1, MU.values.ravel().size, MU.values.ravel().size), MU.values.ravel(), 5))
a_pol = BP(np.linspace(1, MU.values.ravel().size, MU.values.ravel().size))

plt.figure()
plt.plot(MU.values.ravel())
plt.plot(a_pol, color = 'black')
 
BP2 = np.poly1d(np.polyfit(np.linspace(1, NU.values.ravel().size, NU.values.ravel().size), NU.values.ravel(), 5))
a_pol2 = BP2(np.linspace(1, NU.values.ravel().size, NU.values.ravel().size))

plt.figure()
plt.plot(NU.values.ravel(), color = 'red')
plt.plot(a_pol2, color = 'black')
       
MU['dow'] = map(lambda date: date.weekday(), MU.index) 
NU['dow'] = map(lambda date: date.weekday(), NU.index) 

dow = 6

plt.figure()
plt.plot(MU['MO [MWh]'].ix[MU['dow'] == dow])
plt.figure()
plotting.autocorrelation_plot(MU['MO [MWh]'].ix[MU['dow'] == dow])

plt.figure()
plt.plot(NU['MO [MWh]'].ix[NU['dow'] == dow],color = 'red')
plt.figure()
plotting.autocorrelation_plot(NU['MO [MWh]'].ix[NU['dow'] == dow], color = 'red')