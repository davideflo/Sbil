# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 11:38:23 2017

@author: utente

Sbilanciamento - 5th file - slims the 3rd
"""

from __future__ import division
import pandas as pd
import numpy as np
import calendar
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from collections import OrderedDict
import datetime
#import time
import os
from sklearn.externals import joblib
from sklearn.linear_model import RANSACRegressor, LinearRegression

today = datetime.datetime.now()
####################################################################################################
#################################################################################################### 
def convertDates(vec):
    CD = vec.apply(lambda x: datetime.datetime(year = int(str(x)[6:10]), month = int(str(x)[3:5]), day = int(str(x)[:2]), hour = int(str(x)[11:13])))
    return CD
####################################################################################################
def Get_SampleAsTS(db, zona):
    db["Giorno"] = pd.to_datetime(db["Giorno"])
    db = db.ix[db["Area"] == zona]
    final = max(db["Giorno"].dt.date)
    strm = str(final.month) if len(str(final.month)) > 1 else "0" + str(final.month)
    strd = str(final.day) if len(str(final.day)) > 1 else "0" + str(final.day)
    final_date = str(final.year) + '-' + strm + '-' + strd    
    dr = pd.date_range('2017-01-01', final_date, freq = 'D')
    res = []
    for i in dr.tolist():
        dbd = db[db.columns[3:]].ix[db["Giorno"] == i].sum()/1000
        res.extend(dbd.values.tolist())
        diz = pd.DataFrame(res)
    diz.columns = [[zona]]
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
    dr = pd.date_range('2017-01-01', '2017-05-31', freq = 'D')
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
def Get_Z(df, zona):
#    df = df.ix[df["CODICE RUC"] == "UC_DP1608_" + zona]
    df = df.ix[df.index.date > datetime.date(2015,12,31)]
    dr = pd.date_range('2016-01-01', '2017-05-31', freq = 'D')
    res = []
    for i in dr.tolist():
        if i.to_pydatetime().date() not in [datetime.date(2016,3,27), datetime.date(2016,10,30),datetime.date(2017,3,26), datetime.date(2017,10,29)]:
            dfd = df.ix[df.index.date == i.to_pydatetime().date()]
            res.extend((dfd.values).tolist())
        else:
            dfd = df.ix[df.index.date == i.to_pydatetime().date()]
            for hour in range(24):
                dfdh = dfd.ix[dfd.index.hour == hour]
                if dfdh.shape[0] == 0:
                    res.append(0)
                elif dfdh.shape[0] == 2:
                    res.append(dfdh.sum())
                else:
                    res.append(dfdh.values[0])
    diz = pd.DataFrame(res)
#    diz.columns = [['MO [MWh]']]
    diz = diz.set_index(pd.date_range('2016-01-01', '2017-12-31', freq = 'H')[:diz.shape[0]])
    return diz
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
        #cmyd = db[db.columns[3:]].ix[db["Giorno"] == i.date()].sum(axis = 0).values.ravel()/1000
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

    TE = pd.DataFrame({'y': test['y'].values.ravel().tolist()}).set_index(test.index)
    TE = TE['y'].sort_index()        
    
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
def DifferentialDataset(db, meteo, zona):
    DFF = OrderedDict()
    db = db.ix[db["Area"] == zona]
    dr = pd.date_range('2017-01-01', '2017-05-31', freq = 'H')
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
#        wd = i.weekday()        
#        td = 2
#        if wd == 0:
#            td = 3
#        cmym = db[db.columns[3:]].ix[db["Giorno"] == (i.date() - datetime.timedelta(days = td))].sum(axis = 0).values.ravel()/1000
        ll.append(meteo['Tmax'].ix[meteo.index.date == i.date()].values[0] - meteo['Tmax'].ix[meteo.index.date == (i.date() - datetime.timedelta(days = 365))].values[0])
        #ll.append(db['y'].ix[db.index.date == i.date()].mean() - db['y'].ix[db.index.date == (i.date() - datetime.timedelta(days = 365))].mean())
        ll.append(meteo['Tmax'].ix[meteo.index.date == i.date()].mean() - meteo['Tmax'].ix[meteo.index.date == (i.date() - datetime.timedelta(days = 7))].mean())
        
#        ll.append(cmym[h])
        
        DFF[i] = ll
    
    DFF = pd.DataFrame.from_dict(DFF, orient = 'index')
    DFF.columns = [['ponte','daylightsaving','endsdaylightsaving','ponte7','daylightsaving7','endsdaylightsaving7',
                    'hol', 'hol7','t0','t1','t2','t3','t4','t5','t6','t7','t8','t9','t10','t11','t12','t13','t14',
                    't15','t16','t17','t18','t19','t20','t21','t22','t23','diff_sample','diff_tmaxY','diff_tmax']] 
    
    month = ["Jan", "Feb", "March","Apr", "May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    for m in month:
        im = month.index(m)
        M = DFF.index.month == (im+1)        
        atm = [int(x) for x in M]
        DFF[m] = atm

    days = ["Lun", "Mar", "Mer","Gio", "Ven","Sab","Dom"]
    for d in days:
        Id = days.index(d)
        M = DFF.index.day == Id        
        atm = [int(x) for x in M]
        DFF[d] = atm
    
    DFF = DFF[["Lun", "Mar", "Mer","Gio", "Ven","Sab","Dom",
               "Jan", "Feb", "March","Apr", "May","Jun","Jul","Aug","Sep","Oct","Nov","Dec",
               'ponte','daylightsaving','endsdaylightsaving','ponte7','daylightsaving7','endsdaylightsaving7',
               'hol', 'hol7','t0','t1','t2','t3','t4','t5','t6','t7','t8','t9','t10','t11','t12','t13','t14',
               't15','t16','t17','t18','t19','t20','t21','t22','t23','diff_tmaxY','diff_tmax','diff_sample']]
    
    return DFF   
####################################################################################################
def DifferentialDatasetForecast(db, meteo, zona, di, df):
    DFF = OrderedDict()
    db = db.ix[db["Area"] == zona]
    dr = pd.date_range(di, df, freq = 'H')
    dr = dr[:dr.size-1]
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
        ll.extend(hvector.tolist())
#        wd = i.weekday()        
#        td = 2
#        if wd == 0:
#            td = 3
#        cmym = db[db.columns[3:]].ix[db["Giorno"] == (i.date() - datetime.timedelta(days = td))].sum(axis = 0).values.ravel()/1000
        ll.append(meteo['Tmax'].ix[meteo.index.date == i.date()].values[0] - meteo['Tmax'].ix[meteo.index.date == (i.date() - datetime.timedelta(days = 365))].values[0])
        #ll.append(db['y'].ix[db.index.date == i.date()].mean() - db['y'].ix[db.index.date == (i.date() - datetime.timedelta(days = 365))].mean())
        ll.append(meteo['Tmax'].ix[meteo.index.date == i.date()].mean() - meteo['Tmax'].ix[meteo.index.date == (i.date() - datetime.timedelta(days = 7))].mean())
        
#        ll.append(cmym[h])
        
        DFF[i] = ll
    
    DFF = pd.DataFrame.from_dict(DFF, orient = 'index')
    DFF.columns = [['ponte','daylightsaving','endsdaylightsaving','ponte7','daylightsaving7','endsdaylightsaving7',
                    'hol', 'hol7','t0','t1','t2','t3','t4','t5','t6','t7','t8','t9','t10','t11','t12','t13','t14',
                    't15','t16','t17','t18','t19','t20','t21','t22','t23','diff_tmaxY','diff_tmax']] 
    
    month = ["Jan", "Feb", "March","Apr", "May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    for m in month:
        im = month.index(m)
        M = DFF.index.month == (im+1)        
        atm = [int(x) for x in M]
        DFF[m] = atm

    days = ["Lun", "Mar", "Mer","Gio", "Ven","Sab","Dom"]
    for d in days:
        Id = days.index(d)
        M = DFF.index.day == Id        
        atm = [int(x) for x in M]
        DFF[d] = atm
    
    DFF = DFF[["Lun", "Mar", "Mer","Gio", "Ven","Sab","Dom",
               "Jan", "Feb", "March","Apr", "May","Jun","Jul","Aug","Sep","Oct","Nov","Dec",
               'ponte','daylightsaving','endsdaylightsaving','ponte7','daylightsaving7','endsdaylightsaving7',
               'hol', 'hol7','t0','t1','t2','t3','t4','t5','t6','t7','t8','t9','t10','t11','t12','t13','t14',
               't15','t16','t17','t18','t19','t20','t21','t22','t23','diff_tmaxY','diff_tmax']]
    
    return DFF   
####################################################################################################
def DifferentialDatasetOOS(db, meteo, zona):
    DFF = OrderedDict()
    dr = pd.date_range('2017-01-01', '2017-06-30', freq = 'D')
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
        dvector = np.repeat(0,7)
        dvector[i.weekday()] = 1
        ll.extend(dvector.tolist())
        ll.extend([bri, dls, edls, bri7, dls7, edls7])
        ll.extend([AddHolidaysDate(i.date()), AddHolidaysDate(i.date() - datetime.timedelta(days = 7))])
        sam_d = db.ix[db.index.date == i.date()].mean().values.ravel()
        sam_w = db.ix[db.index.date == (i.date() - datetime.timedelta(days = 7))].mean().values.ravel()
        ll.append(sam_d[0] - sam_w[0])
#        wd = i.weekday()        
#        td = 2
#        if wd == 0:
#            td = 3
#        cmym = db[db.columns[3:]].ix[db["Giorno"] == (i.date() - datetime.timedelta(days = td))].sum(axis = 0).values.ravel()/1000
#        ll.append(meteo['Tmax'].ix[meteo.index.date == i.date()].values[0] - meteo['Tmax'].ix[meteo.index.date == (i.date() - datetime.timedelta(days = 365))].values[0])
        #ll.append(db['y'].ix[db.index.date == i.date()].mean() - db['y'].ix[db.index.date == (i.date() - datetime.timedelta(days = 365))].mean())
        ll.append(meteo['Tmax'].ix[meteo.index.date == i.date()].mean() - meteo['Tmax'].ix[meteo.index.date == (i.date() - datetime.timedelta(days = 7))].mean())
        
#        ll.append(cmym[h])
        
        DFF[i] = ll
    
    DFF = pd.DataFrame.from_dict(DFF, orient = 'index')
    DFF.columns = [["Lun", "Mar", "Mer","Gio", "Ven","Sab","Dom",'ponte','daylightsaving','endsdaylightsaving','ponte7','daylightsaving7','endsdaylightsaving7',
                    'hol', 'hol7','diff_sample','diff_tmax']] 
    
    month = ["Jan", "Feb", "March","Apr", "May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    for m in month:
        im = month.index(m)
        M = DFF.index.month == (im+1)        
        atm = [int(x) for x in M]
        DFF[m] = atm
    
    DFF = DFF[["Lun", "Mar", "Mer","Gio", "Ven","Sab","Dom",
               "Jan", "Feb", "March","Apr", "May","Jun","Jul","Aug","Sep","Oct","Nov","Dec",
               'ponte','daylightsaving','endsdaylightsaving','ponte7','daylightsaving7','endsdaylightsaving7',
               'hol', 'hol7','diff_tmax','diff_sample']]
    
    return DFF   
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
def MakeExtendedDatasetWithSampleCurve2(df, db, meteo, zona, di, fd = None):
#### @PARAM: df is the dataset from Terna, db, All zona those for computing the perc consumption
#### and the sample curve
#### @BRIEF: extended version of the quasi-omonimous function in Sbilanciamento.py
#### every day will have a dummy variable representing it
    #wdays = ['Lun', 'Mar', 'Mer', 'Gio', 'Ven', 'Sab', 'Dom']
    timedelta = 7
    final = max(df.index.date)
    strm = str(final.month) if len(str(final.month)) > 1 else "0" + str(final.month)
    strd = str(final.day) if len(str(final.day)) > 1 else "0" + str(final.day)
    final_date = str(final.year) + '-' + strm + '-' + strd
    if not fd is None:
        final_date = fd
    to_dt = datetime.datetime.strptime(di, '%Y-%m-%d')    
    psample = percentageConsumption2(db, zona, di, final_date)
    psample = psample.set_index(pd.date_range(di, final_date, freq = 'D')[:psample.shape[0]])
    oos = Get_OutOfSample(df, db, zona)
    oos = oos.ix[oos.index.date >= to_dt.date()]
    dts = OrderedDict()
    df = df.ix[df.index.date >= (to_dt + datetime.timedelta(days = timedelta)).date()]
    indices = pd.date_range((to_dt + datetime.timedelta(days = timedelta)).strftime("%Y-%m-%d"), final_date, freq = 'H')
    for i in indices:
        bri = Bridge(i.date())
        dls = StartsDaylightSaving(i.date())
        edls = EndsDaylightSaving(i.date())
        bri7 = Bridge(i.date() - datetime.timedelta(days = 7))
        dls7 = StartsDaylightSaving(i.date() - datetime.timedelta(days = 7))
        edls7 = EndsDaylightSaving(i.date() - datetime.timedelta(days = 7))
        ll = []        
        hvector = np.repeat(0, 24)
        dvector = np.repeat(0, 7)
        mvector = np.repeat(0,12)
        wd = i.weekday()        
        td = 7
#        if wd == 0:
#            td = 3
        cmym = db[db.columns[3:]].ix[db["Giorno"] == (i.date()- datetime.timedelta(days = td))].sum(axis = 0).values.ravel()/1000
        cmyd = db[db.columns[3:]].ix[db["Giorno"] == i.date()].sum(axis = 0).values.ravel()/1000
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
        ll.extend([dy, Tmax, rain, wind, hol, bri, dls, edls, bri7, dls7, edls7])
        ll.append(meteo['Tmax'].ix[meteo.index.date == i.date()].mean() - meteo['Tmax'].ix[meteo.index.date == (i.date() - datetime.timedelta(days = 7))].mean())
        ll.extend(cmym.tolist())
        
        y_sample = cmyd[h]        
        y_oos = oos.ix[oos.index.date == i.date()].values.ravel()[h]        
        
        ll.extend([y_sample])
        ll.append((ps[0].values[0])/(1 - ps[0].values[0]))
        ll.extend([y_oos])
        
        dts[i] =  ll
    dts = pd.DataFrame.from_dict(dts, orient = 'index')
    dts.columns =[['Lun','Mar','Mer','Gio','Ven','Sab','Dom','Gen','Feb','March','Apr','Mag','Giu','Lug','Ago','Set','Ott','Nov','Dic',
    't0','t1','t2','t3','t4'	,'t5'	,'t6','t7','t8','t9','t10','t11','t12','t13','t14','t15','t16','t17','t18'	,'t19','t20','t21','t22','t23',
    'pday','tmax','pioggia','vento','hol','ponte','dls','edls','ponte7','dls7','edls7','diff_tmax',
    'r0','r1','r2','r3','r4','r5','r6','r7','r8','r9','r10','r11','r12','r13','r14','r15','r16','r17','r18','r19','r20','r21','r22','r23'	,'ysample','perc','yoos']]
    return dts
####################################################################################################
def MakeForecastDataset2(db, meteo, zona, di, time_delta = 1):
#### @PARAM: df is the dataset from Terna, db, All zona those for computing the perc consumption
#### and the sample curve
#### @BRIEF: extended version of the quasi-omonimous function in Sbilanciamento.py
#### every day will have a dummy variable representing it
    #wdays = ['Lun', 'Mar', 'Mer', 'Gio', 'Ven', 'Sab', 'Dom']
    timedelta = 7
    future = datetime.datetime.now() + datetime.timedelta(days = time_delta + 1)
    strm = str(future.month) if len(str(future.month)) > 1 else "0" + str(future.month)
    strd = str(future.day) if len(str(future.day)) > 1 else "0" + str(future.day)
    final_date = str(future.year) + '-' + strm + '-' + strd
    psample = percentageConsumption2(db, zona, di,final_date)
    psample = psample.set_index(pd.date_range(di, final_date, freq = 'D')[:psample.shape[0]])
    dts = OrderedDict()
    dr = pd.date_range(di, final_date, freq = 'H')
    for i in dr[timedelta*24:dr.size-1]:
        bri = Bridge(i.date())
        dls = StartsDaylightSaving(i.date())
        edls = EndsDaylightSaving(i.date())
        bri7 = Bridge(i.date() - datetime.timedelta(days = 7))
        dls7 = StartsDaylightSaving(i.date() - datetime.timedelta(days = 7))
        edls7 = EndsDaylightSaving(i.date() - datetime.timedelta(days = 7))
        ll = []        
        hvector = np.repeat(0, 24)
        dvector = np.repeat(0, 7)
        mvector = np.repeat(0,12)
        wd = i.weekday()        
        td = 7
#        if wd == 0:
#            td = 3
        cmym = db[db.columns[3:]].ix[db["Giorno"] == (i.date()- datetime.timedelta(days = td))].sum(axis = 0).values.ravel()/1000
        if i.date() in pd.to_datetime(db["Giorno"].values.ravel().tolist()):
            cmyd = db[db.columns[3:]].ix[db["Giorno"] == i.date()].sum(axis = 0).values.ravel()/1000
        else:
            cmyd = np.repeat(0,24)
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
        ll.extend([dy, Tmax, rain, wind, hol, bri, dls, edls, bri7, dls7, edls7])
        ll.append(meteo['Tmax'].ix[meteo.index.date == i.date()].mean() - meteo['Tmax'].ix[meteo.index.date == (i.date() - datetime.timedelta(days = 7))].mean())
        ll.extend(cmym.tolist())
        ll.append(cmyd[h])
        ll.append((ps[0].values[0])/(1 - ps[0].values[0]))
        dts[i] =  ll
    dts = pd.DataFrame.from_dict(dts, orient = 'index')
    dts.columns =[['Lun','Mar','Mer','Gio','Ven','Sab','Dom','Gen','Feb','March','Apr','Mag','Giu','Lug','Ago','Set','Ott','Nov','Dic',
    't0','t1','t2','t3','t4'	,'t5'	,'t6','t7','t8','t9','t10','t11','t12','t13','t14','t15','t16','t17','t18'	,'t19','t20','t21','t22','t23',
    'pday','tmax','pioggia','vento','hol','ponte','dls','edls','ponte7','dls7','edls7','diff_tmax',
    'r0','r1','r2','r3','r4','r5','r6','r7','r8','r9','r10','r11','r12','r13','r14','r15','r16','r17','r18','r19','r20','r21','r22','r23'	,'ysample','perc']]
    return dts
####################################################################################################
def MakeCompletedDatasets(df, db, meteo1, meteo2, zona, di1, di2, time_delta = 1, fd = None):
    DBB = MakeExtendedDatasetWithSampleCurve2(df, db, meteo1, zona, di1, fd)
    DFT = MakeForecastDataset2(db, meteo2, di2, time_delta)
    TF = DFT.ix[DFT['ysample'] == 0]
    TT = DFT.ix[DFT['ysample'] > 0]
    TrainSample = DBB[DBB.columns[:81]].append(TT)
    return TrainSample, DBB, TF
####################################################################################################
def Regress_OOS(DBB, TF2, linear = False):
    yf_oos = np.repeat(0,24)
    TF3 = TF2.ix[TF2.index.date == datetime.datetime.now().date() + datetime.timedelta(days = 1)]
    for h in range(24):
        if linear:
            lm = RANSACRegressor(base_estimator = LinearRegression(fit_intercept = True))
            lm.fit(DBB[DBB.columns[:81]].ix[DBB.index.hour == h], DBB['yoos'].ix[DBB.index.hour == h])
            print r2_score(DBB['yoos'].ix[DBB.index.hour == h], lm.predict(DBB[DBB.columns[:81]].ix[DBB.index.hour == h]))
            if TF3.ix[TF3.index.hour == h].shape[0] > 0:
                yf_oos[h] = lm.predict(TF3.ix[TF3.index.hour == h])
            else:
                pass
        else:
            rf = RandomForestRegressor(criterion = 'mse', max_depth = 48, n_estimators = 24, n_jobs = 1)
            rf.fit(DBB[DBB.columns[:81]].ix[DBB.index.hour == h], DBB['yoos'].ix[DBB.index.hour == h])
            print r2_score(DBB['yoos'].ix[DBB.index.hour == h], rf.predict(DBB[DBB.columns[:81]].ix[DBB.index.hour == h]))                
            if TF3.ix[TF3.index.hour == h].shape[0] > 0:
                yf_oos[h] = rf.predict(TF3.ix[TF3.index.hour == h])
            else:
                pass
    return yf_oos
####################################################################################################
def SimplePredictOOS(oos, meteo, dtp):
    wd = dtp.weekday()
    dow = map(lambda date: date.weekday(), oos.index.date)
    candidate = oos.ix[np.where(np.array(dow) == wd)]  #.values.ravel()[-24:]
    candidate_date = max(candidate.index.date)
    if AddHolidaysDate(candidate_date) == 1 and wd < 5:
        candidate = oos.ix[np.where(np.array(dow) == wd)].values.ravel()[-24:]
    else:
        candidate = oos.ix[np.where(np.array(dow) == wd)].values.ravel()[-48:-24]        
    if wd < 5 and AddHolidaysDate(dtp) == 0:        
        model = joblib.load('C:/Users/utente/Documents/Sbilanciamento/Tmax_load_OOS_not_holiday.pkl')
    else:
        model = joblib.load('C:/Users/utente/Documents/Sbilanciamento/Tmax_load_OOS_holiday.pkl')
    estimated_mean = model.predict(meteo['Tmax'].ix[meteo.index.date == dtp].values.ravel().reshape(-1,1))
    return candidate - (np.mean(candidate) - estimated_mean)
####################################################################################################
def Get_Prediction(df, db, meteo1, meteo2, zona, di1, di2, time_delta = 1, fd = None):
    TS, DBB, TF = MakeCompletedDatasets(df, db, meteo1, meteo2, zona, di1, di2, time_delta = 1, fd = None)
    DBtrain = DBB.sample(frac = 1)
    TStrain = TS.sample(frac = 1)
    brs = RandomForestRegressor(criterion = 'mse', max_depth = 48, n_estimators = 24, n_jobs = 1)
    bro = RandomForestRegressor(criterion = 'mse', max_depth = 48, n_estimators = 24, n_jobs = 1)
    brs.fit(TStrain[TStrain.columns[:79]], TStrain['ysample'])
    print r2_score(TS['ysample'], brs.predict(TS[TS.columns[:79]]))
    bro.fit(DBtrain[DBtrain.columns[:81]], DBtrain['yoos'])
    #bro.fit(DBB[DBB.columns[:81]], DBB['yoos'])    
    print r2_score(DBB['yoos'], bro.predict(DBB[DBB.columns[:81]]))
    yf_sample = brs.predict(TF[TF.columns[:79]])
    TF2 = TF
    TF2['ysample'] = yf_sample
    yf_oos = Regress_OOS(DBB, TF2) 
    yf_oos2 = bro.predict(TF2) 
    yfoos = (yf_oos + yf_oos2[-24:])*1/2
    yf = yf_sample[-24:] + yfoos
    YF = pd.DataFrame({zona: yf}).set_index(TF2.ix[TF2.index.date == datetime.datetime.now().date() + datetime.timedelta(days = 1)].index)
    YS = pd.DataFrame({zona: yf_sample[-24:]}).set_index(TF2.ix[TF2.index.date == datetime.datetime.now().date() + datetime.timedelta(days = 1)].index)
    if os.path.exists('C:/Users/utente/Documents/Sbilanciamento/forecast_' + zona + '.h5' ):
        forecast = pd.read_hdf('C:/Users/utente/Documents/Sbilanciamento/forecast_' + zona + '.h5')
        forecast = forecast.append(YF)
        forecast.to_hdf('C:/Users/utente/Documents/Sbilanciamento/forecast_' + zona + '.h5', 'forecast_nord')
        sample = pd.read_hdf('C:/Users/utente/Documents/Sbilanciamento/forecast_campione_' + zona + '.h5', 'sample_nord')
        sample = forecast.append(YS)
        sample.to_hdf('C:/Users/utente/Documents/Sbilanciamento/forecast_campione_' + zona + '.h5', 'sample_nord')
    else:
        YF.to_hdf('C:/Users/utente/Documents/Sbilanciamento/forecast_' + zona + '.h5', 'forecast_nord')
        YS.to_hdf('C:/Users/utente/Documents/Sbilanciamento/forecast_campione_NORD.h5', 'sample_NORD')
    return YF
####################################################################################################
def Get_Prediction2(df, db, meteo1, meteo2, zona, di1, di2, time_delta = 1, fd = None):
    TS, DBB, TF = MakeCompletedDatasets(df, db, meteo1, meteo2, zona, di1, di2, time_delta = 1, fd = None)
    TStrain = TS.sample(frac = 1)
    brs = RandomForestRegressor(criterion = 'mse', max_depth = 1000, n_estimators = 3000, n_jobs = 1)
    brs.fit(TStrain[TStrain.columns[:79]], TStrain['ysample'])
    print r2_score(TS['ysample'], brs.predict(TS[TS.columns[:79]]))
    yf_sample = brs.predict(TF[TF.columns[:79]])
#    jun = pd.read_excel('C:/Users/utente/Documents/Sbilanciamento/Consuntivo_giu.xlsx')
#    sam = Get_SampleAsTS(db, zona)
#    samj = sam.ix[sam.index.month == 6]
#    oosj = jun['NORD'].values.ravel() - samj.values.ravel()    
#    oosj = pd.DataFrame({'oos': oosj}).set_index(samj.index)
#    oos = Get_OutOfSample(nord, DB2, 'NORD')    
#    oos.columns = ['oos']    
#    oos = oos.append(oosj)    
    oos = pd.read_hdf('C:/Users/utente/Documents/Sbilanciamento/storico_oos_' + zona + '.h5')
    yfoos = SimplePredictOOS(oos, meteo2, max(TF.index.date))
    yf = yf_sample[-24:] + yfoos
    YF = pd.DataFrame({zona: yf}).set_index(TF.ix[TF.index.date == datetime.datetime.now().date() + datetime.timedelta(days = 1)].index)
    YS = pd.DataFrame({zona: yf_sample[-24:]}).set_index(TF.ix[TF.index.date == datetime.datetime.now().date() + datetime.timedelta(days = 1)].index)
    if os.path.exists('C:/Users/utente/Documents/Sbilanciamento/forecast_' + zona + '.h5' ):
        forecast = pd.read_hdf('C:/Users/utente/Documents/Sbilanciamento/forecast_' + zona + '.h5')
        forecast = forecast.append(YF)
        forecast.to_hdf('C:/Users/utente/Documents/Sbilanciamento/forecast_' + zona + '.h5', 'forecast_' + zona.lower())
        sample = pd.read_hdf('C:/Users/utente/Documents/Sbilanciamento/forecast_campione_' + zona + '.h5', 'sample_' + zona.lower())
        sample = forecast.append(YS)
        sample.to_hdf('C:/Users/utente/Documents/Sbilanciamento/forecast_campione_' + zona + '.h5', 'sample_' + zona.lower())
    else:
        YF.to_hdf('C:/Users/utente/Documents/Sbilanciamento/forecast_' + zona + '.h5', 'forecast_' + zona.lower())
        YS.to_hdf('C:/Users/utente/Documents/Sbilanciamento/forecast_campione_NORD.h5', 'sample_' + zona.lower())
    return YF
####################################################################################################
def H1_norm(x):
    h1 = np.mean(x**2) + np.mean(np.diff(x)**2)
    return np.sqrt(h1)
####################################################################################################
def CompareSubsequentPrediction(zona, td = 1):
    ED = pd.read_hdf('C:/Users/utente/Documents/Sbilanciamento/variazioni_gg_cons.h5')
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
        qhat = float(np.where(var_nw <= HN)[0].size)/float(var_nw.size)
        phat = float(np.where(var_nw >= HN)[0].size)/float(var_nw.size)
        print 'estimated quantile point of new variation: {}'.format(qhat)
        print 'estimated percentile point of new variation: {}'.format(phat)
        newED = pd.DataFrame({'Data':[ today.date() + datetime.timedelta(days = td)],'var':[HN],'giorno':[(today.date() + datetime.timedelta(days = td)).weekday()]})
        ED = ED.append(newED)
        ED.to_hdf('C:/Users/utente/Documents/Sbilanciamento/variazioni_gg_cons.h5', 'VAR')
    return 1
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
def Get_DependencyWithTemperature(df, meteo):
### @PARAM: df is the dataset I want to compute the dependency of: it could be Sample, OOS or Zonal and
### needs to be a time series
    
    final = min(max(df.index.date), max(meteo.index.date))
    strm = str(final.month) if len(str(final.month)) > 1 else "0" + str(final.month)
    strd = str(final.day) if len(str(final.day)) > 1 else "0" + str(final.day)
    final_date = str(final.year) + '-' + strm + '-' + strd
    
    di = max(min(df.index.date), min(meteo.index.date))    
    
    dts = OrderedDict()
    indices = pd.date_range(di, final_date, freq = 'H')
    for i in indices:
        bri = Bridge(i.date())
        dls = StartsDaylightSaving(i.date())
        edls = EndsDaylightSaving(i.date())
        ll = []        
        hvector = np.repeat(0, 24)
        dvector = np.repeat(0, 7)
        mvector = np.repeat(0,12)
        wd = i.weekday()        

        dvector[wd] = 1
        h = i.hour
        hvector[h] = 1
        mvector[(i.month-1)] = 1
        dy = i.timetuple().tm_yday
        Tmax = meteo['Tmax'].ix[meteo['DATA'] == i.date()].values.ravel()[0]
        rain = meteo['PIOGGIA'].ix[meteo['DATA'] == i.date()].values.ravel()[0]
        wind = meteo['VENTOMEDIA'].ix[meteo['DATA'] == i.date()].values.ravel()[0]
        hol = AddHolidaysDate(i.date())
        ll.extend(dvector.tolist())
        ll.extend(mvector.tolist())
        ll.extend(hvector.tolist())        
        ll.extend([dy, Tmax, rain, wind, hol, bri, dls, edls])
#        ll.append(meteo['Tmax'].ix[meteo.index.date == i.date()].mean() - meteo['Tmax'].ix[meteo.index.date == (i.date() - datetime.timedelta(days = 7))].mean())

        y = df.ix[i].values.ravel()[0]
        
        ll.append(y)        
        dts[i] =  ll
        
    dts = pd.DataFrame.from_dict(dts, orient = 'index')
    dts.columns =[['Lun','Mar','Mer','Gio','Ven','Sab','Dom','Gen','Feb','March','Apr','Mag','Giu','Lug','Ago','Set','Ott','Nov','Dic',
    't0','t1','t2','t3','t4'	,'t5'	,'t6','t7','t8','t9','t10','t11','t12','t13','t14','t15','t16','t17','t18'	,'t19','t20','t21','t22','t23',
    'pday','tmax','pioggia','vento','hol','ponte','dls','edls','y']]
    return dts
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
    #basal_cons = min(df.ix[df.index.month == 4].mean().values.ravel()[0] ,df.ix[df.index.month == 5].mean().values.ravel()[0])   
    basal_cons = 0#df.mean().values[0]
    
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
def DependencyWithTemperature(dtf, meteo):
### @PARAM: df is the dataset I want to compute the dependency of: it could be Sample, OOS or Zonal and
### needs to be a time series
    
    dts = OrderedDict()
    for d in dtf:
        bri = Bridge(d)
        dls = StartsDaylightSaving(d)
        edls = EndsDaylightSaving(d)
        ll = []        
        #hvector = np.repeat(0, 24)
        dvector = np.repeat(0, 7)
        mvector = np.repeat(0,12)
        wd = d.weekday()        
        dvector[wd] = 1
        #h = d.hour
        #hvector[h] = 1
        mvector[(d.month-1)] = 1
        dy = d.timetuple().tm_yday
        Tmax = meteo['Tmax'].ix[meteo.index.date == d].values.ravel()[0]
        rain = meteo['pioggia'].ix[meteo.index.date == d].values.ravel()[0]
        wind = meteo['vento'].ix[meteo.index.date == d].values.ravel()[0]
        hol = AddHolidaysDate(d)
        ll.extend(dvector.tolist())
        ll.extend(mvector.tolist())
        #ll.extend(hvector.tolist())        
        ll.extend([dy, Tmax, rain, wind, hol, bri, dls, edls])
    #        ll.append(meteo['Tmax'].ix[meteo.index.date == i.date()].mean() - meteo['Tmax'].ix[meteo.index.date == (i.date() - datetime.timedelta(days = 7))].mean())
    
            
        dts[d] =  ll
        
    dts = pd.DataFrame.from_dict(dts, orient = 'index')
    dts.columns =[['Lun','Mar','Mer','Gio','Ven','Sab','Dom','Gen','Feb','March','Apr','Mag','Giu','Lug','Ago','Set','Ott','Nov','Dic',
    'pday','tmax','pioggia','vento','hol','ponte','dls','edls']]
    return dts
####################################################################################################
def ModelDependencyTemperature(df, meteo):
    
    DWT = Get_DependencyWithTemperature(df, meteo)
    
    DWT = DWT.sample(frac = 1)

    rf = RandomForestRegressor(criterion = 'mse', n_estimators = 3000, n_jobs = 1)
    rf.fit(DWT[DWT.columns[:28]], DWT['y'])
    
    print r2_score(DWT['y'], rf.predict(DWT[DWT.columns[:51]]))
    print mean_squared_error(DWT['y'], rf.predict(DWT[DWT.columns[:51]]))
    
#    print r2_score(ESte['y'], rf.predict(ESte[ESte.columns[:28]]))
####################################################################################################
def Estimate_Sbil(db, meteo, zona):

    Sam = Get_SampleAsTS(db, zona)
    Samd = Sam.resample('D').mean()
    Samdd = Samd.diff()
    
    Err = Samdd.ix[1:].values.ravel()/Samd.ix[1:].values.ravel()
    Err = pd.DataFrame({'Sbil': Err}).set_index(Samd.ix[1:].index)
   
    dts = OrderedDict()                     ## '2017-07-31'
    indices = pd.to_datetime(pd.date_range('2017-01-01', max(Sam.index.date), freq = 'D'))
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
#        Tmax = meteo['Tmax'].ix[meteo.index.date == i.date()].values.ravel()[0]
#        rain = meteo['pioggia'].ix[meteo.index.date == i.date()].values.ravel()[0]
#        wind = meteo['vento'].ix[meteo.index.date == i.date()].values.ravel()[0]
        Tmax = meteo['Tmax'].ix[meteo.index == i.date()].values.ravel()[0]
        rain = meteo['pioggia'].ix[meteo.index == i.date()].values.ravel()[0]
        wind = meteo['vento'].ix[meteo.index == i.date()].values.ravel()[0]                
        hol = AddHolidaysDate(i.date())
        ll.extend(dvector.tolist())
        ll.extend(mvector.tolist())
        ll.extend([dy, Tmax, rain, wind, hol, bri, dls, edls])
#        ll.append(meteo['Tmax'].ix[meteo.index.date == i.date()].mean() - meteo['Tmax'].ix[meteo.index.date == (i.date() - datetime.timedelta(days = 7))].mean())
        ll.append(Samd.ix[i].values.ravel()[0])
        ll.append(Samdd.ix[i].values.ravel()[0])
        y = Err.ix[i].values.ravel()[0]
        
        #ll.append(np.exp(y)) 
        ll.append(y)
        dts[i] =  ll
        
    dts = pd.DataFrame.from_dict(dts, orient = 'index')
    dts.columns =[['Lun','Mar','Mer','Gio','Ven','Sab','Dom','Gen','Feb','March','Apr','Mag','Giu','Lug','Ago','Set','Ott','Nov','Dic',
                   'pday','tmax','pioggia','vento','hol','ponte','dls','edls','mean_load','mean_diff','y']]
    return dts
####################################################################################################
def Predict_Sbil(db, meteo, zona, dtf):

    Sam = Get_SampleAsTS(db, zona)
    Samd = Sam.resample('D').mean()
    Samdd = Samd.diff()
    
    Err = Samdd.ix[1:].values.ravel()/Samd.ix[1:].values.ravel()
    Err = pd.DataFrame({'Sbil': Err}).set_index(Samd.ix[1:].index)
    
    sample = pd.read_excel('C:/Users/utente/Documents/Sbilanciamento/forecast_campione_' + zona + '.xlsx')
    Spred = sample.ix[dtf].values.ravel()[0]    
    
    dts = OrderedDict()                     ## '2017-07-31'
    bri = Bridge(dtf)
    dls = StartsDaylightSaving(dtf)
    edls = EndsDaylightSaving(dtf)
    ll = []        
    dvector = np.repeat(0, 7)
    mvector = np.repeat(0,12)
    wd = dtf.weekday()        

    dvector[wd] = 1
    mvector[(dtf.month-1)] = 1
    dy = dtf.timetuple().tm_yday
#        Tmax = meteo['Tmax'].ix[meteo.index.date == i.date()].values.ravel()[0]
#        rain = meteo['pioggia'].ix[meteo.index.date == i.date()].values.ravel()[0]
#        wind = meteo['vento'].ix[meteo.index.date == i.date()].values.ravel()[0]
    Tmax = meteo['Tmax'].ix[meteo.index == dtf].values.ravel()[0]
    rain = meteo['pioggia'].ix[meteo.index == dtf].values.ravel()[0]
    wind = meteo['vento'].ix[meteo.index == dtf].values.ravel()[0]                
    hol = AddHolidaysDate(dtf)
    ll.extend(dvector.tolist())
    ll.extend(mvector.tolist())
    ll.extend([dy, Tmax, rain, wind, hol, bri, dls, edls])
#        ll.append(meteo['Tmax'].ix[meteo.index.date == i.date()].mean() - meteo['Tmax'].ix[meteo.index.date == (i.date() - datetime.timedelta(days = 7))].mean())
    #ll.append(Samd.ix[dtf - datetime.timedelta(days = 2)].values.ravel()[0])
    #ll.append(Samdd.ix[dtf - datetime.timedelta(days = 2)].values.ravel()[0])
    ll.append(Spred.mean())
    ll.append(Spred.mean() - Samd.ix[dtf - datetime.timedelta(days = 2)].values.ravel()[0])
        
        #ll.append(np.exp(y)) 
    dts[dtf] = ll
    dts = pd.DataFrame.from_dict(dts, orient = 'index')
    dts.columns =[['Lun','Mar','Mer','Gio','Ven','Sab','Dom','Gen','Feb','March','Apr','Mag','Giu','Lug','Ago','Set','Ott','Nov','Dic',
                   'pday','tmax','pioggia','vento','hol','ponte','dls','edls','mean_load','mean_diff']]
    return dts
####################################################################################################    