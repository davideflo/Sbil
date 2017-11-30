# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 09:36:18 2017

@author: utente

Sbilanciamento 8 -- OOS PODWISE MODEL FROM SOS --
https://www.r-bloggers.com/using-regression-trees-for-forecasting-double-seasonal-time-series-with-trend-in-r/
"""

import pandas as pd
import numpy as np
import datetime
import os
import calendar
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from collections import OrderedDict
import scipy

####################################################################################################
def GetRicDateGeneral(dtf):
### Get the "most similar" corresponding date, given history   
    strm = str(dtf.month) if len(str(dtf.month)) > 1 else "0" + str(dtf.month)

    monthrange = pd.date_range(str(dtf.year) + '-' + strm + '-01', str(dtf.year) + '-' + strm + '-' + str(dtf.day), freq = 'D')
    todow = map(lambda date: date.weekday(), monthrange)

    monthrangey1 = pd.date_range(str(dtf.year - 1) + '-' + strm + '-01', str(dtf.year - 1) + '-' + strm + '-' + str(calendar.monthrange(dtf.year - 1, dtf.month)[1]), freq = 'D')
    todowy1 = map(lambda date: date.weekday(), monthrangey1)
    
    dow = dtf.weekday()
    dow_counter = np.where(np.array(todow) == dow)[0].size - 1
    
    dow_countery11 = [i for i, x in enumerate(todowy1) if x == dow]

    
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
        return GetRicHoliday(dtf)
####################################################################################################
def GetRicHoliday(vd):
    pasquetta = [datetime.date(2015,4,6), datetime.date(2016,3,28), datetime.date(2017,4,17)]
    pasqua = [datetime.date(2015,4,5), datetime.date(2016,3,27), datetime.date(2017,4,16)]

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
    
    if dtf == datetime.date(2017,10,29):
        return datetime.date(2016,10,30)
    if dtf == datetime.date(2017,3,26):
        return datetime.date(2016,3,27)
    
    
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
#def ModifyPredictionPerPOD(pod, consumption, meteo):
### modify podwise prediction with temperature
#    wd = dtp.weekday()
#    dow = map(lambda date: date.weekday(), oos.index.date)
#    candidate = oos.ix[np.where(np.array(dow) == wd)]  #.values.ravel()[-24:]
#    candidate_date = max(candidate.index.date)
#    if AddHolidaysDate(candidate_date) == 1 and wd < 5:
#        candidate = oos.ix[np.where(np.array(dow) == wd)].values.ravel()[-24:]
#    else:
#        candidate = oos.ix[np.where(np.array(dow) == wd)].values.ravel()[-48:-24]        
#    if wd < 5 and AddHolidaysDate(dtp) == 0:        
#        model = joblib.load('C:/Users/utente/Documents/Sbilanciamento/Tmax_load_OOS_not_holiday.pkl')
#    else:
#        model = joblib.load('C:/Users/utente/Documents/Sbilanciamento/Tmax_load_OOS_holiday.pkl')
#    estimated_mean = model.predict(meteo['Tmax'].ix[meteo.index.date == dtp].values.ravel().reshape(-1,1))
#    return candidate - (np.mean(candidate) - estimated_mean)
####################################################################################################
def LearnPodwiseModel(db, pod, meteo, zona, pod_in_sample = False):
    if os.path.exists('C:/Users/utente/Documents/Sbilanciamento/DB_PDO_' + zona + '.h5'):     
        dbpdo = pd.read_hdf('C:/Users/utente/Documents/Sbilanciamento/DB_PDO_' + zona + '.h5')
        dpp = dbpdo.ix[dbpdo['Pod'] == pod]
    dbsos = pd.read_hdf('C:/Users/utente/Documents/Sbilanciamento/DB_SOS_' + zona + '.h5')
    
    dsp = dbsos.ix[dbsos['Pod'] == pod]
    if dpp.shape[0] > 0:
        df = dsp.append(dpp)
    
    dl = df['Giorno'].values.ravel().tolist()
    
    if pod_in_sample:
        db = db.ix[db['POD'] == pod]
    
    dts = OrderedDict()
    for i in dl:
        bri = Bridge(i.date())
        dls = StartsDaylightSaving(i.date())
        edls = EndsDaylightSaving(i.date())
        bri7 = Bridge(GetRicDate(i.date()))
        dls7 = StartsDaylightSaving(GetRicDate(i.date()))
        edls7 = EndsDaylightSaving(GetRicDate(i.date()))
        ll = []      
        
            
        hvector = np.repeat(0, 24)
        dvector = np.repeat(0, 7)
        mvector = np.repeat(0,12)
        wd = i.weekday()        
        dvector[wd] = 1
        
        todow = map(lambda date: date.weekday(), pd.date_range(datetime.date(i.year,i.month,1), i.date(), freq = 'D'))
        dow_counter = np.where(np.array(todow) == wd)[0].size         
        ovector = np.repeat(0,5)        
        ovector[dow_counter-1] = 1
        
        h = i.hour
        hvector[h] = 1
        mvector[(i.month-1)] = 1
        #dy = i.timetuple().tm_yday
        Tmax = meteo['Tmax'].ix[meteo.index.date == i.date()].values.ravel()[0]
        rain = meteo['pioggia'].ix[meteo.index.date == i.date()].values.ravel()[0]
        wind = meteo['vento'].ix[meteo.index.date == i.date()].values.ravel()[0]
        hol = AddHolidaysDate(i.date())
        ll.extend(dvector.tolist())
        ll.extend(ovector.tolist())
        ll.extend(mvector.tolist())
        ll.extend(hvector.tolist())        
        ll.extend([Tmax, rain, wind, hol, bri, dls, edls, bri7, dls7, edls7])
        
        if pod_in_sample:
            cmym = db[db.columns[3:]].ix[db["Giorno"] == (i.date()- datetime.timedelta(days = 7))].sum(axis = 0).values.ravel()/1000
            ll.append(meteo['Tmax'].ix[meteo.index.date == i.date()].mean() - meteo['Tmax'].ix[meteo.index.date == (i.date() - datetime.timedelta(days = 7))].mean())
            ll.extend(cmym.tolist())      
            y = df[str(h+1)].ix[df['Giorno'] == i]
            ll.append(y)
            dts[i] =  ll
            dts = pd.DataFrame.from_dict(dts, orient = 'index')
            dts.columns =[['Lun','Mar','Mer','Gio','Ven','Sab','Dom','primo','secondo','terzo','quarto','quinto','Gen','Feb','March','Apr','Mag','Giu','Lug','Ago','Set','Ott','Nov','Dic',
                           't0','t1','t2','t3','t4'	,'t5'	,'t6','t7','t8','t9','t10','t11','t12','t13','t14','t15','t16','t17','t18'	,'t19','t20','t21','t22','t23',
                           'tmax','pioggia','vento','hol','ponte','dls','edls','ponte7','dls7','edls7','diff_tmax',
                           'r0','r1','r2','r3','r4','r5','r6','r7','r8','r9','r10','r11','r12','r13','r14','r15','r16','r17','r18','r19','r20','r21','r22','r23','y']]
                          
        
        else:
            ll.append(meteo['Tmax'].ix[meteo.index.date == i.date()].mean() - meteo['Tmax'].ix[meteo.index.date == GetRicDate(i.date())].mean())
            y = df[str(h+1)].ix[df['Giorno'] == i]
            ll.append(y)
            dts[i] = ll
            dts = pd.DataFrame.from_dict(dts, orient = 'index')
            dts.columns =[['Lun','Mar','Mer','Gio','Ven','Sab','Dom','primo','secondo','terzo','quarto','quinto','Gen','Feb','March','Apr','Mag','Giu','Lug','Ago','Set','Ott','Nov','Dic',
                           't0','t1','t2','t3','t4'	,'t5'	,'t6','t7','t8','t9','t10','t11','t12','t13','t14','t15','t16','t17','t18'	,'t19','t20','t21','t22','t23',
                           'tmax','pioggia','vento','hol','ponte','dls','edls','ponte7','dls7','edls7','diff_tmax','y']]
                           
    rf = RandomForestRegressor(criterion = 'mse', max_depth = 48, n_estimators = 24, n_jobs = 1)
    if len(dl) > 365:
        dts_train, dts_test, y_train, y_test = train_test_split(dts[dts.columns[:-1]], dts['y'], test_size = 0.2, random_state=42)
        rf.fit(dts_train, y_train)
        print 'R2 on test set = {}'.format(r2_score(y_test, rf.predict(dts_test)))
        print 'MSE on test set: {}'.format(mean_squared_error(y_test, rf.predict(dts_test)))
    else:
        dts_train = dts.sample(frac = 1)        
        rf.fit(dts_train[dts_train.columns[:-1]], dts_train['y'])
        print 'R2 on train set = {}'.format(r2_score(dts_train['y'], rf.predict(dts_train[dts_train.columns[:-1]])))
        print 'MSE on test set: {}'.format(mean_squared_error(dts_train['y'], rf.predict(dts_train[dts_train.columns[:-1]])))
    
    joblib.dump(rf, 'C:/Users/utente/Documents/Sbilanciamento/' + zona + 'Podwise_model/' + pod + '_model.pkl')
    
    return dts
####################################################################################################
def DatasetForPodwiseForecast(db, meteo, dtf, td = 1, OOS = False):
### @PARAM: dtf could be a vector of dates meteo needs to be at least a year backwards of dtf
    if not isinstance(dtf, list):
        dtf = [dtf, dtf + datetime.timedelta(days = td)]
    dts = OrderedDict()
    dr = pd.date_range(dtf[0], dtf[-1], freq = 'H')
    for i in dr:
        bri = Bridge(i.date())
        dls = StartsDaylightSaving(i.date())
        edls = EndsDaylightSaving(i.date())
        bri7 = Bridge(GetRicDate(i.date()))
        dls7 = StartsDaylightSaving(GetRicDate(i.date()))
        edls7 = EndsDaylightSaving(GetRicDate(i.date()))
        ll = []      
        
            
        hvector = np.repeat(0, 24)
        dvector = np.repeat(0, 7)
        mvector = np.repeat(0,12)
        wd = i.weekday()        
        dvector[wd] = 1
        
        todow = map(lambda date: date.weekday(), pd.date_range(datetime.date(i.year,i.month,1), i.date(), freq = 'D'))
        dow_counter = np.where(np.array(todow) == wd)[0].size         
        ovector = np.repeat(0,5)        
        ovector[dow_counter-1] = 1
        
        h = i.hour
        hvector[h] = 1
        mvector[(i.month-1)] = 1
        #dy = i.timetuple().tm_yday
        Tmax = meteo['Tmax'].ix[meteo.index.date == i.date()].values.ravel()[0]
        rain = meteo['pioggia'].ix[meteo.index.date == i.date()].values.ravel()[0]
        wind = meteo['vento'].ix[meteo.index.date == i.date()].values.ravel()[0]
        hol = AddHolidaysDate(i.date())
        ll.extend(dvector.tolist())
        ll.extend(ovector.tolist())
        ll.extend(mvector.tolist())
        ll.extend(hvector.tolist())        
        ll.extend([Tmax, rain, wind, hol, bri, dls, edls, bri7, dls7, edls7])
        
        if not OOS:         
            cmym = db[db.columns[3:]].ix[db["Giorno"] == (i.date()- datetime.timedelta(days = 7))].sum(axis = 0).values.ravel()/1000
            ll.append(meteo['Tmax'].ix[meteo.index.date == i.date()].mean() - meteo['Tmax'].ix[meteo.index.date == (i.date() - datetime.timedelta(days = 7))].mean())
            ll.extend(cmym.tolist())      
            dts[i] =  ll
            dts = pd.DataFrame.from_dict(dts, orient = 'index')
            dts.columns =[['Lun','Mar','Mer','Gio','Ven','Sab','Dom','primo','secondo','terzo','quarto','quinto','Gen','Feb','March','Apr','Mag','Giu','Lug','Ago','Set','Ott','Nov','Dic',
                           't0','t1','t2','t3','t4'	,'t5'	,'t6','t7','t8','t9','t10','t11','t12','t13','t14','t15','t16','t17','t18'	,'t19','t20','t21','t22','t23',
                           'tmax','pioggia','vento','hol','ponte','dls','edls','ponte7','dls7','edls7','diff_tmax',
                           'r0','r1','r2','r3','r4','r5','r6','r7','r8','r9','r10','r11','r12','r13','r14','r15','r16','r17','r18','r19','r20','r21','r22','r23']]
        else:
            ll.append(meteo['Tmax'].ix[meteo.index.date == i.date()].mean() - meteo['Tmax'].ix[meteo.index.date == GetRicDate(i.date())].mean())
            dts[i] =  ll
            dts = pd.DataFrame.from_dict(dts, orient = 'index')
            dts.columns =[['Lun','Mar','Mer','Gio','Ven','Sab','Dom','primo','secondo','terzo','quarto','quinto','Gen','Feb','March','Apr','Mag','Giu','Lug','Ago','Set','Ott','Nov','Dic',
                           't0','t1','t2','t3','t4'	,'t5'	,'t6','t7','t8','t9','t10','t11','t12','t13','t14','t15','t16','t17','t18'	,'t19','t20','t21','t22','t23',
                           'tmax','pioggia','vento','hol','ponte','dls','edls','ponte7','dls7','edls7','diff_tmax']]
                           
    return dts
####################################################################################################
def OOSPodwiseModel(db, crpp, dtf, zona, meteo):
### @PARAM: db is fthe database of the daily measures, crpp is the monthly database of the active pods and
### dtf is the day to forecast
    strm = str(dtf.month) if len(str(dtf.month)) > 1 else "0" + str(dtf.month)
    active = set(crpp['pod'].ix[crpp['Consumo_' + strm] > 0].values.ravel().tolist())
    not_dailies = list(active.difference(set(db['POD'].ix[db['Giorno'] == (dtf - datetime.timedelta(days = 2))].values.ravel().tolist())))
    if os.path.exists('C:/Users/utente/Documents/Sbilanciamento/DB_PDO_' + zona + '.h5'):     
        dbpdo = pd.read_hdf('C:/Users/utente/Documents/Sbilanciamento/DB_PDO_' + zona + '.h5')
    dbsos = pd.read_hdf('C:/Users/utente/Documents/Sbilanciamento/DB_SOS_' + zona + '.h5')
    
    isholiday = AddHolidaysDate(dtf)    

    dttf = DatasetForPodwiseForecast(db, meteo, dtf)    
    prediction = np.repeat(0,24)
    
    if isholiday == 0:
        for p in not_dailies:
            if dbpdo.shape[0] > 0 and p in dbpdo['Pod'].values.ravel().list():
                if os.path.exists('C:/Users/utente/Documents/Sbilanciamento/' + zona + '/Podwise_models/' + p + '_model.pkl'):
                    model = joblib.load('C:/Users/utente/Documents/Sbilanciamento/' + zona + '/Podwise_models/' + p + '_model.pkl')
                    mp = model.predict(dttf)
                    prediction += mp
                else:
                    dfp = dbpdo.ix[dbpdo['Pod'] == p]
                    dfpy = dfp.ix[dfp['Giorno'] == GetRicDate(dtf)]
                    if dfpy.shape[0] > 0:
                        prediction += ModifyPrediction(dfpy['Pod'].values.ravel()[0], dfpy[dfpy.columns[3:]].values.ravel(), meteo)
                    else:
                        dsp = dbsos.ix[dbsos['Pod'] == p]
                        dspy = dsp.ix[dfp['Giorno'] == GetRicDate(dtf)]
                        prediction += ModifyPrediction(dspy['Pod'].values.ravel()[0], dspy[dfpy.columns[3:]].values.ravel(), meteo)
            else:
                dsp = dbsos.ix[dbsos['Pod'] == p]
                dspy = dsp.ix[dfp['Giorno'] == GetRicDate(dtf)]
                prediction += ModifyPrediction(dspy['Pod'].values.ravel()[0], dspy[dfpy.columns[3:]].values.ravel(), meteo)
    else:
        for p in not_dailies:
            if dbpdo.shape[0] > 0 and p in dbpdo['Pod'].values.ravel().list():
                dfp = dbpdo.ix[dbpdo['Pod'] == p]
                dfpy = dfp.ix[dfp['Giorno'] == GetRicHoliday(dtf)]
                if dfpy.shape[0] > 0:
                    prediction += ModifyPrediction(dfpy['Pod'].values.ravel()[0], dfpy[dfpy.columns[3:]].values.ravel(), meteo)
                else:
                    dsp = dbsos.ix[dbsos['Pod'] == p]
                    dspy = dsp.ix[dfp['Giorno'] == GetRicHoliday(dtf)]
                    prediction += ModifyPrediction(dspy['Pod'].values.ravel()[0], dspy[dfpy.columns[3:]].values.ravel(), meteo)
            else:
                dsp = dbsos.ix[dbsos['Pod'] == p]
                dspy = dsp.ix[dfp['Giorno'] == GetRicHoliday(dtf)]
                prediction += ModifyPrediction(dspy['Pod'].values.ravel()[0], dspy[dfpy.columns[3:]].values.ravel(), meteo)
    
    return prediction
####################################################################################################
def Mapper(n):
    A = [0]
    B = [1,2,3,4]
    C = [5]
    D = [6]
        
    if n == 0:
        return A
    elif n == 1:
        return B
    elif n == 2:
        return B
    elif n == 3:
        return B
    elif n == 4:
        return B
    elif n == 5:
        return C
    else:
        return D
####################################################################################################
#def EstimateUnknownCurve(db, p, crpp, d):
#### @BRIEF: funtion to estimate a curve if the pod is new and no history is available yet.
#### There are two cases: I) p is from the sample; II) p is not from the sample.
#    ### case I)
#    A = [0]
#    B = [1,2,3,4]
#    C = [5]
#    D = [6]
#    DAYS = [A, B, C, D]
#    pred = [p, d]
#    tdow = d.weekday()
#    if AddHolidaysDate(d) == 0:
#        if p in list(set(db['POD'].values.ravel())):
#            dbp = db.ix[db['POD'] == p]
#            dow = map(lambda date: date.weekday(), dbp['Giorno'].values.ravel())
#            if list(set(dow)) == [0,1,2,3,4,5,6]:
#                dbpt = dbp.ix[dbp['Giorno'].weekday() == tdow]
#                prop = dbpt.ix[dbpt['Giorno'] == max(dbpt['Giorno'])]
#                if AddHolidatsDate(prop.ix['Giorno']) == 0 and not prop.ix['Giorno'].month == 8:
#                    pred.extend(prop.values.ravel()[3:])
#                elif AddHolidatsDate(prop.ix['Giorno']) == 1 and not prop.ix['Giorno'].month == 8:
#                    prop = dbpt.ix[dbpt['Giorno'] == max(dbpt['Giorno']) - datetime.timedelta(days = 7)]
#                elif AddHolidatsDate(prop.ix['Giorno']) == 0 and prop.ix['Giorno'].month == 8:
#                    ### incremento medio
#                
#                else:
#                    ### incremento medio su
#                    prop = dbpt.ix[dbpt['Giorno'] == max(dbpt['Giorno']) - datetime.timedelta(days = 7)]
#            else:
#                ### "shape estimator"
#                L = Mapper(tdow)
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
def CurveEstimator(pod, d, crpp, db, zona):
    #dm = d.month
    #strm = str(dm) if len(str(dm)) > 1 else "0" + str(dm)
    if pod in list(set(db['POD'].values.ravel())):
        npl = 'S'
    else:
        npl = 'OOS'
    
    ricals = pd.read_excel('C:/Users/utente/Documents/Sbilanciamento/shape_dataset_' + zona + '_' + npl + '.xlsx')
    tot = crpp['CONSUMO_TOT'].ix[crpp['POD'] == pod].values.ravel()
    prop = tot * ricals.ix[ricals.index.date == d]
    
    return prop
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
def PredictOOS_Base(db, zona, dts):
    ### @PARAM: crpp is the "classic" crpp and dts could be a vector of dates
    if not isinstance(dts, list):
        dts = [dts]
    #sos = pd.read_hdf("C:/Users/utente/Documents/Sbilanciamento/sos_elaborati_finiti.h5")
    pdo = pd.read_hdf("C:/Users/utente/Documents/Sbilanciamento/DB_misure.h5")
    sos = pd.read_excel("C:/Users/utente/Documents/Sbilanciamento/sos_elaborati_finiti.xlsx")
#    pdo = pd.read_excel("C:/Users/utente/Documents/Sbilanciamento/DB_misure.xlsx")    
    rical = pd.read_excel("C:/Users/utente/Documents/Sbilanciamento/Rical.xlsx", sheetname = zona) 
    
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
            pred = Taker(rical, pdo, sos, p, d)
            
            if len(pred) < 26:                
                print '#pod not found: {}'.format(missing_counter)
                missing_pod.append((p,d))
                pred = EstimateUnknownCurve(rical, db, p, crpp, d)
            
            PRED[counter] = pred
            counter += 1
    
    PRED = pd.DataFrame.from_dict(PRED, orient = 'index')
    PRED.columns = ['POD', 'Giorno', '1','2','3', '4', '5', '6', '7', '8', '9', '10',
                '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24']    
    Pred = PRED.fillna(0)
    return Pred
####################################################################################################
def PredictS_Base(db, zona, dts):
    ### @PARAM: crpp is the "classic" crpp and dts could be a vector of dates
    if not isinstance(dts, list):
        dts = [dts]
    sos = pd.read_hdf("C:/Users/utente/Documents/Sbilanciamento/sos_elaborati_finiti.h5")
    pdo = pd.read_hdf("C:/Users/utente/Documents/Sbilanciamento/DB_misure.h5")
    rical = pd.read_excel("C:/Users/utente/Documents/Sbilanciamento/Rical.xlsx", sheetname = zona)    
    
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
def PredictS_BaseWithout(db, zona, dts):
    ### @PARAM: crpp is the "classic" crpp and dts could be a vector of dates
    if not isinstance(dts, list):
        dts = [dts]
    sos = pd.read_hdf("C:/Users/utente/Documents/Sbilanciamento/sos_elaborati_finiti.h5")
    pdo = pd.read_hdf("C:/Users/utente/Documents/Sbilanciamento/DB_misure.h5")
    rical = pd.read_excel("C:/Users/utente/Documents/Sbilanciamento/Rical.xlsx", sheetname = zona)    
    
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
                            PREDW[mp] = pred
                            break
                else:
                    ud = [x for x in dbp['Giorno'].values.ravel() if x.weekday() == dow]
                    pred.extend(dbp[dbp.columns[3:]].ix[dbp['Giorno'] == ud[-1]].values.ravel().tolist())
                    PREDW[mp] = pred
                    break
    
    PREDW = pd.DataFrame.from_dict(PREDW, orient = 'index')    
    PREDW.columns = ['POD','Giorno', '1','2','3', '4', '5', '6', '7', '8', '9', '10',
                '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24']
                
    PREDW = PREDW.fillna(0)
    return PREDW
####################################################################################################
def Get_SLYP(db, zona, dts): ### get sample last year prediction
    ### @PARAM: crpp is the "classic" crpp and dts could be a vector of dates
    if not isinstance(dts, list):
        dts = [dts]
    sos = pd.read_hdf("C:/Users/utente/Documents/Sbilanciamento/sos_elaborati_finiti.h5")
    pdo = pd.read_hdf("C:/Users/utente/Documents/Sbilanciamento/DB_misure.h5")
    
    sos['Giorno'] = pd.to_datetime(sos['Giorno'].apply(lambda date: datetime.datetime.strptime(date, '%Y-%m-%d')).values.ravel().tolist()).date   
    pdo['Giorno'] = pd.to_datetime(pdo['Giorno'].apply(lambda date: datetime.datetime.strptime(date, '%Y-%m-%d')).values.ravel().tolist()).date

    
    db = db.ix[db['Area'] == zona]
    PRED = OrderedDict()        
    
    missing_counter = 0
    missing_pod = []
    for d in dts:
        #ricd = GetRicDate(d)
        podlist = list(set(db['POD'].ix[db['Giorno'] == max(db['Giorno'])]))
        for p in podlist:
            pdop = pdo.ix[pdo['POD'] == p]
            sosp = sos.ix[sos['Pod'] == p]
            pred = [d.date()]
            if pdop.shape[0] > 0:
                print 'taken from PDO'
                pdopd = pdop.ix[pdop['Giorno'] == d.date()]
                if pdopd.shape[0] > 0:
                    pred.extend(pdopd[pdopd.columns[4:]].values.ravel().tolist())
            else:
                print 'taken from SOS'                
                sospd = sosp.ix[sosp['Giorno'] == d.date()]
                if sospd.shape[0] > 0:
                    pred.extend(sospd[sospd.columns[2:]].values.ravel().tolist())
                else:
                    print 'not found'                    
                    missing_counter += 1
                    missing_pod.append(p)
            PRED[p] = pred
        
    print '#pod not found: {}'.format(missing_counter)
    
    
    PRED = pd.DataFrame.from_dict(PRED, orient = 'index')    
    PRED.columns = ['Giorno', '1','2','3', '4', '5', '6', '7', '8', '9', '10',
                '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24']
    return PRED
####################################################################################################
def Get_Cluster(db, zona):
    
    db = db.ix[db['Area'] == zona]
    db = db.set_index(db['Giorno'])
    counts = db['POD'].resample('D').count()
    dbagg = (0.001)*(db.resample('D').sum())
    dbagg['DOW'] = map(lambda date: date.weekday(), dbagg.index)
    return dbagg, counts
####################################################################################################
def Adjust_MPrediction(db, dtf, zona):
### @BRIEF: adjust the martingale part base prediction. Given the sample, compare it with the last days measurement
### if the behaviour is sufficiently different, modify it to follow the last trend   
    PREDW = PredictS_BaseWithout(db, zona, dtf)    
    tau = 0.15 ### max level of tolerated sbilanciamento
    tau2 = 1 ### 'tolerance multiplier'
    m = scipy.stats.mode(PRED['Giorno'].values.ravel())[0][0]
    Sample = Get_SampleAsTS(db, zona)
    sample =  pd.read_excel('C:/Users/utente/Documents/Sbilanciamento/forecast_campione_' + zona + '.xlsx')
    
    ### CHECK: variation of the process
    #### if I need to modify, the "L2" part identifies by how much the process needs to be modified,
    #### the "H1" part identifies where the process needs to be modified

    md = min([max(sample.index.date), max(Sample.index.date)])
    
    #daysdifference = (dtf - max(Sample.index.date)).days
    
#    sampled = sample.ix[sample.index.date == md].values.ravel()    
    sampled = sample.values.ravel()[-24:]
    Sampled = Sample.ix[Sample.index.date == md].values.ravel()
    
    error = sampled - Sampled
    perror = error/Sampled
    
    db2tt, n = Get_Cluster(db, zona)
    
    if isinstance(dtf, list):
        dtf = dtf[0].date()
    
    dtw = dtf.weekday()
    MW = 0 < dtw < 5
    #### if dtf is a holiday --> just correct based on weather
    if AddHolidaysDate(dtf) == 0 and MW:
        db2tt = db2tt[db2tt.columns[:24]].ix[db2tt['DOW'] == dtw]
        db2tt = db2tt.ix[-7:] ### 7 days back from the peak of autocorrelation
        fcons = db2tt[db2tt.columns[:24]].mean().values.ravel() 
        fdiff = np.diff(fcons)
        fstd = db2tt[db2tt.columns[:24]].std().values.ravel()
        sbil_mean = fstd/fcons ### mean sbil of a standard deviation from the mean
        ### How is the measured sbil compared to sbil_mean?
        alpha = np.repeat(0.0, 24)
        for i in range(24):
            TAU = min(abs(sbil_mean[i]), tau)
            if perror[i] > TAU:
                print tau2*float((1 + TAU)*Sampled[i]/sampled[i])
                alpha[i] += tau2*float((1 + TAU)*Sampled[i]/sampled[i])
            elif perror[i] < -TAU:
                print tau2*float((1 - TAU)*Sampled[i]/sampled[i])
                alpha[i] += tau2*float((1 - TAU)*Sampled[i]/sampled[i])
            else:
                alpha[i] += tau2
                
        proposal = PREDW.sum().values.ravel() * alpha #* min( [1 - (daysdifference - 1)*0.2, 1])                                     
        Hsm = np.sqrt(np.mean(fdiff/np.trapz(fdiff) - np.diff(proposal)/np.trapz(np.diff(proposal)))**2)
        print Hsm        
        if Hsm > 0.25:
            print 'the shape of the proposed forecast is quite different from the one deduced from the data'
        
        return proposal
    else:
        return PREDW.sum().values.ravel()        
####################################################################################################
def WeatherAdaptedProcess(dtf, proposal, meteo, zona, what = 'OOS'):
    
    dtf = pd.date_range(dtf, dtf + datetime.timedelta(days = 1), freq = 'H').to_pydatetime()[:24]
    vtf = DependencyWithTemperature(dtf, meteo)
    
    if what == 'SAMPLE':
        model = joblib.load('C:/Users/utente/Documents/Sbilanciamento/model_weather_SAMPLE_' + zona + '.pkl')
    else:
        model = joblib.load('C:/Users/utente/Documents/Sbilanciamento/model_weather_OOS_' + zona + '.pkl')
    
    predicted_mean = model.predict(vtf)
    proposed_mean = np.mean(proposal)
    
    if abs(proposed_mean) > (1.15)*predicted_mean: ### TO MODEL ###
        print 'proposed correction due to temperature: {}'.format(predicted_mean - proposed_mean)
        proposal += predicted_mean - proposed_mean
    
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
        yf = pd.DataFrame({zona: oosh + wm * sh.values.ravel()}).set_index(pd.date_range(d, d + datetime.timedelta(days = 1), freq = 'H')[:24])
        
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
def Saver(Pred, PRED, PREDW, zona, dtf, wm, proposal):
    Pred = Pred.set_index(pd.to_datetime(Pred['Giorno']))
    oosh = Pred.resample('D').sum().values.ravel()/1000
    PRED = PRED.set_index(pd.to_datetime(PRED['Giorno']))
    sh = PRED.resample('D').sum().values.ravel()/1000
    PREDW = PREDW.set_index(pd.to_datetime(PREDW['Giorno']))
    shw = PREDW.resample('D').sum().values.ravel()/1000    
    shw = proposal/1000
    dmin = min(Pred.index)
    dmax = max(Pred.index)
    d = (dmax - dmin).days
    
    shw = pd.DataFrame({zona: shw}).set_index(pd.date_range(dmin, dmax + datetime.timedelta(days = 1), freq = 'H')[:24*(d+1)])
    sh = pd.DataFrame({zona: sh}).set_index(pd.date_range(dmin, dmax + datetime.timedelta(days = 1), freq = 'H')[:24*(d+1)])
    yf = pd.DataFrame({zona: oosh + sh.values.ravel()}).set_index(pd.date_range(dmin, dmax + datetime.timedelta(days = 1), freq = 'H')[:24*(d+1)])
    yf = pd.DataFrame({zona: oosh + wm*shw.values.ravel()}).set_index(pd.date_range(dmin, dmax + datetime.timedelta(days = 1), freq = 'H')[:24*(d+1)])
    yf.to_excel('C:/Users/utente/Documents/Sbilanciamento/forecast_' + zona + '_fino_' + str(dmax.date()) + '.xlsx')
    sh.to_excel('C:/Users/utente/Documents/Sbilanciamento/forecast_campione_' + zona + '_fino' + str(dmax.date()) + '.xlsx')   

    
    shw.to_excel('C:/Users/utente/Documents/Sbilanciamento/forecast_campione_SP_' + zona + '_fino_' + str(dmax.date()) + '.xlsx')   
        
    yf.to_excel('C:/Users/utente/Documents/Sbilanciamento/forecast_' + zona + '_' + str(dtf) + '.xlsx')
    sh.to_excel('C:/Users/utente/Documents/Sbilanciamento/forecast_campione_' + zona + '_' + str(dtf) + '.xlsx')   
    sample = pd.read_excel('C:/Users/utente/Documents/Sbilanciamento/forecast_campione_' + zona + '.xlsx')
    sample = sample.append(shw)
    sample.to_excel('C:/Users/utente/Documents/Sbilanciamento/forecast_campione_' + zona + '.xlsx')
####################################################################################################
def Get_SobolevSemiNorm(db, zona, dow):
    db = db.ix[db["Area"] == zona]
    db['DOW'] = map(lambda date:date.weekday(), db['Giorno'])
    db = db.fillna(0)
    db = db.set_index(db['Giorno'])
    dbd = db.ix[db['DOW'] == dow]
    dbdr = db.resample('D').mean().dropna().reset_index(drop = True)
    meancurve = dbd[dbd.columns[3:-1]].mean().values.ravel() #### guarda questo che è sbagliato -- non è media giorno giusto
    HSM = []
    for i in dbdr.index:
        hsm = np.sqrt(np.mean(np.diff(meancurve)/np.trapz(np.diff(meancurve)) - np.diff(dbdr[dbd.columns[3:-1]].ix[i])/np.trapz(np.diff(dbdr[dbd.columns[3:-1]].ix[i])))**2)
        HSM.append(hsm)
    print np.mean(HSM)
    return HSM
####################################################################################################
def Get_SobolevNorm(db, zona, dow):
    db = db.ix[db["Area"] == zona]
    db['DOW'] = map(lambda date:date.weekday(), db['Giorno'])
    db = db.fillna(0)
    db = db.set_index(db['Giorno'])
    dbd = db.ix[db['DOW'] == dow]
    dbdr = db.resample('D').mean().dropna().reset_index(drop = True)
    meancurve = dbd[dbd.columns[3:-1]].mean().values.ravel() #### guarda questo che è sbagliato -- non è media giorno giusto
    HSM = []
    for i in dbdr.index:
        hsm = np.sqrt(np.mean(np.diff(meancurve)/np.trapz(np.diff(meancurve)) - np.diff(dbdr[dbd.columns[3:-1]].ix[i])/np.trapz(np.diff(dbdr[dbd.columns[3:-1]].ix[i])))**2)
        hsmL = np.sqrt(np.mean(meancurve/np.trapz(meancurve) - dbdr[dbd.columns[3:-1]].ix[i]/np.trapz(dbdr[dbd.columns[3:-1]].ix[i]))**2)
        HSM.append(hsmL + hsm)
    print np.mean(HSM)
    return HSM
####################################################################################################
def GetPast(p, sos, pdo, d):

    pred = []    
    pdop = pdo.ix[pdo['POD'] == p]
    sosp = sos.ix[sos['Pod'] == p]
    sosprd = sosp.ix[sosp['Giorno'] == GetRicDate(d.date())]         
         
    if pdop.shape[0] > 0:
        pdopd = pdop.ix[pdo['Giorno'] == d.date()]
        if pdopd.shape[0] > 0:
            val = pdopd[pdopd.columns[4:]].ix[max(pdopd.index)].values.ravel()
            pred.extend(val.tolist())
                  
    elif pdop.shape[0] == 0 and sosp.shape[0] > 0:
        sospd = sosp.ix[sosp['Giorno'] == d.date()]
        if sospd.shape[0] > 0:
            val = sospd[sospd.columns[2:]].values.ravel()
            pred.extend(val.tolist())

    elif pdop.shape[0] == 0 and sosp.shape[0] == 0 and sosprd.shape[0] > 0:
        val = sosprd[sosprd.columns[2:]].values.ravel()
        pred.extend(val.tolist())

    else:
        print 'nothing found'
    
    return pred
####################################################################################################
def GetPastSample(db, sos, pdo, current_month, di, df, zona):
    
    PP = []
    missing = 0
    strm = str(current_month) if len(str(current_month)) > 1 else "0" + str(current_month)
    crpp = pd.read_excel('H:/Energy Management/02. EDM/01. MISURE/4. CRPP/2017/' + strm + '-2017/_All_CRPP_' + strm + '_2017.xlsx')
    podcrpp = set(crpp['POD'].ix[crpp['ZONA'] == zona])    
    db = db.ix[db['Area'] == zona]
    podlist = list(set(db['POD'].ix[db['Giorno'] == max(db['Giorno'])]).intersection(podcrpp))
    date_list = pd.date_range(di, df, freq = 'D')
    for d in date_list:
        locp = np.repeat(0.0,24)
        for p in podlist:        
            locv = GetPast(p, sos, pdo, d)
            if len(locv) > 0:
                locp += np.array(locv)
            else:
                missing += 1
        PP.extend(locp.tolist())
    
    PP = pd.DataFrame({zona: PP}).set_index(pd.date_range(di, '2018-01-01', freq = 'H')[:len(PP)])
    return PP/1000
####################################################################################################
def GetPastSample2(db, rical, sos, pdo, som, year, zona):
    
    current_month = datetime.datetime.now().month
    PP = pd.DataFrame()
    missing = []
    strm = str(current_month) if len(str(current_month)) > 1 else "0" + str(current_month)
    crpp = pd.read_excel('H:/Energy Management/02. EDM/01. MISURE/4. CRPP/2017/' + strm + '-2017/_All_CRPP_' + strm + '_2017.xlsx')
    podcrpp = set(crpp['POD'].ix[crpp['ZONA'] == zona])    
    db = db.ix[db['Area'] == zona]
    podlist = list(set(db['POD'].ix[db['Giorno'] == max(db['Giorno'])]).intersection(podcrpp))
    kc = [0,1,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27]

    ricpod = []    
    if len(som) == 12:
        for p in podlist:
            if p in rical.columns:
                ricpod.append(p)
    
    cols = ['Giorno', 'Ora']
    cols.extend(ricpod)
    RIC = rical[cols]    
    
    podlist = list(set(podlist).difference(set(ricpod)))
    for s in som:
        for p in podlist:        
            pdop = pdo.ix[pdo['POD'] == p]
            pdop = pdop.reset_index(drop = True)
            pdopm = pdop.ix[np.where(np.array( map(lambda date: date.month, pdop['Giorno'].values.ravel()) ) == s)]
            pdopm = pdopm.reset_index(drop = True)
            pdopmy = pdopm.ix[np.where(np.array( map(lambda date: date.year, pdopm['Giorno'].values.ravel()) ) == year)]
            pdopmy = pdopmy.drop_duplicates(subset = ['POD', 'Giorno'], keep = 'last')
            pdopmy = pdopmy[pdopmy.columns[kc]]
            if pdopmy.shape[0] > 0:            
                PP = PP.append(pdopmy, ignore_index = True)
            else:
                sosp = sos.ix[sos['Pod'] == p]
                sosp = sosp.reset_index(drop = True)
                sospm = sosp.ix[np.where(np.array( map(lambda date: date.month, sosp['Giorno'].values.ravel())) == s)]
                sospm = sospm.reset_index(drop = True)
                #sospmy = sospm.ix[np.where(np.array( map(lambda date: date.year, sospm['Giorno'].values.ravel())) == year)]
                sospm = sospm.drop_duplicates(subset = ['Pod', 'Giorno'], keep = 'last')
                #sospm = sospm[sospm.columns[kc]]
                if sospm.shape[0] > 0:
                    PP = PP.append(sospm, ignore_index = True)
                else:
                    missing.append((p, s))
    
    if PP.shape[0] > 0:    
        PP = PP.set_index(pd.to_datetime(PP['Giorno']))
        PP = PP.resample('D').sum()/1000    
        print 'missing {} PODs'.format(len(missing))       
        return PP, missing, RIC
        
    else:
        return RIC, missing
####################################################################################################
def GetPastSample3(db, sos, pdo, som, year, zona):
    
    current_month = datetime.datetime.now().month
    PP = pd.DataFrame()
    missing = []
    strm = str(current_month) if len(str(current_month)) > 1 else "0" + str(current_month)
    crpp = pd.read_excel('H:/Energy Management/02. EDM/01. MISURE/4. CRPP/2017/' + strm + '-2017/_All_CRPP_' + strm + '_2017.xlsx')
    podcrpp = set(crpp['POD'].ix[crpp['ZONA'] == zona])    
    db = db.ix[db['Area'] == zona]
    podlist = list(set(db['POD'].ix[db['Giorno'] == max(db['Giorno'])]).intersection(podcrpp))
    kc = [0,1,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27]

    for s in som:
        for p in podlist:        
            pdop = pdo.ix[pdo['POD'] == p]
            pdop = pdop.reset_index(drop = True)
            pdopm = pdop.ix[np.where(np.array( map(lambda date: date.month, pdop['Giorno'].values.ravel()) ) == s)]
            pdopm = pdopm.reset_index(drop = True)
            pdopmy = pdopm.ix[np.where(np.array( map(lambda date: date.year, pdopm['Giorno'].values.ravel()) ) == year)]
            pdopmy = pdopmy.drop_duplicates(subset = ['POD', 'Giorno'], keep = 'last')
            pdopmy = pdopmy[pdopmy.columns[kc]]
            if pdopmy.shape[0] > 0:            
                PP = PP.append(pdopmy, ignore_index = True)
            else:
                sosp = sos.ix[sos['Pod'] == p]
                sosp = sosp.reset_index(drop = True)
                sospm = sosp.ix[np.where(np.array( map(lambda date: date.month, sosp['Giorno'].values.ravel())) == s)]
                sospm = sospm.reset_index(drop = True)
                #sospmy = sospm.ix[np.where(np.array( map(lambda date: date.year, sospm['Giorno'].values.ravel())) == year)]
                sospm = sospm.drop_duplicates(subset = ['Pod', 'Giorno'], keep = 'last')
                #sospm = sospm[sospm.columns[kc]]
                if sospm.shape[0] > 0:
                    PP = PP.append(sospm, ignore_index = True)
                else:
                    missing.append((p, s))
       

    PP = PP.set_index(pd.to_datetime(PP['Giorno']))
    PP = PP.resample('D').sum()/1000    
    PP = PP[['1','2','3','4','5','6','7','8','9','10','11','12',
             '13','14','15','16','17','18','19','20','21','22','23','24']]
    print 'missing {} PODs'.format(len(missing))       
    return PP, missing        
####################################################################################################
def ToTS_fr(df):
    df = df.reset_index(drop = True)
    ts = []
    for i in range(df.shape[0]):
        y = df[df.columns[-24:]].ix[i].values.ravel().tolist()
        ts.extend(y)
    DF = pd.DataFrame({'X': ts})                            
    return DF
####################################################################################################
def InferMissingDays(df2, zona, crpp, db, rid2):
#    year = df2.Giorno[0].year
#    month = df2.Giorno[0].month
    pod = df2['POD'].ix[0]
    #new_dates = pd.to_datetime(pd.date_range(datetime.date(year, month, 1), datetime.date(year, month, calendar.monthrange(year, month)[1]), freq = 'D'))
    missing = [r for r in rid2 if not r in df2.Giorno.values.ravel().tolist()]
### REMOVE SET --> USE A FOR LOOP, OTHERWOISE A LOOSE THE ORDER OF THE RECALENDARIZATION    
    for md in missing:
        #prop = CurveEstimator(pod, md, crpp, db, zona)
        prop = np.repeat(0.0,24)
        add = np.concatenate((np.array([pod, md]), prop))
        add = pd.DataFrame(add).T
        add.columns = [['POD', 'Giorno', '1','2','3', '4', '5', '6', '7', '8', '9', '10',
                '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24']]
        df2 = df2.append(add)
    df2 = df2.reset_index(drop = True)
    n_i = indexer(df2, rid2)
    df2 = df2.reindex(n_i)
    return df2
####################################################################################################
def indexer(df, rid2):

    if len(rid2) == df.shape[0]:
        new_index = map(lambda x: df['Giorno'].values.ravel().tolist().index(x), rid2)
    else:
        new_index = []
        for r in rid2:
            if r in df.Giorno.values.ravel().tolist():
                new_index.append(df['Giorno'].values.ravel().tolist().index(r))
    return new_index                
####################################################################################################
def Ricalendarizer(df, month, year, zona, crpp, db):
    df = df.reset_index(drop = True)
### @BRIEF: given last year measures, ricalendarize them
    new_dates = pd.date_range(datetime.date(year, month, 1), datetime.date(year, month, calendar.monthrange(year, month)[1]), freq = 'D')
    rid = map(lambda d: GetRicDate(d), new_dates)
    #rid = map(lambda d: d.date(), rid)
    rid2 = [rid[x].date() if isinstance(rid[x], pd.Timestamp) else rid[x] for x in range(len(rid))]
    
    new_index = indexer(df, rid2)    
    
    df2 = df.ix[new_index]
    if df2.shape[0] == calendar.monthrange(year, month)[1]:
        df2 = df2.set_index(new_dates)
        return df2
    else:
        df2 = InferMissingDays(df2, zona, crpp, db, rid2)
        return df2
####################################################################################################
def RicalendarizerSOS(sospm, month, year):
    
    sospm = sospm.reset_index(drop = True)
    SOS = []
    YEAR = sospm.Giorno.ix[0].year
    dy = year - YEAR
    
    zeit = pd.date_range(datetime.date(year, month, 1), datetime.date(year, month, calendar.monthrange(year, month)[1]), freq = 'D')

    
    for z in zeit:     
        counter = 0
        ricz = GetRicDate(z)        
        while counter < (dy - 1):
            ricz = GetRicDate(ricz)
            counter += 1
        ricz = ricz.date() if isinstance(ricz, pd.Timestamp) else ricz
        ric = sospm[sospm.columns[2:]].ix[sospm.Giorno == ricz].values.ravel()
        if ric.size == 25:
            v = np.repeat(0.0,24)
            for h in range(25):
                if h < 2:
                    v[h] = ric[h]
                elif h == 2:
                    v[h] = ric[2] + ric[3]
                elif h == 3:
                    next
                elif h > 3:
                    v[h-1] = ric[h]
            SOS.extend(v.tolist())
        elif ric.size == 23:
            v = np.concatenate((ric[:2], np.array(0.0), ric[2:]))
            SOS.extend(v.tolist())
        else:
            SOS.extend(ric.tolist())
    
    if len(SOS) == zeit.size*24:
        return SOS
    else:
        print 'Incomplete SOS'
        return None
####################################################################################################
def MonthlyRical(month, zona, year, db):
### @BRIEF: fundamental method to get a rical-like file given a month  
    missing = []
    sos = pd.read_excel("C:/Users/utente/Documents/Sbilanciamento/sos_elaborati_finiti.xlsx")
    pdo = pd.read_hdf("C:/Users/utente/Documents/DB_misure.h5")
    sos['Giorno'] = pd.to_datetime(sos['Giorno'].values.ravel().tolist()).date   
    pdo['Giorno'] = pd.to_datetime(pdo['Giorno'].values.ravel().tolist()).date
    strm = str(month) if len(str(month)) > 1 else "0" + str(month)
    crpp = pd.read_excel('H:/Energy Management/02. EDM/01. MISURE/4. CRPP/2017/' + strm + '-2017/_All_CRPP_' + strm + '_2017.xlsx')
    crpp = crpp.ix[crpp['ZONA'] == zona]
    crppO = crpp.ix[crpp['Trattamento_' + strm] == 'O']    
    podlist = list(set(crppO['POD'].values.ravel().tolist()))
    kc = [0,1,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27]
    RIC = pd.DataFrame()
    for p in podlist:
        pdop = pdo.ix[pdo['POD'] == p]
        pdop = pdop.reset_index(drop = True)
        pdopm = pdop.ix[np.where(np.array( map(lambda date: date.month, pdop['Giorno'].values.ravel()) ) == month)]
        pdopm = pdopm.reset_index(drop = True)
        pdopmy = pdopm.ix[np.where(np.array( map(lambda date: date.year, pdopm['Giorno'].values.ravel()) ) == (year - 1))]
        pdopmy = pdopmy.drop_duplicates(subset = ['POD', 'Giorno'], keep = 'last')
        pdopmy = pdopmy[pdopmy.columns[kc]]
        if pdopmy.shape[0] > 0:            
            pdoR = Ricalendarizer(pdopmy, month, year, zona, crpp, db)
            RIC[p] = ToTS_fr(pdoR[pdoR.columns[2:]])
        else:
            sosp = sos.ix[sos['Pod'] == p]
            sosp = sosp.reset_index(drop = True)
            sospm = sosp.ix[np.where(np.array( map(lambda date: date.month, sosp['Giorno'].values.ravel())) == month)]
            sospm = sospm.reset_index(drop = True)
            sospm = sospm.drop_duplicates(subset = ['Pod', 'Giorno'], keep = 'last')
            if sospm.shape[0] > 0:
                sosR = RicalendarizerSOS(sospm, month, year)
                if not sosR == None:
                    RIC[p] = sosR
                else:
                    missing.append(p)
    RIC = RIC.set_index(pd.date_range(datetime.date(year, month, 1), datetime.date(year, month, calendar.monthrange(year, month)[1]) + datetime.timedelta(days = 1), freq = 'H')[:RIC.shape[0]])
    RIC.to_excel('C:/Users/utente/Documents/Sbilanciamento/ric_' + strm + '_' + zona + '.xlsx')    
    return RIC, missing        
####################################################################################################
def BootstrapBase(rical, group):
    subset = np.random.randint(low = 0, high = len(group), size = int(np.ceil(0.8*len(group))))
    pl = list(set([group[x] for x in subset]).intersection(set(rical.columns)))
    ricals = rical[pl]     
    ricsum = ricals.sum(axis = 1)
    TOT = ricals.sum(axis = 1).sum()
    shape = ricsum/TOT   
    return shape.values.ravel()
####################################################################################################
def GenerateShapeOperator(rical, crpp, db, zona):
### @BRIEF: computes the SHAPE OPERATOR to estimate the totally unknown curves    
### The final SHAPE OPERATOR will be a bootstrapped mean of the subsampled shape operator    

### the indeces have to be reported to the correct year...

    z = zona
    shape_S = np.repeat(0.0, rical.shape[0])
    shape_OOS = np.repeat(0.0, rical.shape[0])
    crppz = crpp.ix[crpp['ZONA'] == z]
    dbz = db.ix[db['Area'] == z]
    S_group = list(set(dbz['POD'].values.ravel()))
    pods = set(crppz['POD'].values.ravel().tolist())
    OOS_group = list(pods.difference(set(S_group)))
    B = 1000
    for b in range(B):
        ### maybe with map?
        shape_S += BootstrapBase(rical, S_group)
        shape_OOS += BootstrapBase(rical, OOS_group)
    shape_S = shape_S/B
    shape_OOS = shape_OOS/B
    print 'done with zone {}'.format(z)
    shape_S = pd.DataFrame(shape_S)
    shape_S = shape_S.set_index(rical.index)
    shape_OOS = pd.DataFrame(shape_OOS)
    shape_OOS = shape_OOS.set_index(rical.index)
    shape_S.to_excel('C:/Users/utente/Documents/Sbilanciamento/shape_dataset_' + z + '_S.xlsx')
    shape_OOS.to_excel('C:/Users/utente/Documents/Sbilanciamento/shape_dataset_' + z + '_OOS.xlsx')
    print 'done zona {}'.format(z)
    return 1
####################################################################################################
def MahalanobisDistance(x):
    mu = np.mean(x)
    sigma = np.std(x)
    ds = []
    for i in range(x.size):
        ds.append((x[i] - mu)/sigma)
    return np.array(ds)
####################################################################################################
def GetWCorrelation(RIC):
    
    diz = OrderedDict()
    ds = sorted(list(set(map(lambda date: date.date(), RIC.index.tolist()))))
    counter = 0
    week_counter = 0
    cons = np.repeat(0.0,7)
    for d in ds:
        dow = d.weekday()
        cons[dow] = RIC.ix[RIC.index.date == d].sum()
        counter += 1
        if counter == 7:
            diz[week_counter] = cons
            week_counter += 1
            counter = 0
            cons = np.repeat(0.0,7)
    
    diz = pd.DataFrame.from_dict(diz, orient = 'index')
    diz.columns = [['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']]
    return diz
####################################################################################################
def GetDCorrelation(RIC):
    
    diz = OrderedDict()
    ds = sorted(list(set(map(lambda date: date.date(), RIC.index.tolist()))))
    for d in ds:
        diz[d] = RIC.ix[RIC.index.date == d].values.ravel()
    
    diz = pd.DataFrame.from_dict(diz, orient = 'index')
    diz.columns = [['1','2','3','4','5','6','7','8','9','10','11','12',
                    '13','14','15','16','17','18','19','20','21','22','23','24']]
    return diz
####################################################################################################
def GetMatrixFun(RIC):
    
    diz = OrderedDict()
    ds = sorted(list(set(map(lambda date: date.date(), RIC.index.tolist()))))
    for d in ds:
        res = []
        res.extend(RIC.ix[RIC.index.date == d].values.ravel().tolist())
        res.append(d.weekday())
        res.append(d.month)
        res.append(AddHolidaysDate(d))
        diz[d] = res
    
    diz = pd.DataFrame.from_dict(diz, orient = 'index')
    diz.columns = [['1','2','3','4','5','6','7','8','9','10','11','12',
                    '13','14','15','16','17','18','19','20','21','22','23','24',
                    'weekday','month','holiday']]
    return diz
####################################################################################################
def MVisualizer(WM, holiday, dow = None, month = None,):

    WM1 = WM.ix[WM.holiday == holiday]

    if dow != None:        
        WM2 = WM1.ix[WM1.weekday == dow]
    else:
        WM2 = WM1
    
    if month != None:        
        WM3 = WM2.ix[WM2.month == month]
    else:
        WM3 = WM2
    
    print 'mean: {}'.format(WM3[WM3.columns[:24]].mean())
    print 'std: {}'.format(WM3[WM3.columns[:24]].std())
    
    WM3[WM3.columns[:24]].T.plot()
    WM3[WM3.columns[:24]].plot(kind = 'box')
#    plotting.andrew_curves(WM3, 'weekday')
#################################################################################################### 
def ChaosFinder(zona, rical, B = 1000):
    
    rical = rical.set_index(pd.date_range('2017-01-01', '2018-12-31', freq = 'H')[:8760])/1000
    diz = OrderedDict()
    for b in range(B):
        res = []
        n = np.random.randint(low = 1, high = (rical.shape[1] - 2), size = 1)
        nl = np.random.randint(low = 2, high = rical.shape[1], size = n).tolist()
        remain = list(set(range(2, rical.shape[1])).difference(nl))
        res.append(len(nl))
        res.append(len(remain))
        removed_weight = rical[rical.columns[nl]].sum(axis = 1).sum()/rical[rical.columns[2:]].sum(axis = 1).sum()
        remain_weight = rical[rical.columns[remain]].sum(axis = 1).sum()/rical[rical.columns[2:]].sum(axis = 1).sum()
        corr_removed = np.mean(np.triu(rical[rical.columns[nl]].corr(), k = 1).ravel())
        corr_remain = np.mean(np.triu(rical[rical.columns[remain]].corr(), k = 1).ravel())                
        res.append(removed_weight)
        res.append(remain_weight)
        mean_removed = removed_weight/len(nl)
        mean_remain = remain_weight/len(remain)
        res.append(mean_removed)
        res.append(mean_remain)        
        rem_sigma_mean = rical[rical.columns[remain]].sum(axis = 1).resample('D').std().mean()
        removed_sigma_mean = rical[rical.columns[nl]].sum(axis = 1).resample('D').std().mean()
        res.append(rem_sigma_mean)
        res.append(removed_sigma_mean)
        res.append(corr_removed)
        res.append(corr_remain)
        diz[b] = res
    diz = pd.DataFrame.from_dict(diz, orient = 'index')
    diz.columns = [['removed', 'remain', 'weight_removed', 'weight_remaining', 'mean_weight_removed',
                    'mean_weight_remaining', 'mean_sigma_removed', 'mean_sigma_remaining', 
                    'mean_corr_removed', 'mean_corr_remain']]
    return diz