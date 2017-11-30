# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 10:42:00 2017

@author: utente

Sbilanciamento 7 -- OUT OF SAMPLE ERROR CONTROL MODULE --
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import os

####################################################################################################
def convertDates(vec):
    CD = vec.apply(lambda x: datetime.datetime(year = int(str(x)[6:10]), month = int(str(x)[3:5]), day = int(str(x)[:2]), hour = int(str(x)[11:13])))
    return CD
####################################################################################################
def Get_ZonalDataset(df, zona):
    df = df.ix[df["CODICE RUC"] == "UC_DP1608_" + zona]
    
    cd = convertDates(df['DATA RIFERIMENTO CORRISPETTIVO'])
    df = df.set_index(cd.values)    
    
    df = df.ix[df.index.date > datetime.date(2016,12,31)]
    dr = pd.date_range('2017-01-01', df.index.date[-1], freq = 'D')
    res = []
    for i in dr.tolist():
        dfd = df.ix[df.index.date == i.to_pydatetime().date()]
        if dfd.shape[0] == 24:
            res.extend((dfd['MO [MWh]'].values).tolist())
        else:
            for hour in range(24):
                dfdh = dfd.ix[dfd.index.hour == hour]
                if dfdh.shape[0] == 0:
                    res.append(0)
                elif dfdh.shape[0] == 2:
                    res.append(dfdh["MO [MWh]"].sum())
                else:
                    res.append(dfdh["MO [MWh]"].values[0])
        
    diz = pd.DataFrame(res)
    diz.columns = [[zona]]
    diz = diz.set_index(pd.date_range('2017-01-01', '2017-12-31', freq = 'H')[:diz.shape[0]])
    return diz
####################################################################################################
def Error_Control(zona, month, what):
### @PARAM: zona and month are self-explanatory, what = {OOS, Sample, ZONA}
    forecast = pd.read_excel('C:/Users/utente/Documents/Sbilanciamento/forecast_' + zona + '.xlsx')
    if what == 'OOS': 
        oos = pd.read_hdf('C:/Users/utente/Documents/Sbilanciamento/storico_oos_' + zona + '.h5')
        sample =  pd.read_hdf('C:/Users/utente/Documents/Sbilanciamento/forecast_campione_' + zona + '.h5', 'sample_' + zona.lower())
        foos = forecast - sample
        oosm = oos.ix[oos.index.month == month]
        foosm = foos.ix[foos.index.month == month]
        error = oosm - foosm
        den = oosm
        SETPARAM = 'Out Of Sample'
    elif what == 'Sample':
        sample =  pd.read_hdf('C:/Users/utente/Documents/Sbilanciamento/forecast_campione_' + zona + '.h5', 'sample_' + zona.lower())
        dt = pd.read_excel("C:/Users/utente/Documents/Sbilanciamento/Aggregatore_orari-2017.xlsx")
        dt.columns = [str(i) for i in dt.columns]
        dt = dt[["POD", "Area", "Giorno", "1","2","3","4","5","6","7","8","9","10","11","12","13","14","15",
                 "16","17","18","19","20","21","22","23","24"]]
        
        dt = dt.drop_duplicates(subset = ['POD', 'Area', 'Giorno'], keep = 'last')
        Sample = Get_SampleAsTS(dt, zona)
        samplem = sample.ix[sample.index.month == month]
        Samplem = Sample.ix[Sample.index.month == month]
        error = Samplem - samplem
        den = Samplem
        SETPARAM = 'sample'
    else:
        df = pd.read_excel('C:/Users/utente/Documents/misure/aggregato_sbilanciamento2.xlsx')
        
        df = Get_ZonalDataset(df, zona)        
        
        dfy = df[zona].ix[df.index.year == 2017]
        dfm = dfy.ix[dfy.index.month == month]
        error = dfm - forecast.ix[forecast.index.month == month]
        den = forecast.ix[forecast.index.month == month]
        SETPARAM = 'ZONA ' #+ zona
    print 'mean error on {}: {}'.format(SETPARAM, np.mean(error.values.ravel()))
    print 'median error on {}: {}'.format(SETPARAM, np.median(error.values.ravel()))
    print 'max error on {}: {}'.format(SETPARAM, np.max(error.values.ravel()))
    print 'standard deviation of error on {}: {}'.format(SETPARAM, np.std(error.values.ravel()))
    print 'mean absolute error on {}: {}'.format(SETPARAM, np.mean(np.abs(error.values.ravel())))
    print 'median absolute error on {}: {}'.format(SETPARAM, np.median(np.abs(error.values.ravel())))
    print 'max absolute error on {}: {}'.format(SETPARAM, np.max(np.abs(error.values.ravel())))
    print 'standard deviation of absolute error on {}: {}'.format(SETPARAM, np.std(np.abs(error.values.ravel())))
    Sbil = error/den
    AbsSbil = error.abs()/den
    plt.figure()
    plt.plot(Sbil.index, Sbil.values.ravel())
    plt.title('Sbilanciamento {} zona {} in month {}'.format(SETPARAM, zona, month))
    plt.figure()
    plt.plot(AbsSbil.index, AbsSbil.values.ravel())
    plt.title('Sbilanciamento assoluto {} zona {} in month {}'.format(SETPARAM, zona, month))
    print 'mean sbilanciamento on {}: {}'.format(SETPARAM, np.mean(Sbil.values.ravel()))
    print 'median sbilanciameto on {}: {}'.format(SETPARAM, np.median(Sbil.values.ravel()))
    print 'max sbilanciamento on {}: {}'.format(SETPARAM, np.max(Sbil.values.ravel()))
    print 'min sbilanciamento on {}: {}'.format(SETPARAM, np.min(Sbil.values.ravel()))    
    print 'standard deviation of sbilanciamento on {}: {}'.format(SETPARAM, np.std(Sbil.values.ravel()))
    print 'mean absolute sbilanciamento on {}: {}'.format(SETPARAM, np.mean(AbsSbil.values.ravel()))
    print 'median absolute sbilanciamento on {}: {}'.format(SETPARAM, np.median(AbsSbil.values.ravel()))
    print 'max absolute sbilanciamento on {}: {}'.format(SETPARAM, np.max(AbsSbil.values.ravel()))
    print 'standard deviation of absolute sbilanciamento on {}: {}'.format(SETPARAM, np.std(AbsSbil.values.ravel()))
    return 1
####################################################################################################
def Daily_Error_Control(day, zona):
    sample =  pd.read_hdf('C:/Users/utente/Documents/Sbilanciamento/forecast_campione_' + zona + '.h5', 'sample_' + zona.lower())

    dt = pd.read_excel("C:/Users/utente/Documents/Sbilanciamento/Aggregatore_orari-2017.xlsx")
    dt.columns = [str(i) for i in dt.columns]
    dt = dt[["POD", "Area", "Giorno", "1","2","3","4","5","6","7","8","9","10","11","12","13","14","15",
             "16","17","18","19","20","21","22","23","24"]]
        
    dt = dt.drop_duplicates(subset = ['POD', 'Area', 'Giorno'], keep = 'last')

    Sample = Get_SampleAsTS(dt, zona)
    
    sampled = sample.ix[sample.index.date == day]
    Sampled = Sample.ix[Sample.index.date == day]
    SETPARAM = 'Sample'
    error = Sampled - sampled
    Sbil = error/Sampled
    Asbil = error.abs()/Sampled
    print 'mean error on {}: {}'.format(SETPARAM, np.mean(error.values.ravel()))
    print 'median error on {}: {}'.format(SETPARAM, np.median(error.values.ravel()))
    print 'max error on {}: {}'.format(SETPARAM, np.max(error.values.ravel()))
    print 'standard deviation of error on {}: {}'.format(SETPARAM, np.std(error.values.ravel()))
    print 'mean absolute error on {}: {}'.format(SETPARAM, np.mean(np.abs(error.values.ravel())))
    print 'median absolute error on {}: {}'.format(SETPARAM, np.median(np.abs(error.values.ravel())))
    print 'max absolute error on {}: {}'.format(SETPARAM, np.max(np.abs(error.values.ravel())))
    print 'standard deviation of absolute error on {}: {}'.format(SETPARAM, np.std(np.abs(error.values.ravel())))
    plt.figure()
    plt.plot(Sbil.index, Sbil.values.ravel())
    plt.title('Sbilanciamento {} zona {} in day {}'.format(SETPARAM, zona, day))
    plt.figure()
    plt.plot(Asbil.index, Asbil.values.ravel())
    plt.title('Sbilanciamento assoluto {} zona {} in day {}'.format(SETPARAM, zona, day))
    print 'mean sbilanciamento on {}: {}'.format(SETPARAM, np.mean(Sbil.values.ravel()))
    print 'median sbilanciameto on {}: {}'.format(SETPARAM, np.median(Sbil.values.ravel()))
    print 'max sbilanciamento on {}: {}'.format(SETPARAM, np.max(Sbil.values.ravel()))
    print 'min sbilanciamento on {}: {}'.format(SETPARAM, np.min(Sbil.values.ravel()))    
    print 'standard deviation of sbilanciamento on {}: {}'.format(SETPARAM, np.std(Sbil.values.ravel()))
    print 'mean absolute sbilanciamento on {}: {}'.format(SETPARAM, np.mean(Asbil.values.ravel()))
    print 'median absolute sbilanciamento on {}: {}'.format(SETPARAM, np.median(Asbil.values.ravel()))
    print 'max absolute sbilanciamento on {}: {}'.format(SETPARAM, np.max(Asbil.values.ravel()))
    print 'standard deviation of absolute sbilanciamento on {}: {}'.format(SETPARAM, np.std(Asbil.values.ravel()))
    return 1
####################################################################################################
def PDOs_To_TS(pdo, zona):
    pdo["Giorno"] = pd.to_datetime(pdo["Giorno"])
    pdo = pdo.ix[pdo["zona"] == zona]
    dr = pd.date_range(min(pdo["Giorno"].values.ravel()), max(pdo["Giorno"].values.ravel()), freq = 'D')
    res = []
    for i in dr.tolist():
        dbd = pdo[pdo.columns[4:]].ix[pdo["Giorno"] == i].sum()/1000
        res.extend(dbd.values.tolist())
        diz = pd.DataFrame(res)
    diz.columns = [['MO [MWh]']]
    diz = diz.set_index(pd.date_range(min(pdo["Giorno"].values.ravel()), '2017-12-31', freq = 'H')[:diz.shape[0]])
    return diz    
####################################################################################################
def Terna_vs_PDOs(terna, pdo, zona):
    pdots = PDOs_To_TS(pdo, zona)
#    common_indeces = list(set(terna.index).intersection(set(pdots.index)))
    error = terna['MO [MWh]'].ix[terna.index.year > 2016].values.ravel() - pdots.ix[pdots.index.year > 2016].values.ravel()
    return error
####################################################################################################
def ModelComparison(forecast1, forecast2, terna):
    forecast1 = forecast1.ix[terna.index]
    forecast2 = forecast2.ix[terna.index]
    pred_error1 = terna['MO [MWh]'] - forecast1
    pred_error2 = terna['MO [MWh]'] - forecast2
    between_error = forecast1 - forecast2
    return pred_error1, pred_error2, between_error
####################################################################################################
#def CompareTrendToTerna(df, dtc, zona):
### @BRIEF: function to compare the trend of the required dataset to the trend given by Terna.
### @PARAM: df is the dataset from Terna (assumed to be already a time series with the proper correct time index)
### dtc is the dates to compare (it could be OOS or S)
    #df.loc[datetime.datetime()]
####################################################################################################
def PDOTaker(pdo, p, d):
    if p in pdo['POD'].values.ravel().tolist():
        pdop = pdo.ix[pdo['POD'] == p]
        pdopd = pdop.ix[pdop['Giorno'] == d]
        if pdopd.shape[0] > 0:
            return pdopd[pdopd.columns[4:]].values.ravel()
        else:
            print '** POD not in PDO **'
            return np.repeat(0,24)
####################################################################################################
def CompareTrueSample(db, zona):
### @BRIEF: this function compares the predicted sample to the correct measurments of the same PODs actually predicted    
    Sample = Get_SampleAsTS(db, zona)
    sample = pd.read_excel('C:/Users/utente/Documents/Sbilanciamento/forecast_campione_' + zona + '.xlsx')
    pdo = pd.read_hdf("C:/Users/utente/Documents/Sbilanciamento/DB_misure.h5")
    list_of_dates = list(set(map(lambda date: date.date(), sample.index)))
    meas = []
    for d in list_of_dates:
        AM = np.repeat(0,24)
        sad = SampleAtDay(db, d, zona)
        for p in sad:
            predpdop = PDOTaker(pdo, p, d)
            AM += predpdop/1000
        meas.extend(AM.tolist())
    Meas = pd.DataFrame({zona: meas}).set_index(sample.index)
    return Meas
####################################################################################################
def ToTS(df):
    ts = []
    dmin = df['Giorno'].min() 
    dmax = df['Giorno'].max()
    for i in df.index:
        y = df[df.columns[-24:]].ix[i].values.ravel().tolist()
        ts.extend(y)
    DF = pd.DataFrame({'X': ts}).set_index(pd.date_range(dmin, dmax + datetime.timedelta(days = 1), freq = 'H')[:len(ts)])                            
    return DF
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
def getSbilanciamentoPOD(pod, rical, zona):
### @BREIF: returns the actual sbilanciamento given a POD
    dl = setRicalIndex(rical)
    val = rical[pod].values.ravel()
    RP = pd.DataFrame({'X': val}).set_index(pd.date_range(dl[0].date(), dl[-1].date() + datetime.timedelta(days = 1), freq = 'H')[:len(dl)])
    pdo = pd.read_hdf("C:/Users/utente/Documents/Sbilanciamento/DB_misure.h5")
    pdop = pdo.ix[pdo['POD'] == pod]
    pdop['Giorno'] = pdop['Giorno'].apply(lambda s: datetime.datetime.strptime(s, '%Y-%m-%d').date())
    pdop = pdop.ix[pdop['Giorno'] > datetime.date(2016,12,31)]
    pdop = pdop.drop_duplicates(subset = ['Giorno'], keep = 'last')
    TS = ToTS(pdop)
    ricalp = RP.ix[TS.index]
    SBIL = (TS.values.ravel() - ricalp.values.ravel())/TS.values.ravel()
    SBIL = pd.DataFrame({'Sbilanciamento_' + pod: SBIL}).set_index(TS.index)
    return SBIL