# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 12:14:09 2017

@author: utente

Sbilanciamento 9 -- Run quasi-automatically the forecast algorithm
"""

zona = ''

import Sbilanciamento10 as FF
import datetime
import pandas as pd
import matplotlib.pyplot as plt

#sos = pd.read_hdf("C:/Users/utente/Documents/Sbilanciamento/sos_elaborati_finiti.h5")
sos = pd.read_excel("C:/Users/utente/Documents/Sbilanciamento/sos_elaborati_finiti.xlsx")
pdo = pd.read_hdf("C:/Users/utente/Documents/DB_misure.h5")
rical = pd.read_excel("C:/Users/utente/Documents/Sbilanciamento/Rical.xlsx", sheetname = zona)    
    
##### Updating step #####


#########################
    
db = pd.read_excel("H:/Energy Management/02. EDM/01. MISURE/3. DISTRIBUTORI/ENEL Distribuzione S.p.A/2017/Aggregatore_orari-2017.xlsx")
db.columns = [str(i) for i in db.columns]
db = db[["POD", "Area", "Giorno", "1","2","3","4","5","6","7","8","9","10","11","12","13","14","15",
         "16","17","18","19","20","21","22","23","24"]]
#Agg = pd.read_excel("H:/Energy Management/02. EDM/01. MISURE/3. DISTRIBUTORI/A2A Milano Reti Elettriche S.p.A_12883450152/2017/Unareti_elaborati.xlsx")
#db = db.append(Agg, ignore_index = True)
db = db.drop_duplicates(subset = ['POD', 'Area', 'Giorno'], keep = 'last')

#dtf = dts = [datetime.date(2017,11,1),datetime.date(2017,11,2)]
dtf = dts = [datetime.date(2017,12,8)]


for d in dtf:
    Pred = FF.PredictOOS_Base(pdo, sos, rical, db, zona, d)
    PREDW = FF.PredictS_BaseWithout(pdo, sos, rical, db, zona, d)
    
    ad, adMW = FF.Activator(db, d, zona)    
    
    Pred = FF.PODremover(Pred, ['IT001E74733617'])
    ## Enterra:  IT001E74733617
    if ad:
        print 'weekdays correction'
        proposal = FF.Adjust_MPrediction(PREDW, db, d, zona)
    elif adMW:
        print "Wednesday's correction"
        Sample = FF.Get_SampleAsTS(db, zona)
        proposal = FF.MTCorrection(PREDW, db, d, zona, Sample)
    else:
#        PREDW = PREDW[PREDW.columns[2:]].set_index(pd.to_datetime(PREDW['Giorno']))
#        proposal = PREDW.resample('D').sum().values.ravel()
        proposal = PREDW.sum().values.ravel()/1000
    
    if not d.weekday() in [5,6]:
        proposal = FF.Adjust_Peaks(proposal, db, d, zona)
    
    plt.plot(proposal)
    plt.plot(proposal + Pred.sum().values.ravel()/1000)
    
    FF.Saver(Pred, proposal, zona, d, db)

#FF.Saver(Pred, proposal, zona, dts[0], db)
#FF.Saver(Pred, proposal, zona, dts[1], db)
#FF.Saver(Pred, proposal, zona, dts[2], db)