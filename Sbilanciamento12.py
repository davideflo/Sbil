# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 16:21:41 2017

@author: utente

Sbilanciamento 12 -- comparisons
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import Sbilanciamento10 as FF
import datetime
import statsmodels.api
from collections import OrderedDict


####################################################################################################
def SampleAtDay(db, dtd, zona):
    db = db.ix[db["Area"] == zona]
    return list(set(db["POD"].ix[db["Giorno"] == dtd].tolist()))
####################################################################################################
#def GetTrueSample(db, zona, rical, pdo, sos):
#    db = db.ix[db["Area"] == zona]
#    sam = pd.read_excel('C:/Users/utente/Documents/Sbilanciamento/forecast_campione_' + zona + '.xlsx')
#    Sample = FF.Get_SampleAsTS(db, zona)
#    tsam = []
#    for d in db['Giorno']:
#        sad = SampleAtDay(db, d, zona)
#        sadb = SampleAtDay(db, d - datetime.timedelta(days = 1), zona)
#        to_remove = list(set(sad).symmetric_difference(set(sadb)))
#        PRED = OrderedDict()     
#        counter = 0
#        if len(to_remove) == 0:
#            v = sam.ix[sam.index.date == d].values.ravel()
#        else:
#            if p in sad:
#                for p in to_remove:
#                    pred = FF.Taker(rical, pdo, sos, p, d)
#                    PRED[counter] = pred
#                    counter += 1
#                PRED = pd.DataFrame.from_dict(PRED, orient = 'index')
#                PRED.columns = ['POD', 'Giorno', '1','2','3', '4', '5', '6', '7', '8', '9', '10',
#                                '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24']    
#                PRED = PRED.fillna(0)
#                TR = PRED[PRED.columns[2:]].sum().values.ravel()/1000
#                v = Sample.ix[Sample.index.date == d].values.ravel() - TR
            
        

db = pd.read_excel("H:/Energy Management/02. EDM/01. MISURE/3. DISTRIBUTORI/ENEL Distribuzione S.p.A/2017/Aggregatore_orari-2017.xlsx")
db.columns = [str(i) for i in db.columns]
db = db[["POD", "Area", "Giorno", "1","2","3","4","5","6","7","8","9","10","11","12","13","14","15",
         "16","17","18","19","20","21","22","23","24"]]
db = db.drop_duplicates(subset = ['POD', 'Area', 'Giorno'], keep = 'last')

zona = 'NORD'

Sample = FF.Get_SampleAsTS(db, zona)
sample = pd.read_excel('C:/Users/utente/Documents/Sbilanciamento/forecast_campione_' + zona + '.xlsx')
nsa = pd.read_excel('C:/Users/utente/Documents/nsa.xlsx')

ssett = Sample.ix[Sample.index.date > datetime.date(2017,9,4)]
ssett = ssett.ix[ssett.index.date < datetime.date(2017,9,26)]

sam = sample.ix[sample.index.date > datetime.date(2017,9,4)]
sam = sam.ix[sam.index.date < datetime.date(2017,9,26)]

snsa = nsa.ix[nsa.index.date > datetime.date(2017,9,4)]
snsa = snsa.ix[snsa.index.date < datetime.date(2017,9,26)]

errorA = ssett.values.ravel() - sam.values.ravel()
errorB = ssett.values.ravel() - snsa.values.ravel()

plt.figure()
plt.plot(sam.index, errorA, color = 'blue')

plt.figure()
plt.plot(sam.index, errorB, color = 'red')

print np.mean(errorA)
print np.mean(errorB)
print np.std(errorA)
print np.std(errorB)

dec = statsmodels.api.tsa.seasonal_decompose(errorA, freq = 24)
plt.figure();dec.plot()

dec.seasonal[:24]

maeA = errorA/ssett.values.ravel()

print np.mean(maeA)
print np.std(maeA)
print np.max(maeA)
print np.min(maeA)

plt.figure()
plt.hist(maeA, bins = 20, color = 'blue')