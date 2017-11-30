# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 14:26:30 2017

@author: utente

Sbilanciamento 11 -- user defined choice on what to do
"""

### https://stackoverflow.com/questions/37565793/how-to-let-the-user-select-an-input-from-a-finite-list
### https://stackoverflow.com/questions/4960208/python-2-7-getting-user-input-and-manipulating-as-string-without-quotations

import pandas as pd
from matplotlib import colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import datetime
import Sbilanciamento10 as FF

####################################################################################################
def ToTS(df):
    ts = []
    dmin = df['Giorno'].min() 
    dmax = df['Giorno'].max()
    for i in df.index:
        print i
        y = df[df.columns[-24:]].ix[i].values.ravel().tolist()
        ts.extend(y)
    DF = pd.DataFrame({'X': ts}).set_index(pd.date_range(dmin, dmax + datetime.timedelta(days = 1), freq = 'H')[:len(ts)])                            
    return DF
####################################################################################################
def PODManualRemoduler(Pred, PREDW, pod, sos, rical, pdo, db, zona, d):
    
    insert2 = 99
    sorted_names = mcolors.cnames.keys()
    counter = 0
    candidate = np.repeat(0, 24)
    
    where = False    
    
    if pod in Pred['POD'].values.ravel().tolist():
        candidate = Pred[Pred.columns[2:]].ix[Pred['POD'] == pod].values.ravel()
    elif pod in PREDW['POD'].values.ravel().tolist():
        candidate = PREDW[Pred.columns[2:]].ix[PREDW['POD'] == pod].values.ravel()
        where = True
    else:
        print 'This pod is not in this zone'
        
    plt.figure()
    plt.plot(candidate)
    plt.title('Base prediction for POD {}'.format(pod))
    
    while insert2 > 0:
        print 'what do you want to do?'
        print '0) exit the manual remoduler;'        
        print '1) compare pod to previous days;'
        print '2) increase/decrease prediction'
        insert2 = raw_input("Type your choice here: ")
        
        if int(insert2) == 1:
            print 'How many days do you want to go backwards?'
            num_days = int(raw_input(' '))
            if where:
                db2 = db.ix[db['POD'] == pod]
                db2['Giorno'] = map(lambda date: date.date(), db2['Giorno'].dt.to_pydatetime())
                dbp = db2.ix[db2['Giorno'].values > (d - datetime.timedelta(days = num_days + 1))]
                dbpts = ToTS(dbp)
            else:
                dbpts = pd.DataFrame()
                for i in range(num_days):
                    ts = pd.DataFrame(FF.TakerWithout(rical, pdo, sos, pod, d))
                    dbpts = dbpts.append(ts, ignore_index = True)
                dbpts = ToTS(dbpts)
            dbpts.plot()
        elif int(insert2) == 2:
            if where:
                PREDW = PREDW.drop(PREDW.ix[PREDW['POD'] == pod].index)
            else:
                Pred = Pred.drop(Pred.ix[Pred['POD'] == pod].index)
            YN = True
            while YN:
                for h in range(24):
                    print 'By how much do you want to increase or decrease the forecast in the {}th hour?'.format(h)
                    percentage_h = float(raw_input("Type here the percentage (written as a rational number): "))
                    candidate[h] = (1 + percentage_h) * candidate[h]
                plt.figure()
                plt.plot(candidate, color = sorted_names[counter])
                plt.title('Manual remodulation for pod {}'.format(pod))
                counter += 1
                print 'Are you satisfied?'
                YNl = raw_input("Y/N: ")
                if YNl == 'N':
                    YN = False
                else:
                    YN = True
            pred = [pod, d]
            pred.extend(candidate.tolist())
            pred = pd.DataFrame.from_dict({'0': pred}, orient = 'index')
            
            if where:
                pred.columns = PREDW.columns
                PREDW = PREDW.append(pred, ignore_index = True)
            else:
                pred.columns = Pred.columns
                Pred = Pred.append(pred, ignore_index = True)
                
        elif int(insert2) == 0:
            break
        else:
            print 'invalid choice'