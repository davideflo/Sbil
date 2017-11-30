# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 10:28:19 2017

@author: utente

Updating of meteo's trend

"""

import pandas as pd
import matplotlib.pyplot as plt



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

plt.figure()
mi2017['Tmax'].plot()
plt.axvline(x = datetime.date(2017,9,23), color = 'blue')
plt.axvline(x = datetime.date(2017,9,29), color = 'red')
plt.axvline(x = datetime.date(2017,9,30), color = 'black')


fi2017['Tmedia'].plot()
ro2017['Tmedia'].plot()
ba2017['Tmedia'].plot()
pa2017['Tmedia'].plot()
ca2017['Tmedia'].plot()

TM = pd.DataFrame.from_dict({'NORD': mi2017['Tmedia'].values.ravel().tolist(),
                             'CNOR': fi2017['Tmedia'].values.ravel().tolist(),
                             'CSUD': ro2017['Tmedia'].values.ravel().tolist(),
                             'SUD': ba2017['Tmedia'].values.ravel().tolist(),
                             'SICI': pa2017['Tmedia'].values.ravel().tolist(),
                             'SARD': ca2017['Tmedia'].values.ravel().tolist()}, orient = 'columns')
                             
TM = TM.set_index(mi2017.index)

TM['NORD'].plot()

TM.to_excel('Tmedie_per_zona.xlsx')

plt.figure();plt.scatter(mi5['Tmax'].values.ravel(),nord['MO [MWh]'].ix[nord.index.year == 2017].resample('D').mean().values.ravel())


ex = nord['MO [MWh]'].ix[nord.index.year == 2017].resample('D').mean().values.ravel()

exok = ex[ex > 62]

exmi = mi5['Tmax'].values.ravel()

exmiok = exmi[ex > 62]

plt.figure()
plt.scatter(exmiok, exok)


dat = pd.DataFrame({'y': exok, 'T': exmiok, 'T2': exmiok**2})

dat = dat.sort_values(by = 'y')

lm = LinearRegression(fit_intercept = True)

lm.fit(dat[dat.columns[1:2]], dat['y'])

lm.coef_
model = make_pipeline(PolynomialFeatures(3), LinearRegression(fit_intercept = True))
model.fit(dat[dat.columns[1:2]], dat['y'])
y_plot = model.predict(dats[dats.columns[1:2]])
y_plot2 = model.predict(dat[dat.columns[1:2]])
print r2_score(dat['y'], y_plot2)


dats = pd.DataFrame({'T': np.sort(exmiok), 'T2': np.sort(exmiok)**2})

plt.figure()
plt.scatter(exmiok, exok)
plt.plot(dats['T'], y_plot)


residuals = dat['y'] - y_plot2

plt.figure()
plt.hist(residuals, bins = 20)

plt.figure()
plt.scatter(residuals, y_plot)


np.mean(residuals)
np.std(residuals)

perdite = pd.read_excel('C:/Users/utente/Documents/Sbilanciamento/perdite.xlsx')       

podlist = list(set(db['POD'].ix[db['Giorno'] == max(db['Giorno'])]).intersection(podcrpp))
ps = []
for p in podlist:
    per = perdite['PERDITE'].ix[perdite['POD'] == p].values.ravel()    
    if per.size > 1:
        per = per[np.where(per == max(per))[0]]
    ps.append(per[0])



