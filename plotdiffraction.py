import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
excel_file = '../data/231106/spectrumHOE_20231106_G.xlsx'
df = pd.read_excel(excel_file)

df1 = df['wavelength']
df2 = df['spectrumGlass']
df3 = df['spectrumHOE']
wlength = df1.values  # wavelengthue
glass = df2.values  # spectrum values through glass
hoe = df3.values  # spectrum values through HOE
bias = 0  # 12.6				#Value to compensate for room's illumination
tolerance = 0.2  # Tolerance to find the wavelength
plotqu = True  # Question to plot the graphs

# Crop data 1080/1980
wlength = wlength[165:1016]
glass = glass[165:1016]
hoe = hoe[165:1016]

# Diffraction efficiency estimation
a = glass-bias
b = hoe-bias
c = np.divide(b, a)
d = 1-c
DE = 100*d  # Diffraction efficiency (%)

# Wavelengths for haze values [nm]
whaze1 = 450
whaze2 = 530
whaze3 = 650

# Compensation estimation. Only use the two intervals [w1,w2] and [w3,w4] [nm]
# Red
""" w1 = 550
w2 = 570
w3 = 580
w4 = 600 """
# Green
""" w1 = 440
w2 = 460
w3 = 600
w4 = 620 """
# Blue
""" w1 = 530
w2 = 550
w3 = 600
w4 = 620 """
# IR
""" w1 = 600
w2 = 620
w3 = 700
w4 = 720 """
# Full color
w1 = 430
w2 = 450
w3 = 570
w4 = 590

# Find the required intervals for the compensation
w1ind = np.where(abs(wlength-w1) < tolerance)
w2ind = np.where(abs(wlength-w2) < tolerance)
w3ind = np.where(abs(wlength-w3) < tolerance)
w4ind = np.where(abs(wlength-w4) < tolerance)
limit1 = w1ind[0][0]
limit2 = w2ind[0][0]
limit3 = w3ind[0][0]
limit4 = w4ind[0][0]
serie1 = d[limit1:limit2]
serie2 = d[limit3:limit4]
seriefull = np.concatenate([serie1, serie2])

# Corresponding wavelength values for the compensation plot
wlengthComp1 = wlength[limit1:limit2]
wlengthComp2 = wlength[limit3:limit4]
wlengthComp = np.concatenate([wlengthComp1, wlengthComp2])

# Line fitting
m, b = np.polyfit(wlengthComp, seriefull, 1)
print('****Compensation line****')
print('Slope: ', m)
print('Intercept: ', b)
print('**************')
# Compensation line
compLine = m*wlength+b

# Haze values
whaze1ind = np.where(abs(wlength-whaze1) < tolerance)
whaze2ind = np.where(abs(wlength-whaze2) < tolerance)
whaze3ind = np.where(abs(wlength-whaze3) < tolerance)
whaze1in = whaze1ind[0][0]
whaze2in = whaze2ind[0][0]
whaze3in = whaze3ind[0][0]
print('Haze for ', whaze1, ' nm: ', np.round(compLine[whaze1in], 3)*100, '%')
print('Haze for ', whaze2, ' nm: ', np.round(compLine[whaze2in], 3)*100, '%')
print('Haze for ', whaze3, ' nm: ', np.round(compLine[whaze3in], 3)*100, '%')


# Compensated diffraction efficiency
DEcomp = DE-100*compLine

# Maximum efficiency
DEcomp2 = DEcomp[limit1:limit4]
diffMaxind = np.argmax(DEcomp2)
maxDE = DEcomp2[diffMaxind]
wlength2 = wlength[limit1:limit4]
maxDEwl = wlength2[diffMaxind]
print('Maximum diffraction efficiency: ', np.round(maxDE, 1), ' %')
print('Wavelength for max DE: ', np.round(maxDEwl, 0), ' nm')

print(np.round(maxDE, 1), ' %')
print(np.round(maxDEwl, 0), ' nm')

print(np.round(compLine[whaze1in], 3)*100, '%')
print(np.round(compLine[whaze2in], 3)*100, '%')
print(np.round(compLine[whaze3in], 3)*100, '%')

# Plot results
if (plotqu == True):
    fig = plt.figure()
    plt.plot(wlength, glass, label='spectrum through glass')
    plt.grid()
    plt.title('Light source spectrum')
    plt.xlabel('wavelength [nm]')
    plt.ylabel('intensity [A.U.]')
    plt.legend(loc='upper left')

    fig2 = plt.figure()
    plt.plot(wlength, hoe, label='spectrum through hoe')
    plt.grid()
    plt.title('HOE transmitted spectrum')
    plt.xlabel('wavelength [nm]')
    plt.ylabel('intensity [A.U.]')
    plt.legend(loc='upper left')

    fig = plt.figure()
    plt.plot(wlengthComp, seriefull, label='compensation')
    plt.grid()
    plt.title('Data plot for compensation (haze)')
    plt.xlabel('wavelength [nm]')
    plt.ylabel('intensity [A.U.]')
    plt.legend(loc='upper left')

    fig = plt.figure()
    plt.plot(wlength, compLine, label='compensationFitting')
    plt.grid()
    plt.title('Data plot for compensation (haze)-Fitted line')
    plt.xlabel('wavelength [nm]')
    plt.ylabel('intensity [A.U.]')
    plt.legend(loc='upper left')

    fig3 = plt.figure()
    plt.plot(wlength, DE, label='diffraction efficiency without compensation')
    plt.plot(wlength, 100*compLine, label='compensationFitting')
    plt.xlim(400, 700)
    plt.ylim(-20, 100)
    plt.grid()
    plt.title('HOE diffracted spectrum')
    plt.xlabel('wavelength [nm]')
    plt.ylabel('Diffraction efficiency [%]')
    plt.legend(loc='upper left')

    fig = plt.figure()
    plt.plot(wlength, DEcomp, label='diffraction efficiency WITH compensation')
    plt.xlim(400, 700)
    plt.ylim(-20, 100)
    plt.grid()
    plt.title('HOE diffracted spectrum')
    plt.xlabel('wavelength [nm]')
    plt.ylabel('Diffraction efficiency [%]')
    plt.legend(loc='upper left')

    plt.show()
