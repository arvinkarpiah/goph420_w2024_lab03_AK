
#This script:

# 1. Reads the M_data
# 2. Plots magnitude of earthquake vs time for each day
# 3. Computes number of earthquakes >= given magnitude s for specified intervals
# 4. Performs multi-linear regression for said intervals by calling the multi_regress function
# 5. Plots logN vs M for said intervals and best fit model with parameters.   

import numpy as np
import matplotlib.pyplot as plt
import math
import os

from lab03_python_package.regression import multi_regress


# Set the data_path to save figures
data_path = "/Users/arvinkarpiah/Desktop/GOPH420/Lab_03/Repository/goph420_w2024_lab03_AK/data/M_data.txt"
figpath = "/Users/arvinkarpiah/Desktop/GOPH420/Lab_03/Repository/goph420_w2024_lab03_AK/figures"

# Read the contents of M_data.txt
with open(data_path, 'r') as file:
    # Use np.loadtxt to load the data from the file
    EarthquakeData = np.loadtxt(file)

    Time = EarthquakeData[:,0]
    Magnitude = EarthquakeData[:,1]

# Check to see if data are evenly sampled
# print(Time[2]-Time[1])
# print(Time[145]-Time[144]) 
# print(Time[898]-Time[897])

#Find index for beginning and end of each interval for each day

index_day1_start = 0
index_day1_end = np.abs(Time - 24).argmin()

index_day2_start = np.abs(Time - 24).argmin()
index_day2_end = np.abs(Time - 48).argmin()

index_day3_start = np.abs(Time - 48).argmin()
index_day3_end = np.abs(Time - 72).argmin()

index_day4_start = np.abs(Time - 72).argmin()
index_day4_end = np.abs(Time - 96).argmin()

index_day5_start = np.abs(Time - 96).argmin()
index_day5_end = np.abs(Time - 120).argmin()

#Plot raw earthquake data
plt.figure(figsize=(10, 6))
plt.subplot(3,1,1)
plt.plot(Time, Magnitude, 'b.')
plt.title('Day1-Day5')
plt.ylabel('Event magnitude, M')
plt.legend()

# Plot raw earthquake data day 1
plt.subplot(3,1,2)
plt.plot(Time[index_day1_start : index_day1_end], Magnitude[index_day1_start : index_day1_end], 'b.')
plt.title('Day 1')
plt.ylabel('Event magnitude, M')

# Plot raw earthquake data for day 2
plt.subplot(3,1,3)
plt.plot(Time[index_day2_start : index_day2_end], Magnitude[index_day2_start : index_day2_end], 'b.')
plt.xlabel('Time(hr)')
plt.ylabel('Event magnitude, M')
plt.title('Day 2')
filename = "figure1.png"
filepath = os.path.join(figpath, filename)
plt.savefig(filepath)

# Plot raw earthquake data for day 3
plt.figure(figsize=(10, 6))
plt.subplot(3,1,1)
plt.plot(Time[index_day3_start : index_day3_end], Magnitude[index_day3_start : index_day3_end], 'b.')
plt.xlabel('Time(hr)')
plt.ylabel('Event magnitude, M')
plt.title('Day 3')

# Plot raw earthquake data for day 4
plt.subplot(3,1,2)
plt.plot(Time[index_day4_start : index_day4_end], Magnitude[index_day4_start : index_day4_end], 'b.')
plt.ylabel('Event magnitude, M')
plt.title('Day 4')

# Plot raw earthquake data for day 5
plt.subplot(3,1,3)
plt.plot(Time[index_day5_start : index_day5_end], Magnitude[index_day5_start : index_day5_end], 'b.')
plt.xlabel('Time(hr)')
plt.ylabel('Event magnitude, M')
plt.title('Day 5')
filename = "figure2.png"
filepath = os.path.join(figpath, filename)
plt.savefig(filepath)

#Define indexes where each interval is to start and end for different intervals

index_1_start = 0
index_1_end = np.abs(Time - 24).argmin()

index_2_start = np.abs(Time - 24).argmin()
index_2_end = np.abs(Time - 33).argmin()

index_3_start = np.abs(Time - 33).argmin()
index_3_end = np.abs(Time - 47).argmin()

index_4_start = np.abs(Time - 47).argmin()
index_4_end = np.abs(Time - 70).argmin()

index_5_start = np.abs(Time - 72).argmin()
index_5_end = np.abs(Time - 96).argmin()

index_6_start = np.abs(Time - 96).argmin()
index_6_end = np.abs(Time - 120).argmin()

i1 = [index_1_start,index_1_end]
i2 = [index_2_start,index_2_end]
i3 = [index_3_start,index_3_end]
i4 = [index_4_start,index_4_end]
i5 = [index_5_start,index_5_end]
i6 = [index_6_start,index_6_end]

M1 = Magnitude[i1[0] : i1[1]] 
M2 = Magnitude[i2[0] : i2[1]]  
M3 = Magnitude[i3[0] : i3[1]]  
M4 = Magnitude[i4[0] : i4[1]]
M5 = Magnitude[i5[0] : i5[1]]
M6 = Magnitude[i6[0] : i6[1]]

MUI = [M1,M2,M3,M4,M5,M6]
TUI = [i1,i2,i3,i4,i5,i6]

# Count number of event larger than a certain M
start = -0.5
end = 1
num_points = int((end - start) / 0.1) + 1
x = np.linspace(start, end, num_points)

plt.figure(figsize=(10, 6))

for k in range(len(MUI)):

    Y = np.ones((len(x),1))
    Z = np.ones((len(x),1))
    
    for i in range(len(MUI[k])):
        for j in range(len(x)):
            if MUI[k][i] >= x[j]:
                Y[j] += 1
                Z[j] = x[j]

    # Perform linear regression
    A, e, rsq = multi_regress(np.log10(Y), -Z)
    a = A[0]
    b = A[1]
    N_pred = a - b * Z
    
    # Plot results
    plt.subplot(3,2,k+1)
    plt.plot(Z, np.log10(Y), 'b.', markersize=10)
    plt.plot(Z, N_pred)
    plt.xlabel('M')
    plt.ylabel('log10(N))',fontsize=7)
    plt.xticks(fontsize=7)
    plt.ylim(0,4)
    text = f'{Time[TUI[k][0]]:.0f}hr to {Time[TUI[k][1]-1]:.0f}hr'
    plt.text(0.5, 0.95, text, fontsize=10, transform=plt.gca().transAxes, ha='center', va='top')
    text = f'a={a[0]:.3f}\nb={b[0]:.3f}\nrsq={rsq:.3f}'
    plt.text(0.95, 0.95, text, fontsize=10, transform=plt.gca().transAxes, ha='right', va='top')
    filename = "figure3.png"
    filepath = os.path.join(figpath, filename)
    plt.savefig(filepath)


plt.tight_layout()
plt.show()



