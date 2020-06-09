
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

def calTime(stime):
    etime=time.time()
    diff=etime-stime
    return [diff/(3600*24),diff/3600,diff/60,diff]

sTime=time.time()
# load dataset parking Vancouver (N x M (N Locations & M Instances))
parkingVancDataAdd='processed_Vacouvar.csv'
parkingVancData = pd.read_csv(parkingVancDataAdd).fillna(value=0)
parkingVancDataCol=parkingVancData.columns
weatherParkingVancDataAdd='vacouver_sync_weather_v2.csv'
weatherParkingVancData = pd.read_csv(weatherParkingVancDataAdd)
weatherParkingVancDataCol=weatherParkingVancData.columns

combWeatherParkingVancData=pd.concat([parkingVancData, weatherParkingVancData.iloc[:,1:]],axis=1,ignore_index=True)
combWeatherParkingVancData.columns=list(parkingVancDataCol)+list(weatherParkingVancDataCol[1:])
# load dataset parking Birmingham
#No,year,month,day,hour,pm2.5,DEWP,TEMP,PRES,cbwd,Iws,Is,Ir ()
pollutionDataAdd='bijingAtmosphere\\PRSA_data_2010.csv'
pollutionData = pd.read_csv(pollutionDataAdd, index_col=0)
# load dataset polution Vancouver (N x 4 : SystemCodeNumber,Capacity,Occupancy,LastUpdated)
#SystemCodeNumber,Capacity,Occupancy,LastUpdated

parkingBirDataAdd='birminghamParking\\birmingham_parking_dataset.csv'
parkingBirData = pd.read_csv(parkingBirDataAdd)
weatherParkingBirDataAdd='birminghamParking\\birmingham_sync_weather_v3.csv'
weatherParkingBirData = pd.read_csv(weatherParkingBirDataAdd)
weatherParkingBirDataCol=weatherParkingBirData.columns

parkingBirDataAdd2='finalBirminghamDataArrDF.csv'
parkingBirData2 = pd.read_csv(parkingBirDataAdd2)


parkingBirDataGroup=parkingBirData.groupby('SystemCodeNumber')



parkingBirDataD={}
for i in range(len(parkingBirData)):
    if(parkingBirData.iloc[i,0] in parkingBirDataD):
        row=[parkingBirData.iloc[i,3],(parkingBirData.iloc[i,2]*100/parkingBirData.iloc[i,1])]
        weath=weatherParkingBirData.iloc[i,1:]
        upRow=list(row)+list(weath)
        parkingBirDataD[parkingBirData.iloc[i,0]].append(upRow)
    else:
        parkingBirDataD[parkingBirData.iloc[i,0]]=[]
        weath=weatherParkingBirData.iloc[i,1:]
        row=[parkingBirData.iloc[i,3],(parkingBirData.iloc[i,2]*100/parkingBirData.iloc[i,1])]
        upRow=list(row)+list(weath)
        parkingBirDataD[parkingBirData.iloc[i,0]].append(upRow)

mainGrouprec=1312

refinedListBir=[]
refinedListBirNames=[]
for k in parkingBirDataD:
    if(len(parkingBirDataD[k])>=mainGrouprec):
        refinedListBir.append(parkingBirDataD[k])
        refinedListBirNames.append(k)
        
finalBirminghamData=[]
for i in range(mainGrouprec):
    newLs=[]
    newLs.append(refinedListBir[0][i][0])
    for j in range(len(refinedListBir)):
        newLs.append(refinedListBir[j][i][1])
    up3Row=list(newLs)+list(refinedListBir[0][i][2:])
    finalBirminghamData.append(up3Row)

finalBirminghamDataArr=np.array(finalBirminghamData)
finCols=[]
finCols.append("date")
finCols=finCols+refinedListBirNames+list(weatherParkingBirDataCol[1:])

finalBirminghamDataArrDF=pd.DataFrame(finalBirminghamDataArr)
finalBirminghamDataArrDF.columns=finCols
finalBirminghamDataArrDF.to_csv("finalBirminghamDataArrDF.csv",sep=",",index=False)



# form differ source
#np.corrcoef(list1, list2)[0, 1]
#pollutionData, 
#=======================for vancouver data========
data=combWeatherParkingVancData
corr = data.corr()
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(corr,cmap='coolwarm', vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,len(data.columns),1)
ax.set_xticks(ticks)
plt.xticks(rotation=90)
ax.set_yticks(ticks)
ax.set_xticklabels(data.columns)
ax.set_yticklabels(data.columns)
plt.show()


data=pollutionData.iloc[:,4:]
corr = data.corr()
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(corr,cmap='coolwarm', vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,len(data.columns),1)
ax.set_xticks(ticks)
plt.xticks(rotation=90)
ax.set_yticks(ticks)
ax.set_xticklabels(data.columns)
ax.set_yticklabels(data.columns)
plt.show()

#=============for the Birmingham parking data====

data=parkingBirData2.iloc[:,10:]
corr = data.corr()
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(corr,cmap='coolwarm', vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,len(data.columns),1)
ax.set_xticks(ticks)
plt.xticks(rotation=90)
ax.set_yticks(ticks)
ax.set_xticklabels(data.columns)
ax.set_yticklabels(data.columns)
plt.show()



#=====================
# Angle of collision - variable 1 in correlation example

xData = np.array([24.40,10.25,20.05,22.00,16.90,7.80,15.00,22.80,34.90,13.30])

 

# Energy lost - variable 2 in correlation example

yData = np.array([-4.40,0.25,-0.05,2.00,6.90,-0.80,5.00,2.80,-4.90,3.30])

 

# Draw the scatter plot

lines = plot.xcorr(xData, yData, maxlags=9, usevlines=True)
plt.title('Hypothetical Data: Angle of collision vs Energy lost')
plt.xlabel('Angle of collision')
plt.ylabel('Energy lost')    
plt.grid(True)
plt.axhline(0, color='red', lw=2)
plt.show()