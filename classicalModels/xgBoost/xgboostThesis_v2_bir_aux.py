#--------------complete code
from math import sqrt
from matplotlib import pyplot as plt
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error

import numpy as np
import time
import pandas as pd
import datetime


def calTime(stime):
    etime=time.time()
    diff=etime-stime
    return [diff/(3600*24),diff/3600,diff/60,diff]
# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg
 
# univariate data preparation
from numpy import array

# split a univariate sequence into samples
def split_sequence(sequence, n_steps):
	X, y = list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the sequence
		if end_ix > len(sequence)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)
def minMaxScale1D(data):
    dataMinMax=(data-min(data))/(max(data)-min(data))
    return dataMinMax,min(data),max(data)
def minMaxScale1DInv(dataMinMax,minDat,maxDat):
    data=(dataMinMax*(maxDat-minDat))+minDat
    return data

def conToDatetimeCol(dateTimeCol):
    colsDateTime=[]
    for i in range(len(dateTimeCol)):
        dateStrp=datetime.datetime.strptime(dateTimeCol.iloc[i], '%Y-%m-%d %H:%M:%S')
        colsDateTime.append([dateStrp.year,dateStrp.month,dateStrp.day,dateStrp.hour,dateStrp.minute])
    colsDateTimeDf=pd.DataFrame(colsDateTime)
    colsDateTimeDf.columns=["year","month","day","hour","minute"]
    return colsDateTimeDf

def getLocWithLatLong(locData,locs):
    locDataSub=[]
    locAlreadyAddes=[]
    for i in range(len(locData)):
        if(str(locData.iloc[i,0]) in locs and locData.iloc[i,0] not in locAlreadyAddes):
            locDataSub.append([locData.iloc[i,0],locData.iloc[i,-2],locData.iloc[i,-1]])
            locAlreadyAddes.append(locData.iloc[i,0])
    locDataSub=np.array(locDataSub)
    return locDataSub
def MSE_calc(y_label,y_pred):
    res=y_label-y_pred
    rse_2=res*res
    mse=rse_2.mean()
    return mse

def RMSE_calc(y_label,y_pred):
    res=y_label-y_pred
    rse_2=res*res
    rmse=np.sqrt(rse_2.mean())
    return rmse

def build_model_M(n_timesteps, n_features,out_vars,nNurons,auxNurons,auxInputDim=0,mergemod='concat'):
    input1 = Input(shape=(n_timesteps, n_features))
    lstm1=LSTM(nNurons[0], activation='relu', input_shape=(n_timesteps, n_features))(input1)
    rep1=RepeatVector(out_vars)(lstm1)
    lstm2=LSTM(nNurons[0], activation='relu', return_sequences=True)(rep1)
    dense1=TimeDistributed(Dense(nNurons[1], activation='relu'))(lstm2)

    input2 = Input(shape=(n_timesteps, auxInputDim))
    aux_dense1=TimeDistributed(Dense(auxNurons[0], activation='relu', input_dim=(n_timesteps, auxInputDim)))(input2)

    pred_merged = concatenate([dense1, aux_dense1])
    final_dense1=Dense(out_vars)(pred_merged)
    final_model = Model(inputs=[input1, input2], outputs=[final_dense1])
    final_model.compile(loss='mean_squared_error', optimizer='adam')	
    return final_model

def plotData(idexLs,dataLs,labelLs,xLab,yLab,tittle,legLoc,saveImg=0,resultGraphBase="",lgdLocCost=0):
    for i in range(len(dataLs)):
        plt.plot(idexLs[i],dataLs[i],label=labelLs[i])
    if(lgdLocCost==0):
        plt.legend(loc=legLoc)
    else:
        plt.legend(bbox_to_anchor=lgdLocCost[0], loc=lgdLocCost[1], borderaxespad=lgdLocCost[2])
    plt.xlabel(xLab)
    plt.ylabel(yLab)
    plt.title(tittle)
    plt.grid(True)
    if(saveImg==1):
        plt.savefig(resultGraphBase+tittle+'.png')
#        plt.show()
def to_supervised2(train, n_input, n_out):
	# flatten data
	data = train.reshape((train.shape[0]*train.shape[1], train.shape[2]))
	X, y = list(), list()
	in_start = 0
	# step over the entire history one time step at a time
	for _ in range(len(data)):
		# define the end of the input sequence
		in_end = in_start + n_input
		out_end = in_end + n_out
		# ensure we have enough data for this instance
		if out_end < len(data):
			X.append(data[in_start:in_end, :])
			y.append(data[in_end:out_end, :])
		# move along one time step
		in_start += 1
	return array(X), array(y)
def split_dataset2(data,trainFrom,trainTo,testFrom,testTo,divFactor):
    # split into standard weeks
    train, test = data[trainFrom:trainTo], data[testFrom:testTo]
    # restructure into windows of weekly data
    train = array(np.split(train, divFactor))
    shapeTrain=train.shape
    train=train.reshape(shapeTrain[1],shapeTrain[0],shapeTrain[2])
    test = array(np.split(test, divFactor))
    shapeTest=test.shape
    test=test.reshape(shapeTest[1],shapeTest[0],shapeTest[2])
    return train, test
def conv1To2DPred(d1,size,pad):
    arr1D=np.hstack(((d1,np.zeros(pad))))
    arr2D=arr1D.reshape((size,size))
    return arr2D

import seaborn as sns
import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor

sTime=time.time()
expNumRange=[1,2]
lagLatRange=[1,2,3]
#outRange=[1,2,3]
expIdx=2104191
nNurons=[[200,100]]
auxNurons=[[100]]
verbose= 1
batch_size=72
epochs=100



resultDicmain={}
dataSrc='bir'
sizeMat,pad=4,2
dateColumn="date"
modelName="xgboostAux"
resultDicmainStr=dataSrc+"_"+modelName+"_resultDicmain_4Time3InOut"

#baseAddress="D:\\work_jaguar\\thesisRun\\models\\step3_lstm_multi_stacked\\"
dataFileBir='finalBirminghamDataArrDF.csv'

resultGraphBase="41multiExpResul2BirAux\\"

dataFileBirAux2='bir_date_aux_x2.csv'
# load the new file
dataset = read_csv(dataFileBir, header=0, infer_datetime_format=True, parse_dates=[dateColumn], index_col=0).fillna(0)
values = dataset.iloc[:,:-18].values
aux1Values = dataset.iloc[:,-18:].values
aux2Values = read_csv(dataFileBirAux2, header=0, infer_datetime_format=True).fillna(0)
print("values.shape: ",values.shape)
print("aux1Values.shape: ",aux1Values.shape)
print("aux2Values.shape: ",aux2Values.shape)

aux12Values=np.concatenate((aux1Values,aux2Values),axis=1)

print("aux12Values.shape: ",aux12Values.shape)

scaler = MinMaxScaler(feature_range=(0, 1))
scaledValues = scaler.fit_transform(values)
scaledaux1Values = scaler.fit_transform(aux1Values)
scaledaux2Values = scaler.fit_transform(aux2Values)
scaledaux12Values = scaler.fit_transform(aux12Values)
#print(scaledValues.shape)
# div factor is week
#n_train=52416
n_train=1187+1
#n_train+8064
trainFrom,trainTo,testFrom,testTo,divFactor=0,n_train,n_train,len(values),1
train, test = split_dataset2(values,trainFrom,trainTo,testFrom,testTo,divFactor)
trainAux1, testAux1 = split_dataset2(scaledaux1Values,trainFrom,trainTo,testFrom,testTo,divFactor)
trainAux2, testAux2 = split_dataset2(scaledaux2Values,trainFrom,trainTo,testFrom,testTo,divFactor)
trainAux12, testAux12 = split_dataset2(scaledaux12Values,trainFrom,trainTo,testFrom,testTo,divFactor)
print("train.shape: ",train.shape)
#test=test[:-1]
print("test.shape: ",test.shape)
print("trainAux1.shape: ",trainAux1.shape)
print("testAux1.shape: ",testAux1.shape)
print("trainAux2.shape: ",trainAux2.shape)
print("testAux2.shape: ",testAux2.shape)
print("trainAux12.shape: ",trainAux12.shape)
print("testAux12.shape: ",testAux12.shape)
auxdata=[0,[trainAux1,testAux1],[trainAux2,testAux2],[trainAux12,testAux12]]
auxDataIdx=[1,2,3]
auxDataName=["","Aux1","Aux2","Aux12"]






#def  RunGetResult(dataSrc,expNum,inN,outN,train,test):
               
for auxIdx in auxDataIdx:
    for lag in lagLatRange:
        for Lat in lagLatRange:
#        for expNum in expNumRange:
            expNum=1
            print("Current Lopps param auxIdx:",auxIdx," lag:",lag," Lat:",Lat," expNum:",expNum)
            train_x, train_y = to_supervised2(train, lag,Lat)
            test_x, test_y = to_supervised2(test, lag,Lat)
            trainAux_x, trainAux_y = to_supervised2(auxdata[auxIdx][0], lag,Lat)
            testAux_x, testAux_y = to_supervised2(auxdata[auxIdx][1], lag,Lat)
            
            train_x=np.reshape(train_x,(train_x.shape[0],(train_x.shape[1]*train_x.shape[2])))
            train_y=np.reshape(train_y,(train_y.shape[0],(train_y.shape[1]*train_y.shape[2])))
            test_x=np.reshape(test_x,(test_x.shape[0],(test_x.shape[1]*test_x.shape[2])))
            test_y=np.reshape(test_y,(test_y.shape[0],(test_y.shape[1]*test_y.shape[2])))
            print("train_x.shape: ",train_x.shape)
            print("train_y.shape: ",train_y.shape)
            print("test_x.shape: ",test_x.shape)
            print("test_y.shape: ",test_y.shape)
            
    #        trainAux_x=np.reshape(trainAux_x,(trainAux_x.shape[0],(trainAux_x.shape[1]*trainAux_x.shape[2])))
            trainAux_y=np.reshape(trainAux_y,(trainAux_y.shape[0],(trainAux_y.shape[1]*trainAux_y.shape[2])))
    #        testAux_x=np.reshape(testAux_x,(testAux_x.shape[0],(testAux_x.shape[1]*testAux_x.shape[2])))
            testAux_y=np.reshape(testAux_y,(testAux_y.shape[0],(testAux_y.shape[1]*testAux_y.shape[2])))
    #        print(auxDataName[auxIdx]+"_trainAux_x.shape: ",trainAux_x.shape)
            print(auxDataName[auxIdx]+"_trainAux_y.shape: ",trainAux_y.shape)
    #        print(auxDataName[auxIdx]+"testAux_x.shape: ",testAux_x.shape)
            print(auxDataName[auxIdx]+"testAux_y.shape: ",testAux_y.shape)
            
            train_x_with_aux=np.concatenate((train_x,trainAux_y),axis=1)
            test_x_with_aux=np.concatenate((test_x,testAux_y),axis=1)
            print("train_x_with_aux.shape: ",train_x_with_aux.shape)
            print("test_x_with_aux.shape: ",test_x_with_aux.shape)
            # design network
    #            n_timesteps, n_features= train_x.shape[1], train_x.shape[2]
            model = MultiOutputRegressor(xgb.XGBRegressor(objective='reg:linear',verbosity=verbose)).fit(train_x, train_y)
            
            trainyhat = model.predict(train_x)
            mseTrain= MSE_calc(trainyhat, trainyhat)
            test_yhat = model.predict(test_x)
            mseTest = MSE_calc(test_y, test_yhat)
            test_y_scaled=test_x*100
            test_yhat_scaled=test_yhat*100
            mseTestScaled = MSE_calc(test_y_scaled,test_yhat_scaled)
            
            #train_yhat = model.predict(train_x)
            #mseTrain = MSE_calc(train_y, train_yhat)
            
    #        print(dataSrc+'1_Train MSE: %.3f' % modelTrainHist.history['loss'][-1])
            print(dataSrc+'1_Test MSE: %.3f' % mseTest)
            print(dataSrc+'1_mseTestScaled: %.3f' % mseTestScaled)
            #Key #dataSrc_AuxData_model_expNum_in_out_nNurons_variableName
      
            
            test_res=test_y-test_yhat
            test_res=test_res**2
            test_res=np.reshape(test_res,(test_res.shape[0],Lat,test_res.shape[1]//Lat))
            test_res_sum=np.sum(test_res,axis=0)
            
            key="_".join([dataSrc,auxDataName[auxIdx],modelName,str(expNum),str(lag),str(Lat),"_".join(np.array(nNurons[0]).astype(str)),"testMSELocHeatmap"])
            test_res_sum2D=conv1To2DPred(test_res_sum[0],sizeMat,pad)
            ax = sns.heatmap(test_res_sum2D)
            plt.savefig(resultGraphBase+key+'.png')
            plt.clf()
    #            key="_".join([dataSrc,auxDataName[auxIdx],modelName,str(expNum),str(lagLat),str(lagLat),"_".join(np.array(nNurons[0]).astype(str)),"trainLoss"])
    #            resultDicmain[key]=modelTrainHist.history['loss']
    #            key="_".join([dataSrc,auxDataName[auxIdx],modelName,str(expNum),str(lagLat),str(lagLat),"_".join(np.array(nNurons[0]).astype(str)),"valLoss"])
    #            resultDicmain[key]=modelTrainHist.history['val_loss']
            key="_".join([dataSrc,auxDataName[auxIdx],auxDataName[auxIdx],modelName,str(expNum),str(lag),str(Lat),"_".join(np.array(nNurons[0]).astype(str)),"mseTest"])
            resultDicmain[key]=mseTest
            key="_".join([dataSrc,auxDataName[auxIdx],auxDataName[auxIdx],modelName,str(expNum),str(lag),str(Lat),"_".join(np.array(nNurons[0]).astype(str)),"mseTestScaled"])
            resultDicmain[key]=mseTestScaled

            key="_".join([dataSrc,auxDataName[auxIdx],auxDataName[auxIdx],modelName,str(expNum),str(lag),str(Lat),"_".join(np.array(nNurons[0]).astype(str)),"test_y"])
            resultDicmain[key]=test_y
            key="_".join([dataSrc,auxDataName[auxIdx],auxDataName[auxIdx],modelName,str(expNum),str(lag),str(Lat),"_".join(np.array(nNurons[0]).astype(str)),"test_yhat"])
            resultDicmain[key]=test_yhat         
                

np.save(resultDicmainStr,resultDicmain)

print("*****************************************************")
allTimes=calTime(sTime)
print("Time after whole encode & decode: ",allTimes)
#print("Time in sec per eppoch: ",allTimes[-1]/epochs)
print("*****************************************************")


#van_trainloss_50_150_1_1=np.load('van_trainloss_50_100_1_1_1552316411.npy', mmap_mode='r')