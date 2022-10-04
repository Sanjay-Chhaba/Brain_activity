import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import random
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score as auc
from sklearn.metrics import multilabel_confusion_matrix as cm
from scipy.interpolate import make_interp_spline, BSpline
from scipy.signal import butter, lfilter
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
from tqdm import tqdm
 



def getSubjectsData(testSeries=3,test=False):
  ''' Function to get Individual Subjects's Data
  Arguments: test_series (int), Test(boolean)
  Output: Subject's EEG Data (df), Subjects Event Data (df)
  '''
  # define which series will be used as the test series
  test = testSeries
  train = list(np.arange(1,9))
  train.remove(test)

  
  path = '/content/drive/My Drive/Colab Notebooks/Bionic AI/Kaggle EEG Data/train/'

  subjectsData = []
  subjectsEvents = []


  if test == False:
    for i in range(1,13):
  # initiate the dartaframes by passing them the first series of subject i
      stackData = pd.read_csv(path + f'subj{i}_series1_data.csv').iloc[:,1:]
      stackEvents = pd.read_csv(path + f'subj{i}_series1_events.csv').iloc[:,1:]
  # for subject i, import data from all other training series and stack them
      for j in train[1:]: 
          data = pd.read_csv(path + f'subj{i}_series{j}_data.csv').iloc[:,1:]
          stackData = pd.concat([stackData,data])  
      
          events = pd.read_csv(path + f'subj{i}_series{j}_events.csv').iloc[:,1:]
          stackEvents = pd.concat([stackEvents,events])
  # normalize the stacked series data for subject i
      stackDataNorm = ((stackData - stackData.mean(axis=0))/stackData.std(axis=0))
  # Rest the index for both 
      stackDataNorm = stackDataNorm.reset_index(drop=True)
      stackEvents = stackEvents.reset_index(drop=True)

      subjectsData.append(stackDataNorm)
      subjectsEvents.append(stackEvents)

    return subjectsData, subjectsEvents

# if test= True, only return the test series for a subject
  else:
    for i in range(1,13):
   # get the testing data for all subjects. AKA, the series that will be used for testing for each subject i 
      data = pd.read_csv(path + f'subj{i}_series{testSeries}_data.csv').iloc[:,1:]
      events = pd.read_csv(path + f'subj{i}_series{testSeries}_events.csv').iloc[:,1:]

      dataNorm = ((data - data.mean(axis=0))/data.std(axis=0))

      subjectsData.append(dataNorm)
      subjectsEvents.append(events)

    return subjectsData, subjectsEvents




def getSeriesEvents(subjectNum,seriesNum):
  ''' Function to extract a specific series' events for a given subject
  Arguments: subject_num (int), series_num (int)
  Output: Subjects EEG Data (df), Subject's Event Data (df)
  '''
  path = '/content/drive/My Drive/Colab Notebooks/Bionic AI/Kaggle EEG Data/train/'
  subject = pd.read_csv(path + f'subj{subjectNum}_series{seriesNum}_data.csv').iloc[:,1:]
  events = pd.read_csv(path + f'subj{subjectNum}_series{seriesNum}_events.csv').iloc[:,1:]
  subject = (subject - subject.mean(axis=0)) / subject.std(axis=0)
  return subject, events




def graphTrials(subjectNum,seriesNum, oneTrial=False):
  ''' Function to graph all events within all trials (or just one if oneTrial=True) for one series for a sbject
  Arguments: subjectNum (int), seriesNum (int), oneTrial (boolean)
  Output: graph
  ''' 
  plt.style.use('dark_background')
  fig = plt.figure(figsize=[30,2])
  plt.xlabel('Time(ms)', fontsize=15)
  
  _ , events = getSeriesEvents(subjectNum,seriesNum)
  Motions = events.columns.tolist()
  
  
  for M in Motions:
    if oneTrial==False:
      tf = events[events[M] == 1].index.tolist()
    else:
      tf = events[events[M] == 1].index.tolist()[:150]
    event = [M] * len(tf) 
    plt.scatter(tf, event)




def getAllData(testSeries=3):
  ''' Function to stack data from all subjects
  Arguments: testSeries (int)
  Output: All EEG data (df), all events (df)
  '''
  # define which series will be used as the test series
  train = list(np.arange(1,9))
  train.remove(testSeries)

  path = '/content/drive/My Drive/Colab Notebooks/Bionic AI/Kaggle EEG Data/train/'

  subjectsData = []
  subjectsEvents = []

  for i in range(1,13):
    print(f'reading subject {i} out of 12')
    #initiate the dataframe to hold data for each subject by passing the first series of data to it
    stackData = pd.read_csv(path + f'subj{i}_series1_data.csv').iloc[:,1:]
    stackEvents = pd.read_csv(path + f'subj{i}_series1_events.csv').iloc[:,1:]
    # stack the rest of the series below the first for each subject i
    for j in train:
        data = pd.read_csv(path + f'subj{i}_series{j}_data.csv').iloc[:,1:]
        stackData = pd.concat([stackData,data])  
      
        events = pd.read_csv(path + f'subj{i}_series{j}_events.csv').iloc[:,1:]
        stackEvents = pd.concat([stackEvents,events])

    subjectsData.append(stackData)
    subjectsEvents.append(stackEvents)
    # concatenate all subjects data
  print(f'concatenating all data')
  allData = pd.concat(subjectsData)
  allEvents = pd.concat(subjectsEvents)
    # normalize the final dataframe and reset indices     
  dataNorm = ((allData - allData.mean(axis=0))/allData.std(axis=0))
    
  dataNorm = dataNorm.reset_index(drop=True)
  allEvents = allEvents.reset_index(drop=True)

  return dataNorm, allEvents 




def truncFrame(data,allEvents,ev,step, window):
  ''' Function to get truncated data for a single event 
  for a timeframe of 150 ms before and after the event onset
  Arguments: subject's EEG data(df), subjects EEG events(df), event of choice('string'), step size (int), rolling average window size (int)
  Output: truncated EEG data for the event(df)
  '''
  motionIndex = allEvents[allEvents[ev] == 1].index.to_numpy()
  numRows = len(data.iloc[motionIndex[0]-100:motionIndex[0+149]+50,:][::step])
  allFrames = np.zeros((numRows,32))
  trials = 0
  i = 0
  while i < len(motionIndex/150):
    truncFrame = data.iloc[motionIndex[i]-100:motionIndex[i+149]+50,:][::step].to_numpy()
    allFrames += truncFrame
    trials += 1
    i += 150
  allFrames /= trials
  allFramesAve = pd.DataFrame(allFrames).rolling(window).mean().iloc[window-1:,:]
  
  return allFramesAve.to_numpy()




def getGraph(data,allEvents,ev,window=5,step=10):
  ''' Function to graph a single event to visualize how the brain activity changes during an event 
  Arguments: subject's EEG data(df), subjects EEG events(df), event of choice('string'), step size (int), rolling average window size (int)
  Output: Graph of a single event for a timeframe of 75 ms before and after the event onset (sampling rate = 500 Hz)
  '''
# plot figures for how readings change across all 12 subjects for 'HandStart' action
  plt.style.use('default')
  fig = plt.figure(12, figsize=[50,10])
  plt.title(ev,fontsize=40)
  plt.xlabel('time',size=20)
  plt.ylabel('electrodes',size=20)

  frame = truncFrame(data,allEvents,ev,window,step)
    
  xnew = np.linspace(0, len(frame), 300) # 300 represents number of points to make between T.min and T.max
  spl = make_interp_spline(range(len(frame)), frame)  # type: BSpline
  power_smooth = spl(xnew)

  plt.plot(xnew, power_smooth)
  # plot event onset, taking into account the dropped rows
  plt.axvline(xnew[100+window])
  axes = plt.gca()
  plt.text(xnew[100+window]+0.1,axes.get_ylim()[1]-0.1,'Event Onset', size=20)
  plt.show()




def getBatchTest(data, events, num_samples, test=False, test1Trial=False):
  ''' Function to create batches from data for training/testing
  Arguments: subject's EEG data(df), subjects EEG events(df), number of samples(int), test (boolean), test1Trial (boolean)
  Output: Subjects EEG data (batches of size 2000x256x32) (DoubleTensor), The corresponding events (DoubleTensor)
  '''
  num_features = 32 # number of electrodes
  window_size = 512
  # for demo animation
  if test1Trial==True:
    num_samples = (3400 - 1000)
    indexes = np.arange(1000, 3400)
  # for training and testing
  else:
    index = random.randint(window_size, len(data) - 16 * num_samples) # choose  a starting index: a number bigger than 1024 (window size) and less than the number of indexes that will be used by the batch
    
    if Test == False:
        indexes = np.arange(index, index + 16*num_samples, 16)

    else:
        index = random.randint(window_size, len(data) - num_samples) # much smaller dataset so dont have to make the 16 step jump between single batches
        indexes = np.arange(index, index + num_samples)

  X = np.zeros((num_samples, num_features, window_size//2))
  b = 0

  for i in indexes:
        
      start = i - window_size if i - window_size > 0 else 0
        
      tmp = data.iloc[start:i,:]
      X[b,:,:] = tmp[::2].transpose()
      b += 1
  y = events[events.index.isin(indexes)]
  y = y.to_numpy()

  return torch.DoubleTensor(X), torch.DoubleTensor(y)  




def train(model, Xtrain, ytrain, epochs, batch_size,verbos=1):
  ''' Function to train the model using batches of data
  Arguments: model, Xtrain (double tensor), ytrain (double tensor), epochs (int), batch_size (int),verbos (int)
  Output: trained model
  '''
  optimizer = torch.optim.Adadelta(model.parameters(), lr=1, eps=1e-10)
  model.train()
  for epoch in range(epochs):
    total_loss = 0 # set loss = 0
    
    for i in tqdm(range(len(Xtrain)//batch_size)):

      optimizer.zero_grad()
      x, y = getBatch(Xtrain, ytrain, batch_size, Test=False)
      while y.shape[0] != batch_size:
        x, y = getBatch(Xtrain, ytrain, batch_size, Test=False)
      outputs = model(x)
      loss = F.binary_cross_entropy(outputs.reshape(-1),y.reshape(-1)) # flattens both
      loss.backward() # backward propagation
      total_loss += loss.item()
      optimizer.step() # update the weights using the selected optimizer function 
      
      print(f'\t epoch: {epoch}, iteration: {i}/{len(Xtrain)//batch_size}, loss: {total_loss}')
      total_loss = 0


def getPredictions(model,Xtest,ytest,window_size,batch_size):
  ''' Function to predict using the trained model and batches of data
  Arguments: model, Xtest(double tensor) , ytest(double tensor) , window_size(int) , batch_size(int)
  Output: y pred (array), y test (array)
  '''
  model.eval()

  p = []
  tru = []

  while window_size < len(Xtest):
    if window_size + batch_size > len(Xtest):
      batch_size = len(Xtest) - window_size
    x_test, y_test = getBatch(Xtest, ytest, 2000, Test=True)
    x_test = (x_test)

    preds = model(x_test)

    p.append(np.array(preds.data))
    tru.append(np.array(y_test.data))

    window_size += batch_size
  preds = p[0]
  for i in p[1:]:
    preds = np.vstack((preds,i))
  
  test = tru[0]
  for i in tru[1:]:
    test = np.vstack((test,i))
  return preds, test




class CNN(nn.Module):
  ''' PyTorch CNN Model'''
  def __init__(self):
    super().__init__()

    self.conv1 = nn.Conv1d(32, 64, kernel_size=3,padding=0,stride=1) # why 32 and 64 ?
    self.bn = nn.BatchNorm1d(64)
    self.pool = nn.MaxPool1d(2,stride=2) # max pooling over a 2x2 window
    self.dropout1 = nn.Dropout(0.5)
    self.linear1 = nn.Linear(8128, 2048) # ????? isnt it 256x32 = 8192??
    self.dropout2 = nn.Dropout(0.5)
    self.dropout3 = nn.Dropout(0.5)
    self.linear2 = nn.Linear(2048, 124)
    self.dropout4 = nn.Dropout(0.5)
    self.dropout5 = nn.Dropout(0.5)
    self.linear3 = nn.Linear(124,6) ## check the output

    self.conv = nn.Sequential(self.conv1, nn.ReLU(inplace = True), self.bn, self.pool, 
                             self.dropout1)
    
    self.net = nn.Sequential(self.linear1, nn.ReLU(inplace=True), 
                             self.dropout2, self.dropout3, self.linear2, nn.ReLU(inplace=True),self.dropout4, self.dropout5, self.linear3   )

  def forward(self, x):
    batch_size = x.size(0)
    x = self.conv(x)
    x = x.reshape(batch_size, -1) # If there is any situation that you don't know how many columns you want but are sure of the number of rows, then you can specify this with a -1.
    out = self.net(x) # try the way pytorch does it
        
    return torch.sigmoid(out)




class LR(nn.Module):
  ''' PyTorch Log Reg Model'''
  def __init__(self):
    super().__init__()

    self.linear1 = nn.Linear(8192, 124) # ????? isnt it 256x32 = 8192??
    self.dropout1 = nn.Dropout(0.5)
    self.dropout2 = nn.Dropout(0.5)
    self.linear2 = nn.Linear(124, 6)

    self.reg = nn.Sequential(self.linear1, nn.ReLU(inplace=True), 
                             self.dropout1, self.dropout2, self.linear2)

  def forward(self, x):
    batch_size = x.size(0)
    x = x.reshape(batch_size, -1) # If there is any situation that you don't know how many columns you want but are sure of the number of rows, then you can specify this with a -1.
    out = self.reg(x) # try the way pytorch does it

    return torch.sigmoid(out)




def getLoss(x1true,x2true,x1pred,x2pred):
  ''' Function to get a loss function for defining a cut off threshold for each event prediction
  Arguments: first element of y test array (int), last element of y test array (int), first element of y pred array (int), last element of y pred array (int)
  Output: loss
  '''
  return sum([abs(x1true-x1pred),abs(x2true-x2pred)])




def getBestThresh(preds,test):
  ''' Function to get the cut off threshold for each event prediction
  Arguments: y pred (array), y test(array)
  Output: best thresholds (dict)
  '''
  # threshold range
  threshold = list(np.arange(0,1,0.001))
  bestTh = {}
  events = [0,1,2,3,4,5]
  eventNums = {0: 'HandStart', 1:'FirstDigitTouch', 2:'BothStartLoadPhase', 3:'LiftOff', 4:'Replace', 5:'BothReleased'}
  plt.style.use('dark_background')

  for event in events:  
    truMotion = np.where(test[:,event] == 1)[0]
    losses = {}
    for th in threshold:
      predMotion = np.where(preds[:,event] > th)[0]
      try:
        loss = getLoss(truMotion[0],truMotion[-1],predMotion[0],predMotion[-1])
        #overlap = len(set(truMotion) & set(predMotion))
        lenDiff = abs(len(truMotion) - len(predMotion))
        losses[th] = loss + lenDiff

      except IndexError:
        pass
    bestTh[event] = min(losses, key=losses.get)
  return bestTh