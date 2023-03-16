import pandas as pd
import numpy as np

import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from matplotlib import pyplot
import torch
import torch.nn.functional as F
import gzip
import os

torch.manual_seed(0)
np.random.seed(0)
#torch.backends.cudnn.benchmark = False
#torch.backends.cudnn.deterministic = True

#reading data from excel sheet 
df = pd.read_excel("ralstonia_high.xlsx")

inputFeatures = [0.0,          1.0,          2.0,          3.0,
                4.0,          5.0,          6.0,          7.0,          8.0,
                9.0,         10.0,         11.0,         12.0,         13.0,
               14.0,         15.0,         16.0,         17.0,         18.0,
               19.0,         20.0,         21.0,         22.0,         23.0,
               24.0,         25.0,         26.0,         27.0,         28.0,
               29.0,         30.0,         31.0,         32.0,         33.0,
               34.0,         35.0,         36.0,         37.0,         38.0,
               39.0,         40.0,         41.0]
targetVariable  = ['ralstonia']

df_train=df.sample(frac=0.8,random_state=200)
df_test=df.drop(df_train.index)

#operation to repeat the 851 values, 851 times
df_Repeat = df_train.iloc[np.repeat(np.arange(len(df_train)), len(df_train))] # this would repeat sample1 851 times, then sample2 851 times... 
df_tile = df_train.iloc[np.tile(np.arange(len(df_train)), len(df_train))] 
#print(df_Repeat, df_tile)

#taking X and Y values
x_Repeat = df_Repeat[inputFeatures].to_numpy()
y_Repeat = df_Repeat[targetVariable].to_numpy()
x_tile = df_tile[inputFeatures].to_numpy()
y_tile = df_tile[targetVariable].to_numpy()

# the difference of samples in the left and samples in the right (of twin regression network)
y_diff = y_Repeat - y_tile
print(x_Repeat.shape,y_Repeat.shape, y_tile.shape, y_diff.shape)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Twin(torch.nn.Module):
     def __init__(self):
         super(Twin, self).__init__()
         self.nn1 = torch.nn.Linear(84,60)
         self.nn2 = torch.nn.Linear(60,30)
         self.nn3 = torch.nn.Linear(30,1) 

            
     def forward(self, x1):
        
         o1 = self.nn1(x1)
         o1 = F.relu(o1)
         o1 = self.nn2(o1)
         o1 = F.relu(o1)
         o1 = self.nn3(o1)     
         return o1
        
model = Twin().to(device)
x_TrainingSamples = np.concatenate((x_Repeat, x_tile), axis=1)
d1 = torch.from_numpy(x_TrainingSamples).to(torch.float32).to(device)
ydiff = torch.from_numpy(y_diff).to(torch.float32).to(device)


optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
model.train()

epochs = 1000

losses = []

for epoch in range(epochs):
     optimizer.zero_grad()
     out = model(d1)
     MSE = torch.nn.MSELoss()
     loss = MSE(out, ydiff)
     print('Epoch ' + str(epoch) + ' :Training loss: {}'.format(loss.item()))
     loss.backward()
     losses.append(loss.item())
     optimizer.step()
torch.save(model, 'saved_models/high_fork4.pt')

model.eval()


#from matplotlib import pyplot as plt
#plt.plot(np.linspace(1, epochs, epochs).astype(int), losses)

df = {}
#df['Ralstonia Value (Mean)'] = []
#df['Variance'] = ''
#df['ErrorMeanSquare'] = ''
#df['#of 1s'] = ''
#df['Sequence'] = []

string = ",".join([str(ijk) for ijk in list(df.keys())])


#os.system(string + ">" + dpFName)
rastValues = []
seqs = []
actuals = []
errorMs = []

X = df_test[inputFeatures]

Y = df_test[targetVariable].to_numpy()

X_np = X.to_numpy()

x1 = torch.from_numpy(X_np).to(torch.float32).to(device)
ydiff = torch.from_numpy(Y).to(torch.float32).to(device)

        
def getSeqRalstonianValue(newSeqDigits, y1):

        synDf = pd.DataFrame(newSeqDigits).T
        synthetic_Repeat = synDf.iloc[np.repeat(np.arange(len(synDf)), X.shape[0])].to_numpy()
        synthetic_np = np.concatenate((synthetic_Repeat, X_np), axis=1)

        synthetic_np = torch.from_numpy(synthetic_np).to(torch.float32).to(device)
   
        pred = model(synthetic_np)
        ydiff_predict = pred.detach().cpu().numpy() + ydiff.detach().cpu().numpy()
        mn = np.mean(ydiff_predict)
        actual_ralstonia = y1
#        vrs = np.var(ydiff_predict, dtype=np.float64)
        mses = np.mean(ydiff_predict - actual_ralstonia)**2
        combs = ",".join([str(ijk) for ijk in synDf.to_numpy()[0]])
        rastValues.append(mn)
        actuals.append(str(actual_ralstonia[0]))
        seqs.append(combs)
        errorMs.append(mses)
#        df = {}
#        df['Ralstonia Value (Mean)'] = [mn]
#        df['Variance'] = [vrs]
#        df['ErrorMeanSquare'] = [mses]
#        df['#of 1s'] = [np.sum(newSeqDigits)]
#        df['Sequence'] = [combs]
        
#        dff = pd.DataFrame.from_dict(df)
#        print(dff)
#        with gzip.open(dpFName, 'a') as compressed_file:
#           dff.to_csv(compressed_file, index=False, header=False)




for ind, newSeqDigits in enumerate(X_np):
    getSeqRalstonianValue(newSeqDigits, Y[ind])

#print(rastValues, seqs, actuals, errorMs)
df['Ralstonia Value (Mean)'] = rastValues
df['Actual Ralstonia Value'] = actuals
#df['Variance'] = ''
df['ErrorMeanSquare'] = errorMs
df['Sequence'] = seqs

dff = pd.DataFrame.from_dict(df)
print(dff)
fileName = "saved_files/high_fork4.csv"
dff.to_csv(fileName, index=False)

