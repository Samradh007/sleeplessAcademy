import math
import numpy as np
import os 
import pandas as pd
from sklearn import metrics

import torch
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim 
from torch.utils.data import Dataset, DataLoader 
from torchsummary import summary

data_csv = './data_new.csv'
df = pd.read_csv(data_csv)
df = df.drop(['Frame_Num'], axis=1)
df = df.drop(df[df['Label'] == 'distracted'].index)
df['Label'] = df['Label'].astype('category').cat.codes

pnum = max(df['P_ID']) + 1
test_df, train_df = [row for _, row in df.groupby(df['P_ID'] < 7)]
test_df, val_df = [row for _, row in test_df.groupby(test_df['P_ID'] < 8)]

train_pnums = train_df['P_ID']
train_label = train_df['Label']
train_df = train_df.drop(['P_ID', 'Label'], axis=1)

test_pnums = test_df['P_ID']
test_label = test_df['Label']
test_df = test_df.drop(['P_ID', 'Label'], axis=1)

val_pnums = val_df['P_ID']
val_label = val_df['Label']
val_df = val_df.drop(['P_ID', 'Label'], axis=1)

print (train_df.shape, val_df.shape, test_df.shape)
print (list(train_df))

class FFEDataset(Dataset):
  def __init__ (self, data, labels, time_steps=5, n_features=4):
    super(FFEDataset, self).__init__()
    self.time_steps = time_steps
    self.n_features = n_features
    self.data = np.array(data).reshape(-1, self.time_steps, self.n_features)
    self.labels = np.array(labels).reshape(-1, self.time_steps, 1).mean(axis=1)
  
  def __len__(self):
    return len(self.labels)

  def __getitem__(self, index):
    return torch.FloatTensor(self.data[index]), torch.FloatTensor(self.labels[index])

class CLF_LSTM(nn.Module):
  def __init__(self, batch_size):
    super(CLF_LSTM, self).__init__()
    self.x_dim = 4
    self.time_steps = 5
    self.input_dim = 64
    self.hidden_dim = 64
    self.num_layers = 2
    self.batch_size = batch_size
    self.pre_fc = nn.Linear(self.time_steps * self.x_dim, self.time_steps * self.input_dim)
    self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers, batch_first=True)
    self.d1 = nn.Dropout(p=0.5)
    self.fc1 = nn.Linear(self.hidden_dim, 256)
    self.d2 = nn.Dropout(p=0.5)
    self.fc2 = nn.Linear(256, 64)
    self.fc3 = nn.Linear(64, 8)
    self.fc4 = nn.Linear(8, 1)
    
  def init_hidden(self):
    self.hidden = (torch.zeros((self.num_layers, self.batch_size, self.hidden_dim)), \
    torch.zeros((self.num_layers, self.batch_size, self.hidden_dim)))
  
  def forward(self, x):
    self.init_hidden()
    out = x.reshape(-1, self.time_steps * self.x_dim)
    out = F.relu(self.pre_fc(out))
    out = out.reshape(-1, self.time_steps, self.input_dim)
    out, self.hidden = self.lstm(out, self.hidden)
    out = out[:, -1, :]
    out = self.d1(out)
    out = F.relu(self.fc1(out))
    out = self.d2(out)
    out = F.relu(self.fc2(out))
    out = F.relu(self.fc3(out))
    out = self.fc4(out)
    return out

train_dataset = FFEDataset(train_df, train_label, time_steps=5)
val_dataset = FFEDataset(val_df, val_label, time_steps=5)
test_dataset = FFEDataset(test_df, test_label, time_steps=5)

batch_size = 4

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

print (len(train_dataset), len(val_dataset), len(test_dataset))
print (len(train_loader), len(val_loader), len(test_loader))

model = CLF_LSTM(batch_size)

for param in model.parameters():
  param.requires_grad = True

lr = 0.001
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
criterion = nn.BCEWithLogitsLoss(pos_weight = torch.tensor([2]))


def train_model(epochs, opt, model, model_path, scheduler=None):
  train_loss = []
  val_loss = []
  min_loss = np.Inf
  max_acc = 0

  for e in range(epochs):
    running_loss = 0.0
    model.train()
    model.to(device)
    for batch_data, batch_label in iter(train_loader):
      batch_data = batch_data.to(device)
      batch_label = batch_label.to(device)
      
      opt.zero_grad()

      batch_pred = model(batch_data)
      loss = criterion(batch_pred, batch_label)
      loss.backward()
      opt.step()
      running_loss += loss.item()
    train_loss.append(running_loss)

    model.eval()

    val_labels = np.zeros((len(val_dataset), 1))
    val_preds = np.zeros((len(val_dataset), 1))
    indx = 0
    running_loss = 0.0
    for batch_data, batch_label in iter(val_loader):
      batch_data = batch_data.to(device)
      batch_label = batch_label.to(device)
      with torch.no_grad():
        batch_pred = model(batch_data)
        loss = criterion(batch_pred, batch_label)
        running_loss += loss.item()
      batch_pred = torch.sigmoid(batch_pred).gt(0.5).int()
      val_preds[ indx : (indx + batch_pred.shape[0]), :] = batch_pred.to("cpu").data.numpy()
      val_labels[ indx : (indx + batch_label.shape[0]), :] = batch_label.to("cpu").data.numpy()
      indx = indx + batch_pred.shape[0]
    val_loss.append(running_loss)
    val_acc = metrics.accuracy_score(val_labels, val_preds)

    if val_loss[-1] < min_loss:
      min_loss = val_loss[-1]
      model.to("cpu")
      torch.save(model.state_dict(), model_path)
      print ('saved model at epoch = %d for Min val loss' %(e))
    

    if scheduler:
      scheduler.step()

    if e % 2 == 0:
      print ('epoch=%d training_loss=%.4f val_loss=%.4f  val_acc = %.4f' %(e, train_loss[-1], val_loss[-1], val_acc))
    
  plt.plot(range(len(train_loss)), train_loss)
  plt.plot(range(len(val_loss)), val_loss)
  plt.legend(["Train Loss", "Val Loss"], loc="upper right")


train_model(52, opt, model, './clf_lstm.pth', scheduler)

def val_model(model):
  test_preds = np.zeros((len(val_dataset), 1))
  test_labels = np.zeros((len(val_dataset), 1))
  model.to(device)
  model.eval()
  indx = 0
  with torch.no_grad():
    for batch_data, batch_labels in iter(val_loader):
      batch_data = batch_data.to(device)
      batch_labels = batch_labels.to(device)
      batch_pred = torch.sigmoid(model(batch_data)).gt(0.5).int()
      test_preds[ indx : (indx + batch_pred.shape[0]), :] = batch_pred.to("cpu").data.numpy()
      test_labels[ indx : (indx + batch_labels.shape[0]), :] = batch_labels.to("cpu").data.numpy()
      indx = indx + batch_pred.shape[0]
  
  accuracy = metrics.accuracy_score(test_labels, test_preds)
  print ('accuracy', accuracy)
  return accuracy, test_preds, test_labels


model = CLF_LSTM(batch_size=batch_size)
model.load_state_dict(torch.load('./clf_lstm.pth'))
accuracy, val_pred, val_lab = val_model(model)
print (metrics.confusion_matrix(val_lab, val_pred))

def test_model(model):
  test_preds = np.zeros((len(test_dataset), 1))
  test_labels = np.zeros((len(test_dataset), 1))
  model.to(device)
  model.eval()
  indx = 0
  with torch.no_grad():
    for batch_data, batch_labels in iter(test_loader):
      batch_data = batch_data.to(device)
      batch_labels = batch_labels.to(device)
      batch_pred = torch.sigmoid(model(batch_data)).gt(0.5).int()
      test_preds[ indx : (indx + batch_pred.shape[0]), :] = batch_pred.to("cpu").data.numpy()
      test_labels[ indx : (indx + batch_labels.shape[0]), :] = batch_labels.to("cpu").data.numpy()
      indx = indx + batch_pred.shape[0]
  
  accuracy = metrics.accuracy_score(test_labels, test_preds)
  print ('accuracy', accuracy)
  return accuracy, test_preds, test_labels

model = CLF_LSTM(batch_size=1)
model.load_state_dict(torch.load('./clf_lstm.pth'))
accuracy, test_pred, test_lab = test_model(model)
metrics.confusion_matrix(test_lab, test_pred)

model = CLF_LSTM(batch_size=6)
model.load_state_dict(torch.load('./clf_lstm.pth'))
model.eval()

x = torch.FloatTensor(torch.rand(6,5,4))
with torch.no_grad():
  print (model(x))
  traced_cell = torch.jit.trace(model, (x))
torch.jit.save(traced_cell, './clf_lstm_jit6.pth')

model_jit = torch.jit.load('./clf_lstm_jit6.pth')
print (model_jit(x))