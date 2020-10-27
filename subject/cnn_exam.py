import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import os
import pickle
from PIL import Image
import numpy as np
import pandas as pd
from sklearn import datasets, model_selection

img_size = 256
rgb_max_size = 255.0
# data
test_size = 0.1 #(10%)
batch_size = 32
is_shuffle = True

# learning
epoch_size = 10
learning_rate = 0.001

# 
home_path = ''
train_path = '{}/plant_data/train'.format(home_path)
val_path = '{}/plant_data/test'.format(home_path)
model_path = '{}/plant.model'.format(home_path)

class CNNNet(nn.Module):
  def __init__(self, class_num):
    super(CNNNet, self).__init__()
    # output class 
    self.class_num = class_num
    # cnn kernel size
    self.filter_size = 5
    # conv1
    self.input_channel = 3 # rgb
    self.conv1_pooling_size = 3
    conv1_output_channel = 10
    conv1_output_img_size = (img_size - (self.filter_size-1)) / self.conv1_pooling_size # 84
    # conv2
    self.conv2_pooling_size = 2
    conv2_output_channel = 10+conv1_output_channel
    conv2_output_img_size = (conv1_output_img_size - (self.filter_size-1)) / self.conv2_pooling_size # 40
    # conv3
    self.conv3_pooling_size = 2
    conv3_output_channel = 10+conv2_output_channel
    conv3_output_img_size = (conv2_output_img_size - (self.filter_size-1)) / self.conv3_pooling_size # 18
    # fc1
    print(conv3_output_channel, conv3_output_img_size)
    self.in_features = int(conv3_output_img_size * conv3_output_img_size * conv3_output_channel) # 18 * 18 * 30
    print("in features", self.in_features)
    self.out_features = 50

    # img input
    self.conv1 = nn.Conv2d(self.input_channel, conv1_output_channel, self.filter_size)
    self.conv2 = nn.Conv2d(conv1_output_channel, conv2_output_channel, self.filter_size)
    self.conv3 = nn.Conv2d(conv2_output_channel, conv3_output_channel, self.filter_size)

    # in_features(input size), out_features(size)
    # 29 11600 <class 'int'> 50
    self.fc1 = nn.Linear(self.in_features, self.out_features)  # shape '[-1, 32000]' is invalid for input of size 31360
    self.fc2 = nn.Linear(self.out_features, self.class_num)

  def forward(self, x):
    x = F.max_pool2d(F.relu(self.conv1(x)), self.conv1_pooling_size)
    print("cnn 1 output shape", x.shape)
    x = F.max_pool2d(F.relu(self.conv2(x)), self.conv2_pooling_size)
    print("cnn 2 output shape", x.shape)
    x = F.max_pool2d(F.relu(self.conv3(x)), self.conv3_pooling_size)
    # cnn 3 output shape torch.Size([32, 20, 18, 18])
    # RuntimeError: shape '[-1, 32000]' is invalid for input of size 207360
    x = x.view(-1, self.in_features)
    x = F.relu(self.fc1(x))
    x = self.fc2(x)
    return F.log_softmax(x)

def read_file(fpath, fname):
  # file_name + \t + plant_label + \t + disease_label
  label = {}
  label_indices = {}
  with open('{}/{}'.format(fpath, fname), 'r') as ff:
    for line in ff.readlines():
      fname, plabel, dlabel = line.strip().split('\t')
      label[fname] = plabel, dlabel
      label_key = '{}_{}'.format(plabel, dlabel)
      if label_key not in label_indices:
        label_indices[label_key] = len(label_indices)
      print('data :', len(label), end='\r')
  return label, label_indices

def load_img_file(fpath, label_dict, label_indices):
  data = []
  labels = []
  p_labels = []
  d_labels = []

  files = os.listdir(fpath) 
  for fname in files:
    if fname[0] == '.':
        continue
    if fname[-3:] != 'jpg':
        continue

    img = Image.open('{}/{}'.format(fpath, fname))
    resize_img = img.resize((img_size, img_size))
    r, g, b = resize_img.split()
    # normalize
    r_resize_img = np.asarray(np.float32(r) / rgb_max_size)
    b_resize_img = np.asarray(np.float32(g) / rgb_max_size)
    g_resize_img = np.asarray(np.float32(b) / rgb_max_size)
    rgb_resize_img = np.asarray([r_resize_img, b_resize_img, g_resize_img])
    # 
    p_label, d_label = label_dict[fname]
    data.append(rgb_resize_img)
    label = '{}_{}'.format(p_label, d_label)
    label_idx = label_indices[label]
    labels.append(label_idx)
    p_labels.append(p_label)
    d_labels.append(d_label)
    print('data :', len(data), 'label:', len(p_labels), len(d_labels), end='\r')
  return data, p_labels, d_labels, labels


def get_data(data, labels):
  train_X, test_X, train_Y, test_Y = model_selection.train_test_split(data, labels, test_size=test_size)
  
  train_X = torch.from_numpy(train_X).float()
  train_Y = torch.from_numpy(train_Y).long()
  
  test_X = torch.from_numpy(test_X).float()
  test_Y = torch.from_numpy(test_Y).long()

  return train_X, test_X, train_Y, test_Y
    
def run(data, labels, class_num):
  ## data
  train_X, test_X, train_Y, test_Y = get_data(data, labels)
  train = TensorDataset(train_X, train_Y)
  train_loader = DataLoader(train, batch_size=batch_size, shuffle=is_shuffle)
  
  # network
  model = CNNNet(class_num)
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr=learning_rate)
  
  # run
  max_acc = -1000
  for epoch in range(epoch_size):
    total_loss = 0
    for train_x, train_y in train_loader:
      train_x, train_y = Variable(train_x), Variable(train_y)
      optimizer.zero_grad()
      output = model(train_x)
      loss = criterion(output, train_y)
      loss.backward()
      optimizer.step()
      total_loss += loss.data.item()
    print(epoch+1, total_loss/(epoch+1))
    if (epoch+1) % 50 == 0:
      print(epoch+1, total_loss)
    test_x, test_y = Variable(test_X), Variable(test_Y)
    result = torch.max(model(test_x).data, 1)[1]
    accuracy = sum(test_y.data.numpy() == result.numpy()) / len(test_y.data.numpy())
    #
    print("plants disease classification [{}] : {}".format(epoch, accuracy))
    if max_acc < accuracy:
      fname = '{}.{}_{}'.format(model_path, epoch, accuracy)
      torch.save(model.state_dict(), fname)
      max_acc = accuracy
  return accuracy

def main():
  pk_fname = 'train_data.dmp'
  
  if os.path.isfile(pk_fname):
      ff = open(pk_fname, 'rb')
      train_dataset = pickle.load(ff)
      ff.close()
      #train_dataset = pickle.load(pk_fname)
      data = train_dataset['data']
      labels = train_dataset['labels']
      label_indices = train_dataset['label_indices']
      p_labels = train_dataset['p_labels']
      d_labels = train_dataset['d_labels']
  else:
      print('start load tsv')
      label, label_indices = read_file(train_path, 'train.tsv')
      print('\n end load tsv')
      print('start load img')
      data, p_labels, d_labels, labels = load_img_file(train_path, label, label_indices)
      print('\n end load img')
  
      train_dataset = {'data':data, 'p_labels':p_labels, 'd_labels':d_labels, 'labels':labels, 'label_indices':label_indices}
      ff = open(pk_fname, 'wb')
      pickle.dump(train_dataset, ff)
      ff.close()
  
  pd.DataFrame(data[0][0]).shape
  
  data = np.array(data, dtype='float32')
  labels = np.array(labels, dtype='int64')
  run(data, labels, len(label_indices))

if __name__ == '__main__':
    main()

