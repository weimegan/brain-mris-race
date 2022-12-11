from __future__ import print_function 
from __future__ import division
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import skimage.transform as st
import matplotlib.pyplot as plt
import os
from utils import *
import random
from itertools import product

root_path = 'data/T1_images'

random.seed(0)

# from pandas.core.dtypes.base import E
class BrainDataset(Dataset):
  def __init__(self, data_dir, predictor, images_per_group):
    # super(BrainIterableDataset).__init__()
    self.data_dir = data_dir
    self.predictor = predictor
    self.images_per_group = images_per_group
    self.metadata = pd.read_csv(f'{root_path}/subj_data.csv')
    self.metadata = self.metadata.loc[self.metadata['ETHNIC_ID'] != 0]
    self.all_subject_combs = self.get_augmentation_combinations()
    self.onehot_race = pd.get_dummies(self.metadata['ETHNIC_ID']).values
    self.labels = [self.get_labels(img[0]) for img in self.all_subject_combs]

  def __len__(self):
    return len(self.all_subject_combs)

  def get_augmentation_combinations(self):
    subjects = np.array([img.split('_')[0] for img in sorted(os.listdir(self.data_dir)) if img.split('_')[0] in self.metadata['subjID'].values])
    race = np.array([self.metadata.loc[self.metadata['subjID'] == subj]['ETHNIC_ID'].values[0] for subj in subjects])

    slices = ['minus5', 'minus10', '', 'plus5', 'plus10']
    flip = ['flip1', 'noflip', 'flip2', 'flip12']
    resize = ['original', 'small', 'smaller']

    all_subject_combs = []

    for group in np.unique(race):
      indices = np.where(race == group, True, False)
      subj_in_group = subjects[indices]

      combs = list(product(subj_in_group, slices, flip, resize))
      random.shuffle(combs)
      combs = combs[:self.images_per_group]

      all_subject_combs += combs
    return all_subject_combs

  def get_labels(self, subject_id):
    if self.predictor == 'sex':
      label = self.metadata.loc[self.metadata['subjID'] == subject_id]['SEX_ID'].values[0] - 1
    elif self.predictor == 'race':
      label = torch.tensor(self.onehot_race[(self.metadata['subjID'] == subject_id).values][0], dtype=torch.float32)
    elif self.predictor == 'age':
      label = self.metadata.loc[self.metadata['subjID'] == subject_id]['AGE'].values[0]
    return label

  def __getitem__(self, idx):
    IMAGE_SIZE = (224, 224)
    if self.all_subject_combs[idx][1] == '':
      filename = f'{self.all_subject_combs[idx][0]}_{SLICE}'
    else:
      filename = f'{self.all_subject_combs[idx][0]}_{SLICE}_{self.all_subject_combs[idx][1]}'
    image = np.load(f'{self.data_dir}/{filename}.npy')

    if self.all_subject_combs[idx][2] == 'flip1':
      image = image[::-1]
    elif self.all_subject_combs[idx][2] == 'flip2':
      image == image[:,::-1]
    elif self.all_subject_combs[idx][2] == 'flip12':
      image = image[::-1, ::-1]

    if self.all_subject_combs[idx][3] == 'original':
      pass
    else:
      if self.all_subject_combs[idx][3] == 'small':
        shrink = 90/100
      else: 
        shrink = 80/100

      pad = (1-shrink) /2
      smaller_image = st.resize(image, (int(len(image)*(shrink)), int(len(image[1])*(shrink))))
      smaller_image = np.pad(smaller_image, pad_width=((int(len(image)*pad), int(len(image)*pad)), (int(len(image[1])*pad), int(len(image[1])*pad))))
      image = smaller_image

    resized_image = st.resize(image, IMAGE_SIZE)
    image_3channel = torch.from_numpy(np.tile(np.expand_dims(resized_image, 0), (3,1,1)))

    subject_id = filename.split('_')[0]
    label = self.get_labels(subject_id)
    
    return image_3channel, label


if __name__ == '__main__':
  PREDICTOR = 'sex'
  SLICE = 'sagittal'
  data_dir = f'{root_path}/{SLICE}' 

  # Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
  model_name = "resnet"

  # Number of classes in the dataset
  num_classes = 2 if PREDICTOR == 'sex' else 4

  # Batch size for training (change depending on how much memory you have)
  batch_size = 3

  # Number of epochs to train for 
  num_epochs = 50

  # Flag for feature extracting. When False, we finetune the whole model, 
  #   when True we only update the reshaped layer params
  feature_extract = True

  print(f'Predict: {PREDICTOR}, Slice: {SLICE}')
  print(f'Model: {model_name}, # Classes: {num_classes}, Batch Size: {batch_size}, Epochs: {num_epochs}')

  dataset = BrainDataset(data_dir, PREDICTOR, 800)
  train_size = int(0.7*len(dataset))
  val_size = int(0.1*len(dataset))
  test_size = int(len(dataset) - train_size - val_size)
  train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])
  train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
  val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)
  test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)
  dataloaders_dict = {"train": train_loader, "val": val_loader, "test": test_loader}

  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

  # Initialize the model for this run
  model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)

  # Print the model we just instantiated
  print(model_ft)

  # Send the model to GPU
  model_ft = model_ft.to(device)

  # Gather the parameters to be optimized/updated in this run. If we are
  #  finetuning we will be updating all parameters. However, if we are 
  #  doing feature extract method, we will only update the parameters
  #  that we have just initialized, i.e. the parameters with requires_grad
  #  is True.
  params_to_update = model_ft.parameters()
  print("Params to learn:")
  if feature_extract:
      params_to_update = []
      for name,param in model_ft.named_parameters():
          if param.requires_grad == True:
              params_to_update.append(param)
              print("\t",name)
  else:
      for name,param in model_ft.named_parameters():
          if param.requires_grad == True:
              print("\t",name)

  # Observe that all parameters are being optimized
  optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)


  # Setup the loss fxn
  criterion = nn.CrossEntropyLoss()

  # Train and evaluate
  model_ft, hist = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, PREDICTOR, num_epochs=num_epochs, is_inception=(model_name=="inception"))

  torch.save(model_ft.state_dict(), f'models/{model_name}_{PREDICTOR}_{SLICE}_lr=0.001.pth')
  test_model(model_ft, dataloaders_dict, PREDICTOR)