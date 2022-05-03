import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def one_hot_vector(x):
    y = torch.zeros((len(x), 1 + max(x)), device='cuda:0')
    for i in range(len(x)):
        y[i, x[i]] = 1
    return y

'''
check_log_loss(loader, model)

Imported from HW22 assignment.
The following condition must be met in order to use this function;
- DataLoader class must be defined for our project.
- Model class must be defined for our project.

'''

def check_log_loss(loader, model):
  if loader.dataset.train:
    print('Checking accuracy on validation set')
  else:
    print('Checking accuracy on test set')

  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  num_correct = 0
  num_samples = 0
  log_sum = 0
  model.eval()
  with torch.no_grad():
    for x, y in loader:
      x = x.to(device=device, dtype=torch.float)
      y = y.to(device=device, dtype=torch.long)
      scores = model(x)
      # scores means classfication class for each class. It should be the tensor with size of (Input size, Number of classes)
      # In binary classification, it should be (batch size, 2) sized tensor
      _, preds = scores.max(1)
      num_correct += (preds == y).sum()
      num_samples += preds.size(0)

      y = one_hot_vector(y)
      log_loss = -(y * scores)
      log_sum += log_loss.sum()

    acc = float(num_correct) / num_samples
    print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))
    print('Log Loss score: (%.2f)' % log_sum / num_samples)
  return acc
