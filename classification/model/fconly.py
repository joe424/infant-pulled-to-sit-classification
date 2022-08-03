import math

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

import time

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.fc1 = nn.Linear(1440, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 16)
        self.fc4 = nn.Linear(16, 3)
        self.fc5 = nn.Linear(512, 3)
        self.fc6 = nn.Linear(1440, 3)
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.2)
        self.dropout4 = nn.Dropout(0.2)
        self.fc7 = nn.Linear(864, 256)
        self.fc8 = nn.Linear(256, 16)
        self.fc9 = nn.Linear(16, 3)
        
    def forward(self, x):
        x = x.contiguous().view(x.shape[0], -1)
        x = self.fc7(x)
        x = self.fc8(x)
        x = self.fc9(x)
        return x
#         x = self.fc1(x)
#         x = self.dropout1(x)
#         x = self.fc2(x)
#         x = self.dropout2(x)
#         x = self.fc3(x)
#         x = self.dropout3(x)
#         x = self.fc4(x)
#         x = self.dropout4(x)
#         return x
        
#         x = self.fc1(x)
#         x = self.dropout1(x)
#         x = self.fc5(x)
#         x = self.dropout2(x)
#         return x
        
        
#         return self.dropout(x)
#         return self.fc6(x)