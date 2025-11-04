import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import os
import joblib
import sys
import time

from spotlight.datasets.movielens import get_movielens_dataset
from spotlight.interactions import *
from collections import defaultdict
from spotlight.factorization.explicit import ExplicitFactorizationModel
from spotlight.cross_validation import *
from spotlight.evaluation import rmse_score

from sklearn.utils import murmurhash3_32
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR

from spotlight.factorization.representations import BilinearNet

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
dataset_100k = joblib.load(os.getcwd() + "/Mlens_100K.joblib")
                                       
model = ExplicitFactorizationModel(loss='regression',
           embedding_dim=128,
           n_iter=100, 
           batch_size=256,
           l2=0, 
           learning_rate=0.001,
           sep_penalty=True,
           item_penlam=0.001,
           user_penlam=0.00001,                 
           jnt_penalty=False,
           all_penlam=0,
           optimizer_func=torch.optim.SGD,
           lr_sched=True,
           use_cuda=torch.cuda.is_available())

train, test = random_train_test_split(dataset_100k, p=0.2)

model.fit(train, test, verbose=False)

train_rmse = rmse_score(model, train)
test_rmse = rmse_score(model, test)
print('final trainRMSE {:.3f}, final testRMSE {:.3f}'.format(train_rmse, test_rmse))
