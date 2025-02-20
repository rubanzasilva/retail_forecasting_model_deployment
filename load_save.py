import numpy as np
from numpy import random
#import fastbook
#fastbook.setup_book()
#from fastbook import *
from fastai.tabular.all import *
import pandas as pd
#import matplotlib.pyplot as plt
#from fastai.imports import *
#np.set_printoptions(linewidth=130)
from pathlib import Path
import os
import xgboost as xgb
#from xgboost import plot_importance
from xgboost import XGBRegressor
import warnings
import gc
import pickle
from joblib import dump, load
import typing as t
import bentoml
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import KFold, cross_val_score,train_test_split
#import wandb


torch.manual_seed(42)

path = Path('data/')

train_df = pd.read_csv(path/'train.csv',index_col='id')
test_df = pd.read_csv(path/'test.csv',index_col='id')
sub_df = pd.read_csv(path/'sample_submission.csv')

train_df = train_df.dropna(subset=['num_sold'])
train_df = add_datepart(train_df,'date',drop=False)
test_df = add_datepart(test_df,'date',drop=False)

cont_names,cat_names = cont_cat_split(train_df, dep_var='num_sold')
splits = RandomSplitter(valid_pct=0.2)(range_of(train_df))
to = TabularPandas(train_df, procs=[Categorify, FillMissing,Normalize],
#to = TabularPandas(train_df, procs=[Categorify,Normalize],
                   cat_names = cat_names,
                   cont_names = cont_names,
                   y_names='num_sold',
                   y_block=CategoryBlock(),
                   splits=splits)
dls = to.dataloaders(bs=64)
#dls = to.dataloaders(bs=1024)
test_dl = dls.test_dl(test_df)

X_train, y_train = to.train.xs, to.train.ys.values.ravel()
X_test, y_test = to.valid.xs, to.valid.ys.values.ravel()

xgb_model = xgb.XGBRegressor()
xgb_model = xgb_model.fit(X_train, y_train)

#xgb_preds = tensor(xgb_model.predict(test_dl.xs))

#xgb_preds_x = tensor(xgb_model.predict(X_test))

#xgb_score = mean_absolute_percentage_error(y_test,xgb_preds_x)
#print(f"Model accuracy: {xgb_score}")

bentoml.xgboost.save_model("sticker_sales_v1", xgb_model)