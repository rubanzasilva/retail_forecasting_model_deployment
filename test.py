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

# Load the model by setting the model tag
booster = bentoml.xgboost.load_model("sticker_sales_v1:xtxsvug5wgsxwaav")

path = Path('data/')
test_df = pd.read_csv(path/'test.csv',index_col='id')

train_df = pd.read_csv(path/'train.csv',index_col='id')

train_df = train_df.dropna(subset=['num_sold'])

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
test_dl = dls.test_dl(test_df)
res = tensor(booster.predict(test_dl.xs))
print(res)

