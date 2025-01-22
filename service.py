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

@bentoml.service(
    resources={"cpu": "8"},
    traffic={"timeout": 10},
)

class StickerSalesRegressor:
    #retrieve the latest version of the model from the BentoML model store
    bento_model = bentoml.models.get("sticker_sales_v1:latest")
    #bento_model = BentoModel('mental_health_v1:q5kcqtf5ys3qoaav')


    def __init__(self):
        self.model = bentoml.xgboost.load_model(self.bento_model)

    def preprocess(self, data):
        path = Path('data/')
        train_df = pd.read_csv(path/'train.csv',index_col='id')
        test_df = pd.read_csv(path/'test.csv',index_col='id')
        train_df = train_df.dropna(subset=['num_sold'])
        cont_names,cat_names = cont_cat_split(train_df, dep_var='num_sold')
        splits = RandomSplitter(valid_pct=0.2)(range_of(train_df))
        to = TabularPandas(train_df, procs=[Categorify, FillMissing,Normalize],
                           cat_names = cat_names,
                           cont_names = cont_names,
                           y_names='num_sold',
                           y_block=CategoryBlock(),
                           splits=splits)
        dls = to.dataloaders(bs=64)
        test_dl = dls.test_dl(data)
        test_df_new = test_dl.xs
        return test_df_new
    
    #def preprocess(self, train_filepath, test_filepath):
        #train_df = pd.read_csv(train_filepath)
        #test_df = pd.read_csv(test_filepath)
        #cont_names,cat_names = cont_cat_split(train_df, dep_var='Depression')
        #splits = RandomSplitter(valid_pct=0.2)(range_of(train_df))
        #to = TabularPandas(train_df, procs=[Categorify, FillMissing,Normalize],
                           #cat_names = cat_names,
                           #cont_names = cont_names,
                           #y_names='Depression',
                           #y_block=CategoryBlock(),
                           #splits=splits)
        #dls = to.dataloaders(bs=64)
        #test_dl = dls.test_dl(test_df)
        #test_df_new = test_dl.xs
        #return test_df_new

    @bentoml.api
    def predict(self, data:pd.DataFrame) -> np.ndarray:
        data = self.preprocess(data)
      # data = preprocess(data)


        prediction = self.model.predict(data)
        #prediction = torch.tensor(prediction)

        return prediction
       #if prediction == 0:
        #   status = "No Depression"
       #elif prediction == 1:
        #   status = "Depression"
       #else:
        #   status = "Error"
       #return status
       
        #Name = data.get("Name")
        #name_id = data.get("Name")
        
        
        #return {"prediction": prediction, "Name": Name}
        
        #return 

        #return self.model.predict(data)
    
    @bentoml.api()
    def predict_csv(self,csv:Path) -> np.ndarray:
        csv_data = pd.read_csv(csv)
        csv_data = self.preprocess(csv_data)
        prediction_csv = self.model.predict(csv_data)
        return prediction_csv
    

