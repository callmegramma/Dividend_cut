# This is a short script to help you test my final model
# The final model predicts a divident cut of at least 15% as described in the notebook

# Imports
# Data
import pandas as pd
import numpy as np
import pickle

# Processing
from sklearn.impute import KNNImputer
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

#Model
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report

#Read data
vars = ['Dividend','Dividend_returns','ASSETS', 'BPS', 'CAPEX', 'CFPS', 'CF_FIN',
       'CF_INV', 'CF_OP', 'DEPR_AMORT', 'EBIT', 'EBITDA', 'EPS', 'EPS_GAAP',
       'EPS_NONGAAP', 'FCF', 'G_A_EXP', 'INT_EXP', 'NDT', 'NET', 'NETBG',
       'PTI', 'SALES', 'SH_EQUITY', 'CAPEX_returns', 'CF_OP_returns',
       'EBIT_returns', 'EBITDA_returns', 'EPS_GAAP_returns', 'NDT_returns',
       'NET_returns', 'NETBG_returns', 'PTI_returns', 'SH_EQUITY_returns',
       'CFPS_returns', 'EPS_returns', 'SALES_returns', 'payout_ratio',
       'earnings_cover', 'cashflow_cover', 'debt_factor', 'profitability','financial_health_score']

rawdata =  pd.read_csv('your filepath', usecols=vars,na_values=["inf","-inf",""," "], keep_default_na=True)

print(vars_to_keep - Values_features)
vars_to_keep[~vars_to_keep.isin(Values_features)]
len(vars)
len(vars_to_keep)

#Process data
clean_data = rawdata.copy()
clean_data['Dividend_returns'] = clean_data['Dividend_returns'].fillna(value=0)
clean_data.drop(columns='Dividend', inplace=True)

clean_data['Div_Cut']=0
clean_data.loc[clean_data['Dividend_returns'] <= float(-0.15),'Div_Cut']=1
clean_data.drop('Dividend_returns', axis=1, inplace=True)

#Impute missing data (you can skip this if no missing data)
imputer = KNNImputer(missing_values=np.nan, n_neighbors=2, weights='uniform', metric='nan_euclidean', copy=True, add_indicator=False)
imputed_data = pd.DataFrame(imputer.fit_transform(clean_data))
#add back var names
imputed_data.columns = clean_data.columns
#add back index
imputed_data.index = clean_data.index

#Now let's add back the firm id as a variable
imputed_data["firm_id"] = imputed_data.index.get_level_values(0)
#and then remove it from the index
imputed_data.index = imputed_data.index.droplevel(0)

#Get data to test on
X_test = imputed_data.loc[2017,imputed_data.columns != 'Div_Cut']
y_test = imputed_data.loc[2017,'Div_Cut']

#Now let's transform our data
#Encode firm_id as dummy var
Firm_Encoder = OneHotEncoder(categories='auto', drop=None, sparse=True, handle_unknown='ignore')
#Scale features that are not finanial_health_score or firm_id by centering around the mean
Values_Encoder = StandardScaler()

Values_features = ['ASSETS', 'BPS', 'CAPEX', 'CFPS', 'CF_FIN',
       'CF_INV', 'CF_OP', 'DEPR_AMORT', 'EBIT', 'EBITDA', 'EPS', 'EPS_GAAP',
       'EPS_NONGAAP', 'FCF', 'G_A_EXP', 'INT_EXP', 'NDT', 'NET', 'NETBG',
       'PTI', 'SALES', 'SH_EQUITY', 'CAPEX_returns', 'CF_OP_returns',
       'EBIT_returns', 'EBITDA_returns', 'EPS_GAAP_returns', 'NDT_returns',
       'NET_returns', 'NETBG_returns', 'PTI_returns', 'SH_EQUITY_returns',
       'CFPS_returns', 'EPS_returns', 'SALES_returns', 'payout_ratio',
       'earnings_cover', 'cashflow_cover', 'debt_factor', 'profitability']

#Now we can apply the transformations to the data

data_transformer = ColumnTransformer(
    transformers=[
        ('firm', Firm_Encoder,['firm_id']),
        ('Values_Enc_1',Values_Encoder, Values_features)
    ],remainder='passthrough')

X_trans = data_transformer.transform(X_test)

#Check my model
Final_model_PI = pickle.load(open('your filepath/Final_model_Plamen_Ivanov.sav','rb'))
Final_predict = Final_model_PI.predict(X_trans)
print(classification_report(y_test, Final_predict))