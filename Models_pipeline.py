#Imports
#Data
import pandas as pd
import numpy as np
#import plotly as plt
#pd.options.plotting.backend = "plotly"
#import matplotlib.pyplot as mlt

#Processing
from sklearn.impute import KNNImputer
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

#Models
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier #fails for some reason
from sklearn.linear_model import SGDClassifier

#Metrics
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import DetCurveDisplay
from sklearn.metrics import det_curve

# Becasue we're working with Balance sheet / Profit Statement items we need can't just backfill/pad missing values
# We could try to impute missing values using MICE (multiple interative imputation) and some sensible financial relationships
# However, this will likely be industry-specific (and GAAP/IFRS specific),
# With no industry variable here, no units (currency, million/billion) for many of the other variables, I'll need something simple

#First off, let's fix the dividend return variable
#It's ok to pad with 0 here as missing values are due to time cut-off
clean_data = rawdata.copy()
clean_data['Dividend_returns'] = clean_data['Dividend_returns'].fillna(value=0)

#Let's also remove the Dividents variable, since we're predicting a cut in this scenario
clean_data.drop(columns='Dividend', inplace=True)

#Now let's encode a target variable, with a given threshold
#In practice you would presumably want to set a threshold based on market reaction,
#for example share price performance in a given period after the cut
threshold = float(-0.15)
clean_data['Div_Cut']=0
clean_data.loc[clean_data['Dividend_returns'] <= threshold,'Div_Cut']=1

#Let's see how many firms had a cut at that threshold
check = clean_data[['Div_Cut']].groupby(axis='index', level=[0]).mean()
check.sort_values(by='Div_Cut', axis=0, ascending=False, inplace=True)

print(len(check[check['Div_Cut']>0])," out of ",len(check)," firms or ",
      round(len(check[check['Div_Cut']>0])/len(check),2)*100, "% of all firms had a cut at the ",
      threshold, " threshold")

#Let's see a rough distribution (across years) of cuts by firm
check.plot().show()

#Let's see how cuts vary across years at this threshold
check = clean_data[['Div_Cut']].groupby(axis='index', level=[1]).sum()
check.sort_values(by='Div_Cut', axis=0, ascending=False)
check.plot().show()

#Ok this threshold seems reasonable to proceed

#Now let's try KNN imputation,
#If we model using pure sk-learn we could also inlcude a mask
#to indicate the number of missing values, which I suspect will help

# I think it's good practiceto make all the key parameters for the imputer explicit here
imputer = KNNImputer(missing_values=np.nan, n_neighbors=2, weights='uniform', metric='nan_euclidean', copy=True, add_indicator=False)

# We are using all the data here, which is a bit of a cheat (I am doing this before the test/train split
# and so test data info will "leak" in the train dataset)
# However, it shouldn't matter when it comes to evaluating performance on the 2017 hold-out set
# Imputing the missing values during the test/train split with KNN can result in extra noise in the model performance
# Given that we have a lot imputing to do, I have chosen to do it pre test-train split
imputed_data = pd.DataFrame(imputer.fit_transform(clean_data))
#add back var names
imputed_data.columns = clean_data.columns
#add back index
imputed_data.index = clean_data.index

#Now let's drop divident returns as a column as this will not be know in the prediction year
imputed_data.drop('Dividend_returns', axis=1, inplace=True)

#Now let's add back the firm id as a variable
imputed_data["firm_id"] = imputed_data.index.get_level_values(0)
#and then remove it from the index
imputed_data.index = imputed_data.index.droplevel(0)

###Split dataset for training
# One way to do this is to backtest a few years of data for all firms
# Another way to do this is to hold out data for some of the firms each time, but use all past years
#A third option is to mix the first two: backtest on a sample of firms
#I have chosen the second option because I all firms may not be included in the hold-out set
# and I want to reduce overfitting

#Set Seed to reproduce results
#seed = 18531856

#As far as I can see there's no test_train split that acounts for both groups and time in sklearn
#Therefore, I will write my own, even if this can be a bit dangerous in my experience

# I want to test the models on all firms for 2016
# Because the data is already sorted by year and firm_id (as this was the index)
# We could take every 6th observation, if all firms had existed in at the start of the data in 2008
# Let's check
print("Avg years per firm are ", str(len(imputed_data) / len(imputed_data["firm_id"].unique())))

# Sadly not all firms have existed since 2008
# Therefore I must take the 2016 observation for each firm for the X_test set
# and leave the rest for the X_train set which leaves us with a slighly unequal panel by firm

X_test = imputed_data.loc[2016,imputed_data.columns != 'Div_Cut']
y_test = imputed_data.loc[2016,'Div_Cut']
X_train = imputed_data.loc[range(2008,2016),imputed_data.columns != 'Div_Cut']
y_train = imputed_data.loc[range(2008,2016),'Div_Cut']

#Check that nothing went wrong
print('Check that X andn y are equal')
print(len(X_test)==len(y_test),"for X and y test")
print(len(X_train)==len(y_train),"for X and y train")

print('Check I split the entire set')
print(len(X_test)+len(X_train)==len(imputed_data))

#Now let's transform our train data
#Encode firm_id as dummy var
Firm_Encoder = OneHotEncoder(categories='auto', drop=None, sparse=True, handle_unknown='error')
#Scale features that are not divident_retuns, finanial_health_score or firm_id by removing the mean
Values_Encoder = RobustScaler(with_centering=True, with_scaling=True, quantile_range=(25.0, 75.0), copy=True)
Values_Encoder_2 = StandardScaler()

Values_features = ['ASSETS', 'BPS', 'CAPEX', 'CFPS', 'CF_FIN',
       'CF_INV', 'CF_OP', 'DEPR_AMORT', 'EBIT', 'EBITDA', 'EPS', 'EPS_GAAP',
       'EPS_NONGAAP', 'FCF', 'G_A_EXP', 'INT_EXP', 'NDT', 'NET', 'NETBG',
       'PTI', 'SALES', 'SH_EQUITY', 'CAPEX_returns', 'CF_OP_returns',
       'EBIT_returns', 'EBITDA_returns', 'EPS_GAAP_returns', 'NDT_returns',
       'NET_returns', 'NETBG_returns', 'PTI_returns', 'SH_EQUITY_returns',
       'CFPS_returns', 'EPS_returns', 'SALES_returns', 'payout_ratio',
       'earnings_cover', 'cashflow_cover', 'debt_factor', 'profitability']

#Check we have the right features
X_train.columns[~X_train.columns.isin(Values_features)]

#Now we can apply the transformations to the data
#Let's set_up a transformer object
#This weird list usage (for firm_id) took me a while to figure out https://github.com/scikit-learn/scikit-learn/issues/14056
data_transformer = ColumnTransformer(
    transformers=[
        ('firm', Firm_Encoder,['firm_id']),
        ('Values_Enc_1',Values_Encoder, Values_features)
    ],remainder='passthrough')

data_transformer_2 = ColumnTransformer(
    transformers=[
        ('firm', Firm_Encoder,['firm_id']),
        ('Values_Enc_1',Values_Encoder_2, Values_features)
    ],remainder='passthrough')

#Fit to the data and transform it
data_transformer_2.fit(X_train)
X_train_trans = data_transformer_2.transform(X_train)

#Fit a Logit model on the data -
# CV Fails to converge
Logit_model = LogisticRegression()

Logit_model.fit(X_train_trans, y_train)

Logit_predictions = Logit_model.predict(data_transformer_2.transform(X_test))

#Fit a FR model on the data - Fails to converge

RF_model = RandomForestClassifier()

RF_model.fit(X_train_trans, y_train)

RF_predictions = RF_model.predict(data_transformer_2.transform(X_test))

# I want to train the model via a cross-validated sample of firms

#Evaluation

#Logit
#d_curve = det_curve(y_test,Logit_predictions)
print(classification_report(y_test, Logit_predictions))
confusion_matrix(y_test, Logit_predictions)

#Random Forest
#d_curve = det_curve(y_test,RF_predictions)
print(classification_report(y_test, RF_predictions))
confusion_matrix(y_test, RF_predictions)

tn, fp, fn, tp = confusion_matrix(y_test, Logit_predictions).ravel()
del(tn, fp, fn, tp)

#Display evaluations

ConfusionMatrixDisplay(confusion_matrix(y_test, Logit_predictions)).plot()

DetCurveDisplay(d_curve).plot()

mlt.show()

