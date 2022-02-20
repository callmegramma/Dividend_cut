#Imports
#Data
import pandas as pd
import numpy as np
import itertools
import pickle
#import plotly as plt
#pd.options.plotting.backend = "plotly"
#import matplotlib.pyplot as mlt

#Processing
from sklearn.impute import KNNImputer
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

#Models
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV

#Metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
#from sklearn.metrics import DetCurveDisplay
#from sklearn.metrics import det_curve
from sklearn.metrics import fbeta_score

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
#check.plot().show()

#Let's see how cuts vary across years at this threshold
check = clean_data[['Div_Cut']].groupby(axis='index', level=[1]).sum()
check.sort_values(by='Div_Cut', axis=0, ascending=False)
#check.plot().show()

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
# One way to do this is to backtest a few years of data for all firms. The problem is this will
# leaves us with an unballanced panel by firm as not all firms have data since 2008
# Another way to do this is to hold out data for some of the firms each time, but use all years
# A third option is to use all data up to 2016 for all firms to train and test on 2016,
# given that is the year closest to the hold-out year (2017)

#I have chosen the third option because all firms may be included in the hold-out set
# and I can reduce overfitting by cross-validation

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
print('Check that X and y are equal')
print(len(X_test)==len(y_test),"for X and y test")
print(len(X_train)==len(y_train),"for X and y train")

print('Check I split the entire set')
print(len(X_test)+len(X_train)==len(imputed_data))

#Now let's transform our train data
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


#Fit to the data and transform it
data_transformer.fit(X_train)
X_train_trans = data_transformer.transform(X_train)

#Let's try out a few reliable models to understandhow they perform

#Fit a Logit model on the data
Logit_model = LogisticRegression()
Logit_model.fit(X_train_trans, y_train)
Logit_predictions = Logit_model.predict(data_transformer.transform(X_test))

#Fit a FR model on the data

RF_model = RandomForestClassifier()
RF_model.fit(X_train_trans, y_train)
RF_predictions = RF_model.predict(data_transformer.transform(X_test))

#Fit a SVM model on the data
SVC_model = LinearSVC(fit_intercept=False,max_iter=100000)
SVC_model.fit(X_train_trans, y_train)
SVC_predictions = SVC_model.predict(data_transformer.transform(X_test))



#Let's see if we can improve the Random Forest adn SVC models through cross validation, by taking out some firms each time
#This should help reduce overfitting on the data and improve performance on the hold-out set
#To make sure I am taking out firms as a whole I am using the GroupKFold cross validator

# Cross validated Random Forest
parameters_RF = {
 'criterion': ('gini', 'entropy'),
 'min_samples_leaf': (8,24,48),
 }

RF_CV_model = GridSearchCV(RF_model, parameters_RF, cv=GroupKFold(10), n_jobs=-1)

RF_CV_model.fit(X_train_trans, y_train, groups=X_train['firm_id'])

RF_CV_predictions = RF_CV_model.predict(data_transformer.transform(X_test))

print('Best parameters for RF are: ' + str(RF_CV_model.best_params_))

# Cross validated SVC
parameters_SVC = {
 'fit_intercept': (True, False),
 'C': (1,5,10,15),
 }

SVC_CV_model = GridSearchCV(SVC_model, parameters_SVC, cv=GroupKFold(10), n_jobs=-1)

SVC_CV_model.fit(X_train_trans, y_train, groups=X_train['firm_id'])

SVC_CV_predictions = SVC_CV_model.predict(data_transformer.transform(X_test))

print('Best parameters for SCV are: ' + str(SVC_CV_model.best_params_))

#Evaluation

#Logit
print(classification_report(y_test, Logit_predictions))
confusion_matrix(y_test, Logit_predictions)

#SVC
print(classification_report(y_test, SVC_predictions))
confusion_matrix(y_test, SVC_predictions)

#Random Forest
print(classification_report(y_test, RF_predictions))

#Random Forest CV
print(classification_report(y_test, RF_CV_predictions))

#SVC CV
print(classification_report(y_test, SVC_CV_predictions))

#Extract confusion values
tn, fp, fn, tp = confusion_matrix(y_test, SVC_CV_predictions).ravel()
tn, fp, fn, tp = confusion_matrix(y_test, RF_CV_predictions).ravel()
tn, fp, fn, tp = confusion_matrix(y_test, Logit_predictions).ravel()
del(tn, fp, fn, tp)
del(row, model, name)
#Compare models

#Let's set a framework for comparing models
#Blah
#Blah

#Initial

model_comparison=pd.DataFrame(index=[0,1,2,3,4,5],columns=['Model','F1 Score','F0.5 Score','True Negative', 'False Positive','False Negative','True Positive'])

for model, name, row in zip([Logit_predictions,RF_predictions,SVC_predictions],["Logit","RF","SVC"],[0,1,2]):
    print(model,name,row)
    tn, fp, fn, tp = confusion_matrix(y_test, model).ravel()
    model_comparison.loc[row] = [name,fbeta_score(y_test,model,beta=1),fbeta_score(y_test,model,beta=0.5),tn, fp, fn, tp]

#Cross validated

for model, name, row in zip([RF_CV_predictions,SVC_CV_predictions],["RF_CV","SVC-CV"],[3,4]):
    print(model,name,row)
    tn, fp, fn, tp = confusion_matrix(y_test, model).ravel()
    model_comparison.loc[row] = [name,fbeta_score(y_test,model,beta=1),fbeta_score(y_test,model,beta=0.5),tn, fp, fn, tp]

#Bonus task: Variables are only known with a 1 year lag

#The current set-up allows me to evaluate how much performance will degrade by simpy removing 2015 from the training data
#Let's have a quick check
X_train = imputed_data.loc[range(2008,2015),imputed_data.columns != 'Div_Cut']
y_train = imputed_data.loc[range(2008,2015),'Div_Cut']

#Check that nothing went wrong
print('Check that X and y are equal')
print(len(X_test)==len(y_test),"for X and y test")
print(len(X_train)==len(y_train),"for X and y train")

#Fit to the data and transform it
data_transformer.fit(X_train)
X_train_trans = data_transformer.transform(X_train)

# Usinhg the best params from the cross validated SVC (I could cross-validate on this training set as well if needed)

SVC_Lag1_model = LinearSVC(fit_intercept=True,C=1,max_iter=100000)
SVC_Lag1_model.fit(X_train_trans, y_train)
SVC_Lag1_predictions = SVC_Lag1_model.predict(data_transformer.transform(X_test))

#At a glance it seems performance doesn't degrade much
print(classification_report(y_test, SVC_Lag1_predictions))

#Let's compare with the other models

tn, fp, fn, tp = confusion_matrix(y_test,SVC_Lag1_predictions).ravel()
model_comparison.loc[5] = ["SVC_Lag1_predictions",fbeta_score(y_test,SVC_Lag1_predictions,beta=1),fbeta_score(y_test,SVC_Lag1_predictions,beta=0.5),tn, fp, fn, tp]
del(tn, fp, fn, tp)

#It seems by excluding the data for 2015 we get more more True positives but also more False positives
#Likely some imformation loss occurs

###Finally let's train the best model (SVM) using all data (including 2016) and pickle it
Final_model = LinearSVC(fit_intercept=True,C=1,max_iter=100000)

#Let's change the training data
X_train = imputed_data.loc[range(2008,2017),imputed_data.columns != 'Div_Cut']
y_train = imputed_data.loc[range(2008,2017),'Div_Cut']

#Check that nothing went wrong
print('Check that X and y are equal')
print(len(X_train)==len(y_train),"for X and y train")

#Fit and transform the data
data_transformer.fit(X_train)
X_train_trans = data_transformer.transform(X_train)
Final_model.fit(X_train_trans, y_train)

Final_predict = Final_model.predict(X_train_trans)
#print(classification_report(y_train, Final_predict))

#Pickle the model
filepath = 'C:/Users/Admiral Akhbar/Desktop/Markit/Final_model_Plamen_Ivanov.sav'

pickle.dump(Final_model, open(filepath,'wb'))

#test it worked
model_check = pickle.load(open(filepath,'rb'))
check_predict = model_check.predict(X_train_trans)

#Check if I saved the final model correctly
#If I did this list shoudl be empty
Final_predict[(Final_predict != check_predict)]

#Display evaluations

#ConfusionMatrixDisplay(confusion_matrix(y_test, Logit_predictions)).plot()

#d_curve = det_curve(y_test,Logit_predictions)

#DetCurveDisplay(d_curve).plot()

#mlt.show()

