#Imports
#Models
import pandas as pd
import plotly as plt
pd.options.plotting.backend = "plotly"

from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

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

print(str(len(check[check['Div_Cut']>0])) + " out of " + str(len(check)) + " firms or " +
      str(round(len(check[check['Div_Cut']>0])/len(check),2)*100) + "% of all firms had a cut at the "
      + str(threshold) + " threshold")

#Let's see a rough distribution (across years) of cuts by firm
check.plot().show()

#Let's see how cuts vary across years at this threshold
check = clean_data[['Div_Cut']].groupby(axis='index', level=[1]).mean()
check.sort_values(by='Div_Cut', axis=0, ascending=False)
check.plot().show()

#Ok this threshold seems reasonable to proceed


#Now let's try KNN imputation, with a mask to indicate the number of missing values
knn_imputer = KNNImputer(, missing_values=nan, n_neighbors=5, weights='uniform', metric='nan_euclidean', copy=True, add_indicator=False)
