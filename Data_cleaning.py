#installs
#plotly

#Imports
#Cleaning
import pandas as pd
import plotly as plt
pd.options.plotting.backend = "plotly"

#Load_raw_data
rawdata =  pd.read_csv('C:/Users/Admiral Akhbar/Desktop/Markit/ds_assessment_data_2008_2016.csv', na_values=["inf","-inf",""," "], keep_default_na=True)

#Set output options to explore data easily
pd.set_option("display.max_rows", 50)
pd.set_option("display.max_columns", 20)

#Check out the data to see if it loaded ok
rawdata.head(10)
rawdata.shape
rawdata.describe()

#Initial missing value check
missing_values = pd.DataFrame(rawdata.isnull().sum(),columns=['NA_count'])
missing_values['var'] = missing_values.index
#can aslo use df.reset_index(, inplace=True)

missing_values.sort_values(by='NA_count',ascending=False,inplace=True)

#Let's spend some time exploring the summary stats
summary_stats = pd.DataFrame(rawdata.describe())

#From the summary stats it's clear we have some infinite values in the raw data here
#I have chosen to fix this by setting them as NA in the data read_csv

# Now let's drop variables with a lot of missing values that are also highly correlated with variables we keep
# These vars are probably not worth imputing/fixing
# An alternative approach here would be to conisder dropping firms with lots of missing values across variables
# However, without knowing much about the firms in the data I believe this can bias the data and model results

#We have 856 year-company observations in the dataset
#Let's first consider throwing away variables with less than half of year-company observations
#In this case I set the threshold at 450 obs
vars_to_drop = missing_values[missing_values['NA_count']>=350]['var']
#This gives us 62 candidates to drop

#Let's set up the data as a panel dataset with id and year
rawdata.set_index(['id','fiscal_year_end_year'], inplace=True, verify_integrity=True)

#Let's get rid of calender year, we won't need it now as year is in the index
rawdata.drop(columns='calendar_end', inplace=True)

#Set up vars that we don't want to drop to check correlations
vars_to_keep = rawdata.columns.to_series()
vars_to_keep = vars_to_keep[~vars_to_keep.isin(vars_to_drop)]

#Set_up a full correlation matrix
cor_matrix_full = rawdata.corr()

#Now find max corr for a the variables we want to throw out and make a call
#There's probably a package that does this properly for panel data but I struggled to find one, so did it by hand
cor_matrix_check = pd.DataFrame(index=vars_to_drop)
cor_matrix_drop = cor_matrix_full.loc[vars_to_drop,vars_to_keep]
#Bring out index values for easy intrective sorting
cor_matrix_check["var_to_drop"] = cor_matrix_check.index.values

#Find max corr
cor_matrix_check['Max_corr']=cor_matrix_drop.max(axis=1)

#Find vars that have maxx corr
#Get variables to keep names
cor_matrix_drop = pd.melt(cor_matrix_drop, ignore_index=False, value_name='corr')
#Leave only variables where correlation matches max cor
cor_matrix_check = pd.merge(cor_matrix_drop,cor_matrix_check,how='inner',left_index=True,right_index=True)
cor_matrix_check = cor_matrix_check.loc[cor_matrix_check['corr']==cor_matrix_check['Max_corr'],:]
print(cor_matrix_check)

#Based on inspecting cor_matrix_check I decided not to drop the following variables:
# FCF
# as they/it have/has no sensible (from a financial analysis point of view) proxies left in the dataset
vars_to_drop = vars_to_drop.drop(['FCF'])
vars_to_keep = pd.concat([vars_to_keep,pd.Series(['FCF'])],ignore_index=True)

#Let's drop the other vars and check missing values again
rawdata.drop(columns=vars_to_drop, inplace=True)

#Second missing value check
missing_values = pd.DataFrame(rawdata.isnull().sum(),columns=['NA_count'])
missing_values.sort_values(by='NA_count',ascending=False,inplace=True)


