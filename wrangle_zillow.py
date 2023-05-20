import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
# houses my function to connect to Codeup DB
import wrangle
import os
import env
from sklearn.model_selection import train_test_split
pd.set_option('display.max_columns', None)


def split_data(df):
    '''
    Takes in a dataframe and returns train, validate, test subset dataframes
    '''
    
    
    train, test = train_test_split(df,
                                   test_size=.2, 
                                   random_state=123, 
                                   
                                   )
    train, validate = train_test_split(train, 
                                       test_size=.25, 
                                       random_state=123, 
                                       
                                       )
    
    return train, validate, test


def new_zillow_data(SQL_query, url):
    '''
    this function will:
    - take in a SQL_query 
    -create a connection url to mySQL
    -return a df of the given query from the zillow database
    
    '''
    
    url= f'mysql+pymysql://{env.username}:{env.password}@{env.hostname}/zillow'
   
    return pd.read_sql(SQL_query, url)   
    
    
def get_zillow_data(filename = "zillow_data.csv"):
    '''
    this function will:
    -check local directory for csv file
        return if exists
    if csv doesn't exist
    if csv doesnt exist:
        - create a df of the SQL_query
        write df to csv
    output zillow df
    
    '''
    directory = os.getcwd()
    
    SQL_query = """select * from properties_2017
left join propertylandusetype using(propertylandusetypeid)
left join predictions_2017 using(parcelid)
left join airconditioningtype using( airconditioningtypeid)
left join architecturalstyletype using(architecturalstyletypeid)
left join buildingclasstype using(buildingclasstypeid)
left join heatingorsystemtype using(heatingorsystemtypeid)
left join storytype using(storytypeid)
left join typeconstructiontype using(typeconstructiontypeid)
where YEAR(transactiondate) = 2017;"""

    
    filename = "zillow_data.csv"
    
    url= f'mysql+pymysql://{env.username}:{env.password}@{env.hostname}/zillow'
    


    if os.path.exists(directory + filename):
        df = pd.read_csv(filename)
        return df
    else:
        df= new_zillow_data(SQL_query, url)
        df.to_csv(filename)
        return df










def get_zillow_csv():
    df = pd.read_csv('zillow_data.csv')
    df = df.sort_values(by = 'transactiondate')
    df = df.drop_duplicates(subset = 'parcelid', keep= 'last')


    df = df.drop(columns = ['typeconstructiontypeid','storytypeid','heatingorsystemtypeid',
                      'buildingclasstypeid', 'architecturalstyletypeid','airconditioningtypeid',
                      'airconditioningtypeid','Unnamed: 0','id', 'propertylandusetypeid','buildingqualitytypeid', 
                      'decktypeid','pooltypeid10','pooltypeid2', 'pooltypeid7','decktypeid','propertylandusetypeid','id.1',
                      'finishedsquarefeet12','finishedsquarefeet13','finishedsquarefeet15','finishedsquarefeet50','finishedsquarefeet6',
                      'finishedfloor1squarefeet','calculatedbathnbr','fullbathcnt','yardbuildingsqft17','yardbuildingsqft26','poolsizesum',
                      'fireplaceflag','taxdelinquencyyear','buildingclassdesc','typeconstructiondesc','structuretaxvaluedollarcnt',
                      'landtaxvaluedollarcnt','basementsqft','garagetotalsqft', 'garagecarcnt','hashottuborspa','regionidneighborhood','roomcnt',
                      'threequarterbathnbr','unitcnt','numberofstories','assessmentyear','architecturalstyledesc','heatingorsystemdesc','storydesc',
                      'taxamount'])

    
    df = df[(df.propertylandusedesc == 'Single Family Residential') | (df.propertylandusedesc == 'Mobile Home')]
    df.lotsizesquarefeet = df.lotsizesquarefeet.fillna(0.0)
    df.taxdelinquencyflag = df.taxdelinquencyflag.fillna('N')
    df.poolcnt = df.poolcnt.fillna(0)
    df.airconditioningdesc = df.airconditioningdesc.fillna('Yes')
    df = df.dropna(subset = ['yearbuilt','regionidzip'])

    df = df[df.censustractandblock.notnull()]
    df.yearbuilt = df.yearbuilt.astype(int)
    df.fips = df.fips.astype(int)
    df.regionidzip = df.regionidzip.astype(int)
    
    df.propertyzoningdesc = df.propertyzoningdesc.fillna('unknown')
    df.dropna(subset = ['calculatedfinishedsquarefeet'], inplace=True)
    df.fireplacecnt = df.fireplacecnt.fillna(0.0)

    df.calculatedfinishedsquarefeet = df.calculatedfinishedsquarefeet[df.calculatedfinishedsquarefeet < 25000]
    df.calculatedfinishedsquarefeet = df.calculatedfinishedsquarefeet[df.calculatedfinishedsquarefeet > 159]
    df = df[df.taxvaluedollarcnt < df.taxvaluedollarcnt.quantile(.95)].copy()
    
    return df


def nulls_by_col(df):
    """
    This function will:
        - take in a dataframe
        - assign a variable to a Series of total row nulls for ea/column
        - assign a variable to find the percent of rows w/nulls
        - output a df of the two variables.
    """
    num_missing = df.isnull().sum()
    pct_miss = (num_missing / df.shape[0]) * 100
    cols_missing = pd.DataFrame({
                    'num_rows_missing': num_missing,
                    'percent_rows_missing': pct_miss
                    })
    
    return  cols_missing


def handle_missing_values(df, prop_required_columns=0.5, prop_required_rows=0.75):
    """
    This function will:
    - take in: 
        - a dataframe
        - column threshold (defaulted to 0.5)
        - row threshold (defaulted to 0.75)
    - calculates the minimum number of non-missing values required for each column/row to be retained
    - drops columns/rows with a high proportion of missing values.
    - returns the new df
    """
    
    column_threshold = int(round(prop_required_columns * len(df.index), 0))
    df = df.dropna(axis=1, thresh=column_threshold)
    
    row_threshold = int(round(prop_required_rows * len(df.columns), 0))
    df = df.dropna(axis=0, thresh=row_threshold)
    
    return df



def get_upper_outliers(s, m=1.5):
    '''
    Given a series and a cutoff value, k, returns the upper outliers for the
    series.

    The values returned will be either 0 (if the point is not an outlier), or a
    number that indicates how far away from the upper bound the observation is.
    '''
    q1, q3 = s.quantile([.25, 0.75])
    iqr = q3 - q1
    upper_bound = q3 + (m * iqr)
    
    return s.apply(lambda x: max([x - upper_bound, 0]))


def add_upper_outlier_columns(df, m=1.5):
    '''
    Add a column with the suffix _outliers for all the numeric columns
    in the given dataframe.
    '''
    for col in df.select_dtypes('number'):
        df[col + '_outliers_upper'] = get_upper_outliers(df[col], m)
    return df


def outlier(df, feature, m=1.5):
    '''
    outlier will take in a dataframe's feature:
    - calculate it's 1st & 3rd quartiles,
    - use their difference to calculate the IQR
    - then apply to calculate upper and lower bounds
    - using the `m` multiplier
    '''
    q1 = df[feature].quantile(.25)
    q3 = df[feature].quantile(.75)
    
    iqr = q3 - q1
    
    upper_bound = q3 + (m * iqr)
    lower_bound = q1 - (m * iqr)
    
    return upper_bound, lower_bound


def remove_columns(df, cols_to_remove):
    """
    This function will:
    - take in a df and list of columns
    - drop the listed columns
    - return the new df
    """
    df = df.drop(columns=cols_to_remove)
    return df

