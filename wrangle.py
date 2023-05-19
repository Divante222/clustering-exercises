import pandas as pd
import numpy as np
import env
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SelectKBest, RFE, f_regression, SequentialFeatureSelector
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.linear_model import LassoLars
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import TweedieRegressor


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
    
    
def preparing_data_zillow(df):
    '''
    droping unwanted rows for zillow first exercise
    converting float columns to integers
    '''
    df = df.drop(columns = ['Unnamed: 0'])
    df.calculatedfinishedsquarefeet = df.calculatedfinishedsquarefeet[df.calculatedfinishedsquarefeet < 25000]
    df.calculatedfinishedsquarefeet = df.calculatedfinishedsquarefeet[df.calculatedfinishedsquarefeet > 159]
    df = df[df.taxvaluedollarcnt < df.taxvaluedollarcnt.quantile(.95)].copy()
    df = df.dropna()
    df.calculatedfinishedsquarefeet = df.calculatedfinishedsquarefeet.astype(int).copy()
    df.fips = df.fips.astype(int).copy()
    df.yearbuilt = df.yearbuilt.astype(int).copy()

    return df


def wrangle_zillow():
    '''
    gets and prepares zillow data
    '''
    df = get_zillow_data()
    df = preparing_data_zillow(df)
    return df


def split_data(df, target):
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


def scaler_quantile_normal(X_train, X_validate, X_test):
    '''
    takes in data and uses a QuantileTransformer on it
    with the hyperperameter output_distribution == 'normal'
    '''
    scaler = QuantileTransformer(output_distribution='normal')
    return scaler.fit_transform(X_train), scaler.transform(X_validate), scaler.transform(X_test)


def scaler_quantile_default(X_train, X_validate, X_test):
    '''
    takes in data and uses a QuantileTransformer on it
    '''
    scaler = QuantileTransformer()
    return scaler.fit_transform(X_train), scaler.transform(X_validate), scaler.transform(X_test)


def scaler_min_max(X_train, X_validate, X_test):
    '''
    takes train, test, and validate data and uses the min max scaler on it
    '''
    scaler = MinMaxScaler()
    return scaler.fit_transform(X_train), scaler.transform(X_validate), scaler.transform(X_test)


def scaler_robust(X_train, X_validate, X_test):
    '''
    takes train, test, and validate data and uses the RobustScaler on it
    '''
    scaler = RobustScaler()
    return scaler.fit_transform(X_train), scaler.transform(X_validate), scaler.transform(X_test)


def standard_scaler(X_train, X_validate, X_test):
    '''
    takes train, test, and validate data and uses the standard_scaler on it
    '''
    scaler = StandardScaler()
    return scaler.fit_transform(X_train), scaler.transform(X_validate), scaler.transform(X_test)


def rfe(X_train, y_train, the_k):
    '''
    This function gets the top features that will best help predict the 
    target variable
    '''
    model = LinearRegression()
    rfe = RFE(model, n_features_to_select=the_k)
    rfe.fit(X_train, y_train)
    the_df = pd.DataFrame(
    {'rfe_ranking':rfe.ranking_},
    index=X_train.columns)
    return the_df[the_df['rfe_ranking'] == 1]


def select_kbest(X_train, y_train, the_k):
    '''
    This function gets the top features that will best help predict the 
    target variable
    '''
    kbest = SelectKBest(f_regression, k=the_k)
    kbest.fit(X_train, y_train)
    return X_train.columns[kbest.get_support()]


def metrics_reg(y, yhat):
    '''
    send in y_true, y_pred and returns rmse, r2
    '''
    rmse = mean_squared_error(y, yhat, squared=False)
    r2 = r2_score(y, yhat)
    return rmse, r2


def get_X_train_val_test(train,validate, test, x_target, y_target):
    '''
    geting the X's and y's and returns them
    '''
    X_train = train.drop(columns = x_target)
    X_validate = validate.drop(columns = x_target)
    X_test = test.drop(columns = x_target)
    y_train = train[y_target]
    y_validate = validate[y_target]
    y_test = test[y_target]

    return X_train, X_validate, X_test, y_train, y_validate, y_test


def get_model_numbers(X_train, X_validate, X_test, y_train, y_validate, y_test):
    '''
    This function takes the data and runs it through various models and returns the
    results in pandas dataframes for train, test and validate data
    '''
    baseline = y_train.mean()
    baseline_array = np.repeat(baseline, len(X_train))
    rmse, r2 = metrics_reg(y_train, baseline_array)

    metrics_train_df = pd.DataFrame(data=[
    {
        'model_train':'baseline',
        'rmse':rmse,
        'r2':r2
    }
    ])
    metrics_validate_df = pd.DataFrame(data=[
    {
        'model_validate':'baseline',
        'rmse':rmse,
        'r2':r2
    }
    ])
    metrics_test_df = pd.DataFrame(data=[
    {
        'model_validate':'baseline',
        'rmse':rmse,
        'r2':r2
    }
    ])


    Linear_regression1 = LinearRegression()
    Linear_regression1.fit(X_train,y_train)
    predict_linear = Linear_regression1.predict(X_train)
    rmse, r2 = metrics_reg(y_train, predict_linear)
    metrics_train_df.loc[1] = ['ordinary least squared(OLS)', rmse, r2]

    predict_linear = Linear_regression1.predict(X_validate)
    rmse, r2 = metrics_reg(y_validate, predict_linear)
    metrics_validate_df.loc[1] = ['ordinary least squared(OLS)', rmse, r2]


    lars = LassoLars()
    lars.fit(X_train, y_train)
    pred_lars = lars.predict(X_train)
    rmse, r2 = metrics_reg(y_train, pred_lars)
    metrics_train_df.loc[2] = ['lasso lars(lars)', rmse, r2]

    pred_lars = lars.predict(X_validate)
    rmse, r2 = metrics_reg(y_validate, pred_lars)
    metrics_validate_df.loc[2] = ['lasso lars(lars)', rmse, r2]


    pf = PolynomialFeatures(degree=2)
    X_train_degree2 = pf.fit_transform(X_train)
   

    pr = LinearRegression()
    pr.fit(X_train_degree2, y_train)
    pred_pr = pr.predict(X_train_degree2)
    rmse, r2 = metrics_reg(y_train, pred_pr)
    metrics_train_df.loc[3] = ['Polynomial Regression(poly2)', rmse, r2]

    X_validate_degree2 = pf.transform(X_validate)
    pred_pr = pr.predict(X_validate_degree2)
    rmse, r2 = metrics_reg(y_validate, pred_pr)
    metrics_validate_df.loc[3] = ['Polynomial Regression(poly2)', rmse, r2]


    glm = TweedieRegressor(power=2, alpha=0)
    glm.fit(X_train, y_train)
    pred_glm = glm.predict(X_train)
    rmse, r2 = metrics_reg(y_train, pred_glm)
    metrics_train_df.loc[4] = ['Generalized Linear Model (GLM)', rmse, r2]

    pred_glm = glm.predict(X_validate)
    rmse, r2 = metrics_reg(y_validate, pred_glm)
    metrics_validate_df.loc[4] = ['Generalized Linear Model (GLM)', rmse, r2]


    X_test_degree2 = pf.transform(X_test)
    pred_pr = pr.predict(X_test_degree2)
    rmse, r2 = metrics_reg(y_test, pred_pr)
    metrics_test_df.loc[1] = ['Polynomial Regression(poly2)', round(rmse,2), r2]


    metrics_train_df.rmse = metrics_train_df.rmse.astype(int)
    metrics_validate_df.rmse = metrics_validate_df.rmse.astype(int)
    metrics_test_df.rmse = metrics_test_df.rmse.astype(int)
    print()
    metrics_train_df.r2 = (metrics_train_df.r2 * 100).astype(int)
    metrics_validate_df.r2 = (metrics_validate_df.r2 * 100).astype(int)
    metrics_test_df.r2 = (metrics_test_df.r2 * 100).astype(int)

    return metrics_train_df, metrics_validate_df, metrics_test_df


def scaled_data_to_dataframe(X_train, X_validate, X_test):
    '''
    This function scales the data and returns it as a pandas dataframe
    '''
    X_train_columns = X_train.columns
    X_validate_columns = X_validate.columns
    X_test_columns = X_test.columns
    X_train_numbers, X_validade_numbers, X_test_numbers = scaler_robust(X_train, X_validate, X_test)

    X_train_scaled = pd.DataFrame(columns = X_train_columns)
    for i in range(int(X_train_numbers.shape[0])):
        X_train_scaled.loc[len(X_train_scaled.index)] = X_train_numbers[i]
    
    X_validate_scaled = pd.DataFrame(columns = X_validate_columns)
    for i in range(int(X_validade_numbers.shape[0])):
        X_validate_scaled.loc[len(X_validate_scaled.index)] = X_validade_numbers[i]
    
    X_test_scaled = pd.DataFrame(columns = X_test_columns)
    for i in range(int(X_test_numbers.shape[0])):
        X_test_scaled.loc[len(X_test_scaled.index)] = X_test_numbers[i]

    return X_train_scaled, X_validate_scaled, X_test_scaled


def over_time(train):
    '''
    this function cuts the years built into bins and updates the bins for better 
    readability
    '''
    train['year_bins'] = pd.qcut(train.yearbuilt, q = 4, precision=0).astype('str')
    BIN_1949_1958 = train[train['year_bins'] == '(1949.0, 1958.0]']
    BIN_1861_1949 = train[train['year_bins'] == '(1861.0, 1949.0]']
    BIN_1958_1974 = train[train['year_bins'] == '(1958.0, 1974.0]']
    BIN_1974_2016 = train[train['year_bins'] == '(1974.0, 2016.0]']

    train['year_bins'][train['year_bins'] == '(1861.0, 1949.0]'] ='1861 - 1949'
    train['year_bins'][train['year_bins'] == '(1949.0, 1958.0]'] ='1949 - 1958'
    train['year_bins'][train['year_bins'] == '(1958.0, 1974.0]'] ='1958 - 1974'
    train['year_bins'][train['year_bins'] == '(1974.0, 2016.0]'] ='1974 - 2016'

    return BIN_1949_1958, BIN_1861_1949, BIN_1958_1974, BIN_1974_2016, train


def bedrooms_bins(train):
    '''
    this function cuts the bedroomcnt into bins and updates the bins for better 
    readability
    '''

    train['bedroom_bins'] = pd.cut(train.bedroomcnt, bins = 4).astype('str')

    BIN_0_2_5 = train[train['bedroom_bins'] == '(-0.009, 2.25]']
    BIN_45675 = train[train['bedroom_bins'] == '(4.5, 6.75]']
    BIN_22545 = train[train['bedroom_bins'] == '(2.25, 4.5]']
    BIN_679 = train[train['bedroom_bins'] == '(6.75, 9.0]']

    train['bedroom_bins'][train['bedroom_bins'] == '(-0.009, 2.25]'] = '0 - 2.25'
    train['bedroom_bins'][train['bedroom_bins'] == '(4.5, 6.75]'] = '4.5 - 6.75'
    train['bedroom_bins'][train['bedroom_bins'] == '(2.25, 4.5]'] = '2.25 - 4.5'
    train['bedroom_bins'][train['bedroom_bins'] == '(6.75, 9.0]'] = '6.75 - 9.0'

    return BIN_0_2_5, BIN_45675, BIN_22545, BIN_679, train

def square_feet_bins(train):
    '''
    this function cuts the square feet into bins and updates the bins for better 
    readability
    '''

    train['size_bins'] = pd.qcut(train.calculatedfinishedsquarefeet, q = 4).astype('str')

    size_19999_1242 = train[train['size_bins'] == '(159.999, 1244.0]']
    size_1242_1592 = train[train['size_bins'] == '(1244.0, 1592.0]']
    size_1592_2117 = train[train['size_bins'] == '(1592.0, 2112.5]']
    size_2117_9200 = train[train['size_bins'] == '(2112.5, 7648.0]']

    train['size_bins'][train['size_bins'] == '(159.999, 1244.0]'] = '160 - 1244'
    train['size_bins'][train['size_bins'] == '(1244.0, 1592.0]'] = '1244 - 1592'
    train['size_bins'][train['size_bins'] == '(1592.0, 2112.5]'] = '1592 - 2112'
    train['size_bins'][train['size_bins'] == '(2112.5, 7648.0]'] = '2112 - 7648'

    return size_19999_1242, size_1242_1592, size_1592_2117, size_2117_9200, train

def bathroom_bins(train):
    '''
    this function cuts the bathroom count into bins and updates the bins for better 
    readability
    '''

    train['bathroom_bins'] = pd.cut(train.bathroomcnt, bins= 4).astype('str')
    
    bath_0_2 = train[train['bathroom_bins'] == '(-0.008, 2.0]']
    bath_2_4 = train[train['bathroom_bins'] == '(2.0, 4.0]']
    bath_4_6 = train[train['bathroom_bins'] == '(4.0, 6.0]']
    bath_6_8 = train[train['bathroom_bins'] == '(6.0, 8.0]']

    train['bathroom_bins'][train['bathroom_bins'] == '(-0.008, 2.0]'] = '0 - 2'
    train['bathroom_bins'][train['bathroom_bins'] == '(2.0, 4.0]'] = '2 - 4'
    train['bathroom_bins'][train['bathroom_bins'] == '(4.0, 6.0]'] = '4 - 6'
    train['bathroom_bins'][train['bathroom_bins'] == '(6.0, 8.0]'] = '6 - 8'

    return bath_0_2, bath_2_4, bath_4_6, bath_6_8, train

