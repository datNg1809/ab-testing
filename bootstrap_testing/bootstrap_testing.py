from email.policy import default
from turtle import color
import numpy as np
import pandas as pd
import math
from scipy import stats
import os
import matplotlib.pyplot as plt
from snowflake.connector import connect


def set_sf_con():
    con = connect(user= os.environ['SNOWFLAKE_USER'],
                  password= os.environ['SNOWFLAKE_PASSWORD'],
                  account= os.environ['SNOWFLAKE_ACCOUNT'],
                  schema= os.environ['SNOWFLAKE_SCHEMA'],
                  database= os.environ['SNOWFLAKE_DB'])
    return con

def import_sf_sql(query, chunks=False, chunksize=10000):
    con = set_sf_con()
    if chunks:
        df = pd.DataFrame()
        for chunk in pd.read_sql(query, con, chunksize=chunksize):
            df = pd.concat([df, chunk])
            df.columns = map(str.lower, df.columns)

    else:
        df = pd.read_sql(query, con)
        df.columns = map(str.lower, df.columns)
    return df

def load_data(filepath):
    df = pd.read_csv(filepath, sep=',', index_col=False, compression=None)
    return df

def get_data(query=None):
    df = import_sf_sql(query)
    df.to_csv(os.path.join(os.environ['HOME'], '/Downloads/bootstrap_demo.csv'), sep=',')
    return df

def get_sample(df:pd.DataFrame, size:int):
    '''
    bootstrap with replacement
    '''
    return df.sample(size, replace=True)

def metric_calc(df:pd.DataFrame, numerator:str, denominator:str):
    return df[numerator].sum()/df[denominator].count()

def bootstrap(df:pd.DataFrame,
              size:int=20000,
              iter:int=5000
            ):
    output = np.array([])
    for i in range(0,iter):
        sample = get_sample(df,size=size)
        metric = metric_calc(sample,'leads', 'session_id')
        output = np.append(output, metric)
    return output

def get_ci(n:np.array):
    x_bar = np.mean(n)
    s = np.std(n)
    n = len(n)
    z = 1.96

    lower = x_bar - (z * (s/math.sqrt(n)))
    upper = x_bar + (z * (s/math.sqrt(n)))
    return lower, upper

def plot_histogram(control:np.array,
                   variant:np.array,
                   bins=25,
                   density:bool=False
                 ):
    mean_c, mean_v = np.mean(control), np.mean(variant)
    lower_c, upper_c = get_ci(control)
    lower_v, upper_v = get_ci(variant)
    uplift = (mean_v-mean_c)/mean_c
    print('control lower, mean, upper bound: ',lower_c, mean_c, upper_c)
    print('variant lower, mean, upper bound: ',lower_v, mean_v, upper_v) 

    plt.hist(control, bins=bins, density=density, alpha=0.8, label='control')
    plt.hist(variant, bins=bins, density=density, alpha=0.8, label='variant')
    plt.axvline(mean_c, ls="--", color='r')
    plt.axvline(mean_v, ls="--", color='r')
    #plt.axvline(lower, ls="-.", label=lower)
    #plt.axvline(upper, ls="-.", label=upper)
    xmin, xmax, ymin, ymax = plt.axis()
    '''plt.annotate('', xy=(min(mean_c,mean_v), ymin+(ymax-ymin)/3),  xycoords = 'axes fraction',
                 xytext=(max(mean_c,mean_v),ymin+(ymax-ymin)/3), textcoords = 'axes fraction',
                 arrowprops=dict(edgecolor='black', arrowstyle = '<->')
                )'''
    plt.text(x=xmin + (xmax-xmin)/4,
            y=ymin + (ymax-ymin)/3,
            s=f'uplift: {uplift}',
            horizontalalignment='center',
            verticalalignment='center', 
            color = '0.25')
    plt.title("histogram")
    plt.legend(loc='upper right')
    plt.xlabel("metric")
    plt.show()