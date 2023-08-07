import pandas as pd
import os
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

    


