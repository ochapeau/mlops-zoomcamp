#!/usr/bin/env python
# coding: utf-8

import sys
import argparse

import pickle
import pandas as pd
import numpy as np


categorical = ['PULocationID', 'DOLocationID']


def read_data(filename):
    df = pd.read_parquet(filename)
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    return df


def load_model(model_path):
    with open(model_path, 'rb') as f_in:
        dv, model = pickle.load(f_in)
    return dv, model


def get_paths(year, month):
    input_file = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet'
    output_file = f'outputs/output_{year:04d}-{month:02d}.parquet'
    return input_file, output_file
    

def save_results(df, y_pred, output_file):
    df_result = pd.DataFrame()
    df_result['ride_id'] = df['ride_id']
    df_result['predicted_duration'] = y_pred
    df_result.to_parquet(
        output_file,
        engine='pyarrow',
        compression=None,
        index=False
    )

    
def ride_duration_prediction(model_path, year, month):
    dv, model = load_model(model_path)
    
    input_file, output_file = get_paths(year, month)
    df = read_data(input_file)
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
    
    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)

    print(f'Mean predicted duration: {np.mean(y_pred)}')
    
    save_results(df, y_pred, output_file)
    

def run():
    # model_path = 'model.bin'
    # year = int(sys.argv[1]) # 2022
    # month = int(sys.argv[2]) # 2

    ap = argparse.ArgumentParser()
    ap.add_argument("-y", "--year", default=2022, help="year of the trip", type=int)
    ap.add_argument("-m", "--month", default=2, help="month of the trip", type=int)
    ap.add_argument("--model", default='models/model.bin', help="model to use")
    args = vars(ap.parse_args())

    print(f'The arguments are: {args}')

    ride_duration_prediction(
        model_path=args['model'],
        year=args['year'],
        month=args['month']
    )


if __name__ == '__main__':
    run()