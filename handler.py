import os
from flask import Flask, request, Response
import pandas as pd
from processing import Processing

app = Flask(__name__)
@app.route('/predict', methods=['POST'])


def general_function():
    r = request.get_json()
    if r:
        start = pd.to_datetime(r['datetime'])
        end = start + pd.Timedelta(6, 'd')
        df = pd.DataFrame()
        date_range = pd.date_range(start=start, end=end, freq='d').strftime('%Y-%m-%d')
        df['date'] = date_range
        df['date'] = pd.to_datetime(df['date'])
        pipeline = Processing()
        df = pipeline.feature_engineering(df)
        df1 = df.copy()
        df = pipeline.data_preparation(df)
        y_hat = pipeline.predict(df)
        df1['predict'] = y_hat
        df1['date'] = date_range 
        df_json = df1.to_json(orient='records')
        return df_json

    else:
        return '<h1> Api for Predict Sales Forecast. </h1>'

if __name__ == '__main__':
    port = os.environ.get('PORT', 5000)
    app.run('0.0.0.0', port=port)