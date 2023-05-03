import pandas as pd
import pickle

class Processing(object):
    def __init__(self):
        self.map_day_of_week = pickle.load(open(r'features/map_day_of_week.pkl', 'rb'))
        self.mm_target = pickle.load(open(r'features/mm_target.pkl', 'rb'))
        self.map_day_of_month = pickle.load(open(r'features/map_day_of_month.pkl', 'rb'))
        self.map_week_of_year = pickle.load(open(r'features/map_week_of_year.pkl', 'rb'))
        self.map_day_of_week_2 = pickle.load(open(r'features/map_day_of_week_2.pkl', 'rb'))
        self.map_year = pickle.load(open(r'features/map_year.pkl', 'rb'))
        self.map_month_of_year = pickle.load(open(r'features/map_month_of_year.pkl', 'rb'))
        self.model = pickle.load(open(r'model/model.pkl', 'rb'))
        return None


    def feature_engineering(self, df):
        df['day_of_month'] = df['date'].dt.day
        df['week_of_year'] = df['date'].dt.isocalendar().week
        df['day_of_week'] = df['date'].dt.day_name()
        df['day_of_week'] = df['day_of_week'].map(self.map_day_of_week)
        df['year'] = df['date'].dt.year
        df['month_of_year'] = df['date'].dt.month
        return df


    def data_preparation(self, df):
        df['day_of_month'] = df['day_of_month'].map(self.map_day_of_month)
        df['week_of_year'] = df['week_of_year'].map(self.map_week_of_year)
        df['day_of_week'] = df['day_of_week'].map(self.map_day_of_week_2)
        df['year'] = df['year'].map(self.map_year)
        df['month_of_year'] = df['month_of_year'].map(self.map_month_of_year)
        return df


    def predict(self, df):
        X = df.drop('date', axis=1).values
        y_hat = self.model.predict(X)
        y_hat = self.mm_target.inverse_transform(y_hat.reshape(-1, 1))
        return y_hat