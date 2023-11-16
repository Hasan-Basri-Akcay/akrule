import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def  get_yearly_data(N=10, max_value=10, noise_std=3, anomaly_percentage=0):
    data_year = np.ones(N)
    trend = np.arange(0,N) / N
    data_year += trend
    scale = max_value/max(data_year)
    end_year = datetime.now().date()
    start_year = end_year - timedelta(days=365.64*N)
    df_year = pd.DataFrame()
    df_year["time"] = pd.date_range(start=start_year, end=end_year, freq="1Y")
    df_year["y"] = data_year*scale
    df_year["Country"] = "A"
    df_year["City"] = "C"
    
    anomaly_values = np.ones(df_year.shape[0])
    anomaly_num = round(df_year.shape[0]*anomaly_percentage/100)
    anomaly_values[:anomaly_num] = 2
    df_year["Anomaly"] = np.random.choice(anomaly_values, N)

    df_year_b = df_year.copy()
    df_year_b["Country"] = "B"
    df_year_b["City"] = "D"
    df_year_b["y"] = (data_year+trend*-2)*scale
    df_year_b["Anomaly"] = np.random.choice(anomaly_values, N)
    df_year = pd.concat([df_year, df_year_b]).reset_index(drop=True)
    
    noise = np.random.normal(0, noise_std, size=df_year.shape[0])
    df_year["y"] += noise
    df_year["y"] *= df_year["Anomaly"]    
    return df_year

def get_yearly_monthly_data(N=10, max_value=10, noise_std=3, anomaly_percentage=0):
    y = [1, 0.5, 0, -0.5, -1, -0.5, 0, 0.5, 1, 0.5, 0, 0.5]
    y = y * int(N/12)
    scale = max_value/max(y)
    end_year = datetime.now().date()
    start_year = end_year - timedelta(days=N*30.47)
    
    df_month = pd.DataFrame()
    df_month["time"] = pd.date_range(start=start_year, end=end_year, freq="1M")
    df_month["y"] = y + np.arange(0,N) / N
    df_month["Country"] = "A"
    df_month["y"] = df_month.y*scale
    
    anomaly_values = np.ones(df_month.shape[0])
    anomaly_num = round(df_month.shape[0]*anomaly_percentage/100)
    anomaly_values[:anomaly_num] = 2
    df_month["Anomaly"] = np.random.choice(anomaly_values, N)
    
    df_month_b = pd.DataFrame()
    df_month_b["time"] = pd.date_range(start=start_year, end=end_year, freq="1M")
    df_month_b["y"] = y + (np.arange(0,N) / N)*-2
    df_month_b["Country"] = "B"
    df_month_b["y"] = df_month_b.y*scale
    df_month_b["Anomaly"] = np.random.choice(anomaly_values, N)
    df_month = pd.concat([df_month, df_month_b]).reset_index(drop=True)
    
    noise = np.random.normal(0, noise_std, size=df_month.shape[0])
    df_month["y"] += noise
    df_month["y"] *= df_month["Anomaly"]
    return df_month

def get_yearly_monthly_daily_data(N=10, max_value=10, noise_std=3, anomaly_percentage=0):
    x = np.arange(0,28)
    y =  np.sin(4*np.pi*x/(28))
    y_map = {i+1:y[i] for i in range(28)}
    end_year = datetime.now().date()
    start_year = end_year - timedelta(days=N*30.47)
    
    df_daily = pd.DataFrame()
    df_daily["time"] = pd.date_range(start=start_year, end=end_year, freq="1D")
    df_daily["y"] = df_daily["time"].dt.day
    df_daily["y"] = df_daily["y"].map(y_map)
    df_daily.loc[df_daily["y"]>1, "y"] = 0
    df_daily["Country"] = "A"
    
    anomaly_values = np.ones(df_daily.shape[0])
    anomaly_num = round(df_daily.shape[0]*anomaly_percentage/100)
    anomaly_values[:anomaly_num] = 2
    trend = np.arange(0,df_daily.shape[0]) * 5 / df_daily.shape[0]
    
    df_daily_b = df_daily.copy()
    df_daily_b["Country"] = "B"
    df_daily["Anomaly"] = np.random.choice(anomaly_values, df_daily.shape[0])
    df_daily_b["Anomaly"] = np.random.choice(anomaly_values, df_daily.shape[0])
    df_daily["y"] = df_daily.y + trend
    df_daily_b["y"] = df_daily_b.y + trend*-2    
    df_daily = pd.concat([df_daily, df_daily_b]).reset_index(drop=True)
    
    noise = np.random.normal(0, noise_std, size=df_daily.shape[0])
    df_daily["y"] += noise
    df_daily["y"] *= df_daily["Anomaly"]
    
    return df_daily