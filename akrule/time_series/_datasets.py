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
    df_year["time"] = pd.date_range(start=start_year, end=end_year, freq="1Y")[:data_year.shape[0]]
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
    df_month["time"] = pd.date_range(start=start_year, end=end_year, freq="1M")[:len(y)]
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
    y =  np.sin(4*np.pi*x/(28))*max_value
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

def get_monthly_data(N=10, max_value=10, noise_std=3, anomaly_percentage=0):
    data_month = np.ones(N)
    trend = np.arange(0,N) / N
    data_month += trend
    scale = max_value/max(data_month)
    end_year = datetime.now().date()
    start_year = end_year - timedelta(days=30.47*N)
    df_month = pd.DataFrame()
    df_month["time"] = pd.date_range(start=start_year, end=end_year, freq="1M")[:data_month.shape[0]]
    df_month["y"] = data_month*scale
    df_month["Country"] = "A"
    df_month["City"] = "C"
    
    anomaly_values = np.ones(df_month.shape[0])
    anomaly_num = round(df_month.shape[0]*anomaly_percentage/100)
    anomaly_values[:anomaly_num] = 2
    df_month["Anomaly"] = np.random.choice(anomaly_values, N)

    df_month_b = df_month.copy()
    df_month_b["Country"] = "B"
    df_month_b["City"] = "D"
    df_month_b["y"] = (data_month+trend*-2)*scale
    df_month_b["Anomaly"] = np.random.choice(anomaly_values, N)
    df_month = pd.concat([df_month, df_month_b]).reset_index(drop=True)
    
    noise = np.random.normal(0, noise_std, size=df_month.shape[0])
    df_month["y"] += noise
    df_month["y"] *= df_month["Anomaly"]    
    return df_month

def get_monthly_weekly_data(N=10, max_value=10, noise_std=3, anomaly_percentage=0):
    y = [1, 0.5, 0, 0.5]
    y = y * int(N/4)
    scale = max_value/max(y)
    end_year = datetime.now().date()
    start_year = end_year - timedelta(days=N*7)

    df_weekly = pd.DataFrame()
    df_weekly["time"] = pd.date_range(start=start_year, end=end_year, freq="1W")[:len(y)]
    df_weekly["y"] = y + np.arange(0,N) / N
    df_weekly["Country"] = "A"
    df_weekly["y"] = df_weekly.y*scale
    
    anomaly_values = np.ones(df_weekly.shape[0])
    anomaly_num = round(df_weekly.shape[0]*anomaly_percentage/100)
    anomaly_values[:anomaly_num] = 2
    trend = np.arange(0,df_weekly.shape[0]) * 5 / df_weekly.shape[0]
    
    df_weekly_b = df_weekly.copy()
    df_weekly_b["Country"] = "B"
    df_weekly["Anomaly"] = np.random.choice(anomaly_values, df_weekly.shape[0])
    df_weekly_b["Anomaly"] = np.random.choice(anomaly_values, df_weekly.shape[0])
    df_weekly["y"] = df_weekly.y + trend
    df_weekly_b["y"] = df_weekly_b.y + trend*-2    
    df_weekly = pd.concat([df_weekly, df_weekly_b]).reset_index(drop=True)
    
    noise = np.random.normal(0, noise_std, size=df_weekly.shape[0])
    df_weekly["y"] += noise
    df_weekly["y"] *= df_weekly["Anomaly"]
    return df_weekly

def get_monthly_weekly_daily_data(N=10, max_value=10, noise_std=3, anomaly_percentage=0):
    y =  [1, 0.5, 0, -0.5, -1, -0.5, 0]
    y = y * int(N/7)
    scale = max_value/max(y)
    end_year = datetime.now().date()
    start_year = end_year - timedelta(days=N)

    df_daily = pd.DataFrame()
    df_daily["time"] = pd.date_range(start=start_year, end=end_year, freq="1D")[:len(y)]
    df_daily["y"] = y
    df_daily["Country"] = "A"
    df_daily["y"] = df_daily.y*scale
    
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

def get_weekly_data(N=10, max_value=10, noise_std=3, anomaly_percentage=0):
    data_week = np.ones(N)
    trend = np.arange(0,N) / N
    data_week += trend
    scale = max_value/max(data_week)
    end_year = datetime.now().date()
    start_year = end_year - timedelta(days=7*N)
    df_week = pd.DataFrame()
    df_week["time"] = pd.date_range(start=start_year, end=end_year, freq="1W")[:data_week.shape[0]]
    df_week["y"] = data_week*scale
    df_week["Country"] = "A"
    
    anomaly_values = np.ones(df_week.shape[0])
    anomaly_num = round(df_week.shape[0]*anomaly_percentage/100)
    anomaly_values[:anomaly_num] = 2
    trend = np.arange(0,df_week.shape[0]) * 5 / df_week.shape[0]
    
    df_week_b = df_week.copy()
    df_week_b["Country"] = "B"
    df_week["Anomaly"] = np.random.choice(anomaly_values, df_week.shape[0])
    df_week_b["Anomaly"] = np.random.choice(anomaly_values, df_week.shape[0])
    df_week["y"] = df_week.y + trend
    df_week_b["y"] = df_week_b.y + trend*-2    
    df_week = pd.concat([df_week, df_week_b]).reset_index(drop=True)
    
    noise = np.random.normal(0, noise_std, size=df_week.shape[0])
    df_week["y"] += noise
    df_week["y"] *= df_week["Anomaly"]
    return df_week

def get_weekly_daily_data(N=10, max_value=10, noise_std=3, anomaly_percentage=0):
    y = [1, 0.5, 0, -0.5, -1, -0.5, 0]
    y = y * int(N)
    scale = max_value/max(y)
    end_year = datetime.now().date()
    start_year = end_year - timedelta(days=N*7)

    df_daily = pd.DataFrame()
    df_daily["time"] = pd.date_range(start=start_year, end=end_year, freq="1D")[:len(y)]
    df_daily["y"] = y
    df_daily["Country"] = "A"
    df_daily["y"] = df_daily.y*scale
    
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

def get_weekly_daily_hourly_data(N=10, max_value=10, noise_std=3, anomaly_percentage=0):
    x = np.arange(0,24)
    y =  np.sin(4*np.pi*x/(24))*max_value
    y_map = {i:y[i] for i in range(24)}
    end_year = datetime.now().date()
    start_year = end_year - timedelta(days=N)

    df_hourly = pd.DataFrame()
    df_hourly["time"] = pd.date_range(start=start_year, end=end_year, freq="1h")
    df_hourly["y"] = df_hourly["time"].dt.hour
    df_hourly["y"] = df_hourly["y"].map(y_map)
    df_hourly["Country"] = "A"
    
    anomaly_values = np.ones(df_hourly.shape[0])
    anomaly_num = round(df_hourly.shape[0]*anomaly_percentage/100)
    anomaly_values[:anomaly_num] = 2
    trend = np.arange(0,df_hourly.shape[0]) * 5 / df_hourly.shape[0]
    
    df_hourly_b = df_hourly.copy()
    df_hourly_b["Country"] = "B"
    df_hourly["Anomaly"] = np.random.choice(anomaly_values, df_hourly.shape[0])
    df_hourly_b["Anomaly"] = np.random.choice(anomaly_values, df_hourly.shape[0])
    df_hourly["y"] = df_hourly.y + trend
    df_hourly_b["y"] = df_hourly_b.y + trend*-2    
    df_hourly = pd.concat([df_hourly, df_hourly_b]).reset_index(drop=True)
    
    noise = np.random.normal(0, noise_std, size=df_hourly.shape[0])
    df_hourly["y"] += noise
    df_hourly["y"] *= df_hourly["Anomaly"]
    return df_hourly

def get_daily_data(N=10, max_value=10, noise_std=3, anomaly_percentage=0):
    data_daily = np.ones(N)
    trend = np.arange(0,N) / N
    data_daily += trend
    scale = max_value/max(data_daily)
    end_year = datetime.now().date()
    start_year = end_year - timedelta(days=N)
    df_daily = pd.DataFrame()
    df_daily["time"] = pd.date_range(start=start_year, end=end_year, freq="1D")[:data_daily.shape[0]]
    df_daily["y"] = data_daily*scale
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

def get_daily_hourly_data(N=10, max_value=10, noise_std=3, anomaly_percentage=0):
    x = np.arange(0,24)
    y =  np.sin(4*np.pi*x/(24))*max_value
    y_map = {i:y[i] for i in range(24)}
    end_year = datetime.now().date()
    start_year = end_year - timedelta(days=N)

    df_hourly = pd.DataFrame()
    df_hourly["time"] = pd.date_range(start=start_year, end=end_year, freq="1h")
    df_hourly["y"] = df_hourly["time"].dt.hour
    df_hourly["y"] = df_hourly["y"].map(y_map)
    df_hourly["Country"] = "A"
    
    anomaly_values = np.ones(df_hourly.shape[0])
    anomaly_num = round(df_hourly.shape[0]*anomaly_percentage/100)
    anomaly_values[:anomaly_num] = 2
    trend = np.arange(0,df_hourly.shape[0]) * 5 / df_hourly.shape[0]
    
    df_hourly_b = df_hourly.copy()
    df_hourly_b["Country"] = "B"
    df_hourly["Anomaly"] = np.random.choice(anomaly_values, df_hourly.shape[0])
    df_hourly_b["Anomaly"] = np.random.choice(anomaly_values, df_hourly.shape[0])
    df_hourly["y"] = df_hourly.y + trend
    df_hourly_b["y"] = df_hourly_b.y + trend*-2    
    df_hourly = pd.concat([df_hourly, df_hourly_b]).reset_index(drop=True)
    
    noise = np.random.normal(0, noise_std, size=df_hourly.shape[0])
    df_hourly["y"] += noise
    df_hourly["y"] *= df_hourly["Anomaly"]
    return df_hourly

def get_daily_hourly_minutely_data(N=10, max_value=10, noise_std=3, anomaly_percentage=0):
    x = np.arange(0,60)
    y =  np.sin(1*np.pi*x/(60))*max_value
    y_map = {i:y[i] for i in range(60)}
    end_year = datetime.now().date()
    start_year = end_year - timedelta(days=N)

    df_minutely = pd.DataFrame()
    df_minutely["time"] = pd.date_range(start=start_year, end=end_year, freq="1T")
    df_minutely["y"] = df_minutely["time"].dt.minute
    df_minutely["y"] = df_minutely["y"].map(y_map)
    df_minutely["Country"] = "A"
    
    anomaly_values = np.ones(df_minutely.shape[0])
    anomaly_num = round(df_minutely.shape[0]*anomaly_percentage/100)
    anomaly_values[:anomaly_num] = 2
    trend = np.arange(0,df_minutely.shape[0]) * 5 / df_minutely.shape[0]
    
    df_minutely_b = df_minutely.copy()
    df_minutely_b["Country"] = "B"
    df_minutely["Anomaly"] = np.random.choice(anomaly_values, df_minutely.shape[0])
    df_minutely_b["Anomaly"] = np.random.choice(anomaly_values, df_minutely.shape[0])
    df_minutely["y"] = df_minutely.y + trend
    df_minutely_b["y"] = df_minutely_b.y + trend*-2    
    df_minutely = pd.concat([df_minutely, df_minutely_b]).reset_index(drop=True)
    
    noise = np.random.normal(0, noise_std, size=df_minutely.shape[0])
    df_minutely["y"] += noise
    df_minutely["y"] *= df_minutely["Anomaly"]
    return df_minutely