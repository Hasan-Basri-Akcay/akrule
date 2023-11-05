import numpy as np
import pandas as pd

def  get_yearly_data(N=10, max_value=10, noise_std=3, anomaly_percentage=0):
    data_year = np.ones(N)
    trend = np.arange(0,N) / N
    data_year += trend
    scale = max_value/max(data_year)
    df_year = pd.DataFrame()
    df_year["time"] = pd.date_range(start='1/1/2010', end='1/1/2020', freq="1Y")
    df_year["y"] = data_year*scale
    df_year["Country"] = "A"
    
    anomaly_values = np.ones(df_year.shape[0])
    anomaly_num = round(df_year.shape[0]*anomaly_percentage/100)
    anomaly_values[:anomaly_num] = 2
    df_year["Anomaly"] = np.random.choice(anomaly_values, N)

    df_year_b = df_year.copy()
    df_year_b["Country"] = "B"
    df_year_b["y"] = (data_year+trend*-2)*scale
    df_year_b["Anomaly"] = np.random.choice(anomaly_values, N)
    df_year = pd.concat([df_year, df_year_b]).reset_index(drop=True)
    
    noise = np.random.normal(0, noise_std, size=df_year.shape[0])
    df_year["y"] += noise
    df_year["y"] *= df_year["Anomaly"]
    
    return df_year