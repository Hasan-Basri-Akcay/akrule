from datetime import datetime, timedelta

import sys
#sys.path.append("./")

from akrule.time_series import get_daily_hourly_minutely_data
from akrule.time_series import get_daily_hourly_data
from akrule.time_series import get_daily_data
from akrule.time_series import get_weekly_daily_hourly_data
from akrule.time_series import get_weekly_daily_data
from akrule.time_series import get_weekly_data
from akrule.time_series import get_monthly_weekly_daily_data
from akrule.time_series import get_monthly_weekly_data
from akrule.time_series import get_monthly_data
from akrule.time_series import get_yearly_monthly_daily_data
from akrule.time_series import get_yearly_monthly_data
from akrule.time_series import get_yearly_data

from akrule.time_series import AKRuleBasedTS


def test_akrulebasedts():
    ##########################################################################
    # Daily_Hourly_Minutely
    data = get_daily_hourly_minutely_data(N=5, max_value=10, noise_std=3, anomaly_percentage=0)    
    split_time = str(datetime.now().date() - timedelta(days=1))
    X_train = data[data["time"]<split_time].copy()
    X_test = data[data["time"]>=split_time].copy()

    tag_features = ["Country"]
    model = AKRuleBasedTS(freqs=["dayofweek", "hourly", "minutely"], tag_features=tag_features, average_num=3, trend_level=1, fillna=True, metric_ci=0.95, inplace=False)
    _ = model.fit_predict(X_train)
    X_pred = model.predict(X_test.drop(["y"], axis=1))
    
    error_msg = f"Daily_Hourly_Minutely incorret shape X_pred:{X_pred.shape[0]}, X_test:{X_test.shape[0]}!"
    assert X_pred.shape[0]==X_test.shape[0], error_msg
    
    y_pred_nan_sum = X_pred["y_pred"].isna().sum()
    error_msg = f"Daily_Hourly_Minutely y_pred has {y_pred_nan_sum} None!"
    assert y_pred_nan_sum==0,error_msg
    ##########################################################################
    
    ##########################################################################
    # Daily_Hourly
    df_hourly = get_daily_hourly_data(N=30, max_value=10, noise_std=3, anomaly_percentage=0)
    split_time = str(datetime.now().date() - timedelta(days=7))
    X_train = df_hourly[df_hourly["time"]<split_time].copy()
    X_test = df_hourly[df_hourly["time"]>=split_time].copy()

    tag_features = ["Country"]
    model = AKRuleBasedTS(freqs=["dayofweek", "hourly"], tag_features=tag_features, average_num=3, trend_level=1, fillna=True, metric_ci=0.95, inplace=False)
    _ = model.fit_predict(X_train)
    X_pred = model.predict(X_test.drop(["y"], axis=1))
    
    error_msg = f"Daily_Hourly incorret shape X_pred:{X_pred.shape[0]}, X_test:{X_test.shape[0]}!"
    assert X_pred.shape[0]==X_test.shape[0], error_msg
    
    y_pred_nan_sum = X_pred["y_pred"].isna().sum()
    error_msg = f"Daily_Hourly y_pred has {y_pred_nan_sum} None!"
    assert y_pred_nan_sum==0,error_msg
    ##########################################################################
    
    ##########################################################################
    # Daily
    df_daily = get_daily_data(N=30, max_value=10, noise_std=3, anomaly_percentage=0)
    split_time = str(datetime.now().date() - timedelta(days=7))
    X_train = df_daily[df_daily["time"]<split_time].copy()
    X_test = df_daily[df_daily["time"]>=split_time].copy()

    tag_features = ["Country"]
    model = AKRuleBasedTS(freqs=["dayofweek"], tag_features=tag_features, average_num=3, trend_level=1, fillna=True, metric_ci=0.95, inplace=False)
    _ = model.fit_predict(X_train)
    X_pred = model.predict(X_test.drop(["y"], axis=1))
    
    error_msg = f"Daily_Hourly incorret shape X_pred:{X_pred.shape[0]}, X_test:{X_test.shape[0]}!"
    assert X_pred.shape[0]==X_test.shape[0], error_msg
    
    y_pred_nan_sum = X_pred["y_pred"].isna().sum()
    error_msg = f"Daily_Hourly y_pred has {y_pred_nan_sum} None!"
    assert y_pred_nan_sum==0,error_msg
    ##########################################################################
    
    ##########################################################################
    # Weekly_Hourly_Daily
    df_hourly = get_weekly_daily_hourly_data(N=30, max_value=10, noise_std=3, anomaly_percentage=0)
    split_time = str(datetime.now().date() - timedelta(days=7))
    X_train = df_hourly[df_hourly["time"]<split_time].copy()
    X_test = df_hourly[df_hourly["time"]>=split_time].copy()

    tag_features = ["Country"]
    model = AKRuleBasedTS(freqs=["weekly", "dayofweek", "hourly"], tag_features=tag_features, average_num=3, trend_level=1, fillna=True, metric_ci=0.95, inplace=False)
    X_val = model.fit_predict(X_train)
    X_pred = model.predict(X_test.drop(["y"], axis=1))
    
    error_msg = f"Daily_Hourly incorret shape X_pred:{X_pred.shape[0]}, X_test:{X_test.shape[0]}!"
    assert X_pred.shape[0]==X_test.shape[0], error_msg
    
    y_pred_nan_sum = X_pred["y_pred"].isna().sum()
    error_msg = f"Daily_Hourly y_pred has {y_pred_nan_sum} None!"
    assert y_pred_nan_sum==0,error_msg
    ##########################################################################
    
    ##########################################################################
    # Weekly_Hourly
    df_daily = get_weekly_daily_data(N=10, max_value=10, noise_std=3, anomaly_percentage=0)
    split_time = str(datetime.now().date() - timedelta(days=7*3))
    X_train = df_daily[df_daily["time"]<split_time].copy()
    X_test = df_daily[df_daily["time"]>=split_time].copy()

    tag_features = ["Country"]
    model = AKRuleBasedTS(freqs=["weekly", "dayofweek"], tag_features=tag_features, average_num=3, trend_level=1, fillna=True, metric_ci=0.95, inplace=False)
    _ = model.fit_predict(X_train)
    X_pred = model.predict(X_test.drop(["y"], axis=1))
    
    error_msg = f"Daily_Hourly incorret shape X_pred:{X_pred.shape[0]}, X_test:{X_test.shape[0]}!"
    assert X_pred.shape[0]==X_test.shape[0], error_msg
    
    y_pred_nan_sum = X_pred["y_pred"].isna().sum()
    error_msg = f"Daily_Hourly y_pred has {y_pred_nan_sum} None!"
    assert y_pred_nan_sum==0,error_msg
    ##########################################################################
    
    ##########################################################################
    # Weekly
    df_weekly = get_weekly_data(N=20, max_value=10, noise_std=3, anomaly_percentage=0)
    split_time = str(datetime.now().date() - timedelta(days=7*3))
    X_train = df_weekly[df_weekly["time"]<split_time].copy()
    X_test = df_weekly[df_weekly["time"]>=split_time].copy()

    tag_features = ["Country"]
    model = AKRuleBasedTS(freqs=["weekly"], tag_features=tag_features, average_num=3, trend_level=1, fillna=True, metric_ci=0.95, inplace=False)
    _ = model.fit_predict(X_train)
    X_pred = model.predict(X_test.drop(["y"], axis=1))
    
    error_msg = f"Daily_Hourly incorret shape X_pred:{X_pred.shape[0]}, X_test:{X_test.shape[0]}!"
    assert X_pred.shape[0]==X_test.shape[0], error_msg
    
    y_pred_nan_sum = X_pred["y_pred"].isna().sum()
    error_msg = f"Daily_Hourly y_pred has {y_pred_nan_sum} None!"
    assert y_pred_nan_sum==0,error_msg
    ##########################################################################
    
    ##########################################################################
    # Monthly-Weekly-Daily
    df_daily = get_monthly_weekly_daily_data(N=150, max_value=10, noise_std=3, anomaly_percentage=0)
    split_time = str(datetime.now().date() - timedelta(days=7*2))
    X_train = df_daily[df_daily["time"]<split_time].copy()
    X_test = df_daily[df_daily["time"]>=split_time].copy()

    tag_features = ["Country"]
    model = AKRuleBasedTS(freqs=["monthly", "weekly", "dayofweek"], tag_features=tag_features, average_num=3, trend_level=1, fillna=True, metric_ci=0.95, inplace=False)
    _ = model.fit_predict(X_train)
    X_pred = model.predict(X_test.drop(["y"], axis=1))
    
    error_msg = f"Daily_Hourly incorret shape X_pred:{X_pred.shape[0]}, X_test:{X_test.shape[0]}!"
    assert X_pred.shape[0]==X_test.shape[0], error_msg
    
    y_pred_nan_sum = X_pred["y_pred"].isna().sum()
    error_msg = f"Daily_Hourly y_pred has {y_pred_nan_sum} None!"
    assert y_pred_nan_sum==0,error_msg
    ##########################################################################
    
    ##########################################################################
    # Monthly-Weekly
    df_weekly = get_monthly_weekly_data(N=56, max_value=10, noise_std=3, anomaly_percentage=0)
    split_time = str(datetime.now().date() - timedelta(days=7*2))
    X_train = df_weekly[df_weekly["time"]<split_time].copy()
    X_test = df_weekly[df_weekly["time"]>=split_time].copy()

    tag_features = ["Country"]
    model = AKRuleBasedTS(freqs=["monthly", "weekly"], tag_features=tag_features, average_num=3, trend_level=1, fillna=True, metric_ci=0.95, inplace=False)
    _ = model.fit_predict(X_train)
    X_pred = model.predict(X_test.drop(["y"], axis=1))
    
    error_msg = f"Daily_Hourly incorret shape X_pred:{X_pred.shape[0]}, X_test:{X_test.shape[0]}!"
    assert X_pred.shape[0]==X_test.shape[0], error_msg
    
    y_pred_nan_sum = X_pred["y_pred"].isna().sum()
    error_msg = f"Daily_Hourly y_pred has {y_pred_nan_sum} None!"
    assert y_pred_nan_sum==0,error_msg
    ##########################################################################
    
    ##########################################################################
    # Monthly
    df_monthly = get_monthly_data(N=24, max_value=10, noise_std=3, anomaly_percentage=0)
    split_time = str(datetime.now().date() - timedelta(days=30.47*4))
    X_train = df_monthly[df_monthly["time"]<split_time].copy()
    X_test = df_monthly[df_monthly["time"]>=split_time].copy()

    tag_features = ["Country"]
    model = AKRuleBasedTS(freqs=["monthly"], tag_features=tag_features, average_num=3, trend_level=1, fillna=True, metric_ci=0.95, inplace=False)
    _ = model.fit_predict(X_train)
    X_pred = model.predict(X_test.drop(["y"], axis=1))
    
    error_msg = f"Daily_Hourly incorret shape X_pred:{X_pred.shape[0]}, X_test:{X_test.shape[0]}!"
    assert X_pred.shape[0]==X_test.shape[0], error_msg
    
    y_pred_nan_sum = X_pred["y_pred"].isna().sum()
    error_msg = f"Daily_Hourly y_pred has {y_pred_nan_sum} None!"
    assert y_pred_nan_sum==0,error_msg
    ##########################################################################
    
    ##########################################################################
    # Yearly-Monthly-Daily
    df_daily = get_yearly_monthly_daily_data(N=48, max_value=10, noise_std=3, anomaly_percentage=0)
    split_time = str(datetime.now().date() - timedelta(days=365.64*1))
    X_train = df_daily[df_daily["time"]<split_time].copy()
    X_test = df_daily[df_daily["time"]>=split_time].copy()

    tag_features = ["Country"]
    model = AKRuleBasedTS(freqs=["yearly", "monthly", "dayofmonth"], tag_features=tag_features, average_num=3, trend_level=1, fillna=True, metric_ci=0.95, inplace=False)
    _ = model.fit_predict(X_train)
    X_pred = model.predict(X_test.drop(["y"], axis=1))
    
    error_msg = f"Daily_Hourly incorret shape X_pred:{X_pred.shape[0]}, X_test:{X_test.shape[0]}!"
    assert X_pred.shape[0]==X_test.shape[0], error_msg
    
    y_pred_nan_sum = X_pred["y_pred"].isna().sum()
    error_msg = f"Daily_Hourly y_pred has {y_pred_nan_sum} None!"
    assert y_pred_nan_sum==0,error_msg
    ##########################################################################
    
    ##########################################################################
    # Yearly-Monthly
    df_month = get_yearly_monthly_data(N=96, max_value=100, noise_std=5.5, anomaly_percentage=5)
    split_time = str(datetime.now().date() - timedelta(days=365.64*1))
    X_train = df_month[df_month["time"]<split_time].copy()
    X_test = df_month[df_month["time"]>=split_time].copy()

    tag_features = ["Country"]
    model = AKRuleBasedTS(freqs=["yearly", "monthly"], tag_features=tag_features, average_num=3, trend_level=1, fillna=True, metric_ci=0.95, inplace=False)
    _ = model.fit_predict(X_train)
    X_pred = model.predict(X_test.drop(["y"], axis=1))
    
    error_msg = f"Daily_Hourly incorret shape X_pred:{X_pred.shape[0]}, X_test:{X_test.shape[0]}!"
    assert X_pred.shape[0]==X_test.shape[0], error_msg
    
    y_pred_nan_sum = X_pred["y_pred"].isna().sum()
    error_msg = f"Daily_Hourly y_pred has {y_pred_nan_sum} None!"
    assert y_pred_nan_sum==0,error_msg
    ##########################################################################
    
    ##########################################################################
    # Yearly
    df_year = get_yearly_data(N=100, max_value=100, noise_std=5.5, anomaly_percentage=5)
    split_time = str(datetime.now().date() - timedelta(days=365.64*10))
    X_train = df_year[df_year["time"]<split_time].copy()
    X_test = df_year[df_year["time"]>=split_time].copy()

    tag_features = ["Country", "City"]
    model = AKRuleBasedTS(freqs=["yearly"], tag_features=tag_features, average_num=3, trend_level=1, fillna=True, metric_ci=0.95, inplace=False)
    _ = model.fit_predict(X_train)
    X_pred = model.predict(X_test.drop(["y"], axis=1))
    
    error_msg = f"Daily_Hourly incorret shape X_pred:{X_pred.shape[0]}, X_test:{X_test.shape[0]}!"
    assert X_pred.shape[0]==X_test.shape[0], error_msg
    
    y_pred_nan_sum = X_pred["y_pred"].isna().sum()
    error_msg = f"Daily_Hourly y_pred has {y_pred_nan_sum} None!"
    assert y_pred_nan_sum==0,error_msg
    ##########################################################################