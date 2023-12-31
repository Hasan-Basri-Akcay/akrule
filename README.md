# AKRULE
[![GitHub Workflow Status (with event)](https://img.shields.io/github/actions/workflow/status/Hasan-Basri-Akcay/akrule/python-publish.yml?label=pytest&logo=github)](https://github.com/Hasan-Basri-Akcay/akrule/actions)
[![GitHub Workflow Status (with event)](https://img.shields.io/github/actions/workflow/status/Hasan-Basri-Akcay/akrule/python-publish.yml?label=python-package&logo=github)](https://github.com/Hasan-Basri-Akcay/akrule/actions)
[![Docs](https://img.shields.io/badge/docs-passing-green)](https://medium.com/@hasan.basri.akcay)
[![PyPI](https://img.shields.io/pypi/v/akrule?logo=python&color=blue)](https://pypi.org/project/akrule/)
[![Python](https://img.shields.io/pypi/pyversions/akrule?logo=python)](https://pypi.org/project/akrule/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

Akrule stands out as a powerful rule-based time series forecasting machine learning model designed for optimal performance in both real-time training and prediction scenarios. Its notable feature lies in its swift training capabilities, enabling efficient and timely predictions. Akrule goes beyond standard forecasting models by providing insightful upper and lower boundaries through the integration of confidence intervals and bootstrapping techniques. This unique attribute makes it particularly valuable for anomaly detection, as it empowers users to identify deviations from expected patterns.

Akrule's effectiveness is underscored by its keen sensitivity to seasonality and trend analysis, making it well-suited for applications where these factors play a crucial role in influencing time series data. By harnessing the power of Akrule, users can not only achieve accurate predictions but also gain a comprehensive understanding of the potential variations and uncertainties associated with their data, enhancing the model's utility across diverse domains.

Detailed [Medium](https://medium.com/@hasan.basri.akcay) post on using akrule.

## Installation
```
pip install akrule
```
## Dependencies
akrule requires:
  * Python (>= 3.9)
  * NumPy (>= 1.24.4)
  * Pandas (>= 2.1.2)

## Time-Series Forecasting and Anomaly Detection
```
from akrule.time_series import get_weekly_daily_hourly_data
from akrule.time_series import plot_pred

df_daily = get_weekly_daily_data(N=10, max_value=10, noise_std=3, anomaly_percentage=20)
split_time = str(datetime.now().date() - timedelta(days=7*3))
X_train_daily = df_daily[df_daily["time"]<split_time].copy()
X_test_daily = df_daily[df_daily["time"]>=split_time].copy()

tag_features = ["Country"]
model = AKRuleBasedTS(freqs=["weekly", "dayofweek"], tag_features=tag_features, average_num=3, trend_level=1, fillna=True,
                      metric_ci=0.90, inplace=False)
X_val = model.fit_predict(X_train_daily)
X_pred = model.predict(X_test_daily.drop(["y"], axis=1))

plot_pred(X_val=X_val, X_pred=X_pred, tag_features=tag_features, figsize=(16,4))
```
<img src="/outputs/weekly_daily_ci90.png?raw=true"/>

```
tag_features = ["Country"]
model = AKRuleBasedTS(freqs=["weekly", "dayofweek"], tag_features=tag_features, average_num=3, trend_level=1, fillna=True,
                      metric_ci=0.70, inplace=False)
X_val = model.fit_predict(X_train_daily)
X_pred = model.predict(X_test_daily.drop(["y"], axis=1))

plot_pred(X_val=X_val, X_pred=X_pred, tag_features=tag_features, figsize=(16,4))
```
<img src="/outputs/weekly_daily_ci70.png?raw=true"/>
