import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
"""資料匯入 初始設置"""
df_orig = pd.read_csv('2542.csv')
df = df_orig.drop(['YEARS'], axis=1)
x = df.drop(['P'], axis=1)
y = df['P']
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=20, test_size=0.1)
"""OLS Regression"""
lr = LinearRegression()
lr.fit(x_train, y_train)
y_pred = lr.predict(x_test)
sc1 = lr.score(x_test, y_test)
coef1 = lr.coef_
intercept1 = lr.intercept_
"""OLS y_hat"""
y_hat1 = intercept1 + x.dot(coef1)
"""ridge需要資料ragularization. 利用ridge 找一個不會過度擬合但又r2高的coef and intercept. 直接用grid去抓"""
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
rr = Ridge()
param_grid = {
    'alpha': [0.001, 0.01, 0.1, 1, 10, 100]
}
rr2 = GridSearchCV(rr, param_grid, cv=4, n_jobs=-1)
rr2.fit(x_train, y_train)
y_pred2 = rr2.predict(x_test)
bp = rr2.best_params_
sc2 = rr2.best_score_
"""rr2是grid 先轉換才能去跑.coef_, .intercept_. 因為coef, intercept是標準化的數值, 所以要把他們轉回來"""
model = rr2.best_estimator_
coef_re = model.coef_ / scaler.scale_
intercept_re = model.intercept_ - (model.coef_ * scaler.mean_ / scaler.scale_).sum()
"""ridge y_hat"""
y_hat = intercept_re + x.dot(coef_re)
"""秀出圖示 實際值和預測值差異"""
plt.figure(figsize=(10, 6))
sns.lineplot(x='YEARS', y='P', data=df_orig, color='black', label='Actual Price')
sns.lineplot(x='YEARS', y=y_hat, data=df_orig, color='red', label='Predicted Price')
sns.lineplot(x='YEARS', y=y_hat1, data=df_orig, color='blue', label='OLS Predicted Price')
plt.xticks(df_orig['YEARS'][::8])
plt.xlabel('Year')
plt.ylabel('Price')
plt.title('2542')
plt.legend(fontsize=10, frameon=False)
plt.show()
"""結合BETA ALPHA R^2 輸出結果"""
feature_name = ['b1 CR(T-1)', 'b2 DER(T-1)', 'b3 BPS(T-1)', 'b4 DPS']
con_co = pd.DataFrame({'2542': feature_name, 'OLS predicted': coef1, 'Predicted': coef_re})
feature_name2 = ['intercept']
con_int = pd.DataFrame({'2542': feature_name2, 'OLS predicted': intercept1, 'Predicted': intercept_re})
feature_name3 = ['R^2']
con_score = pd.DataFrame({'2542': feature_name3, 'OLS predicted': sc1, 'Predicted': sc2})
con_final = pd.concat([con_score, con_int, con_co]).round(4)
print(con_final)