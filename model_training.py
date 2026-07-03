import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
import xgboost as xgb

df = pd.read_csv('data/metal_data.csv')
features = ['M', 'n', 'sigma_y', 'Tg']
y = df['E'].values.ravel()

df['M_per_n'] = df['M'] / df['n']
df['sigma_y_over_Tg'] = df['sigma_y'] / df['Tg']
df['inv_Tg'] = 1.0 / df['Tg']
df['sigma_y_sq'] = df['sigma_y']**2
df['sigma_y_Tg'] = df['sigma_y'] * df['Tg']
df['M_norm'] = df['M'] / df['M'].max()
df['log_M'] = np.log(df['M'])

engineered_features = ['M', 'n', 'sigma_y', 'Tg', 'M_per_n', 'sigma_y_over_Tg',
                       'inv_Tg', 'sigma_y_sq', 'sigma_y_Tg', 'M_norm', 'log_M']
X_eng = df[engineered_features].values

X_train, X_test, y_train, y_test = train_test_split(X_eng, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_basic = df[features].values
Xb_train, Xb_test, yb_train, yb_test = train_test_split(X_basic, y, test_size=0.2, random_state=42)
scaler_basic = StandardScaler()
Xb_train_scaled = scaler_basic.fit_transform(Xb_train)
Xb_test_scaled = scaler_basic.transform(Xb_test)

results = []
def evaluate_model(name, model, Xtr, Xte, ytr, yte):
    model.fit(Xtr, ytr)
    y_pred = model.predict(Xte)
    y_pred_tr = model.predict(Xtr)
    r2 = r2_score(yte, y_pred)
    r2_tr = r2_score(ytr, y_pred_tr)
    rmse = np.sqrt(mean_squared_error(yte, y_pred))
    mae = mean_absolute_error(yte, y_pred)
    mape = np.mean(np.abs((yte - y_pred) / (yte + 1e-10))) * 100
    results.append({
        'Model': name, 'R2': r2, 'R2_train': r2_tr,
        'RMSE': rmse, 'MAE': mae, 'MAPE': mape
    })

evaluate_model('Linear Regression (Basic)', LinearRegression(), Xb_train_scaled, Xb_test_scaled, yb_train, yb_test)
evaluate_model('Linear Regression (Engineered)', LinearRegression(), X_train_scaled, X_test_scaled, y_train, y_test)

E_pred_ref = 25 + 41.4*Xb_test[:,2] - 0.0046*Xb_test[:,3] + 0.0015*Xb_test[:,2]*Xb_test[:,3]
r2_ref = r2_score(yb_test, E_pred_ref)
rmse_ref = np.sqrt(mean_squared_error(yb_test, E_pred_ref))
mae_ref = mean_absolute_error(yb_test, E_pred_ref)
results.append({
    'Model': 'Reference NRM (Eq.8)', 'R2': r2_ref, 'R2_train': 0,
    'RMSE': rmse_ref, 'MAE': mae_ref,
    'MAPE': np.mean(np.abs((yb_test - E_pred_ref) / (yb_test + 1e-10))) * 100
})

ann_baseline = MLPRegressor(hidden_layer_sizes=(10,), activation='relu',
                            solver='adam', max_iter=5000, random_state=42,
                            learning_rate_init=0.001)
evaluate_model('ANN Baseline (10 neurons)', ann_baseline, Xb_train_scaled, Xb_test_scaled, yb_train, yb_test)

ann_wide = MLPRegressor(hidden_layer_sizes=(64, 32, 16), activation='relu',
                        solver='adam', max_iter=5000, random_state=42,
                        learning_rate_init=0.001)
evaluate_model('ANN Deep (64-32-16)', ann_wide, X_train_scaled, X_test_scaled, y_train, y_test)

rf = RandomForestRegressor(n_estimators=300, max_depth=15, min_samples_leaf=3, random_state=42)
evaluate_model('Random Forest', rf, X_train_scaled, X_test_scaled, y_train, y_test)

xgb_m = xgb.XGBRegressor(n_estimators=300, max_depth=8, learning_rate=0.08,
                         subsample=0.8, colsample_bytree=0.8, random_state=42)
evaluate_model('XGBoost', xgb_m, X_train_scaled, X_test_scaled, y_train, y_test)

svr = SVR(kernel='rbf', C=10, gamma='scale', epsilon=0.1)
evaluate_model('SVR (RBF kernel)', svr, Xb_train_scaled, Xb_test_scaled, yb_train, yb_test)

print()
print('=' * 90)
print('MODEL COMPARISON RESULTS (test set)')
print('=' * 90)
print(f'{"Model":<35} {"R2":<10} {"R2(train)":<12} {"RMSE":<10} {"MAE":<10} {"MAPE(%)":<10}')
print('-' * 90)
for r in sorted(results, key=lambda x: x['R2'], reverse=True):
    print(f'{r["Model"]:<35} {r["R2"]:<10.4f} {r["R2_train"]:<12.4f} {r["RMSE"]:<10.2f} {r["MAE"]:<10.2f} {r["MAPE"]:<10.2f}')

print()
print('=' * 60)
print('FEATURE IMPORTANCE (Random Forest)')
print('=' * 60)
rf.fit(X_train_scaled, y_train)
importances = rf.feature_importances_
for name, imp in sorted(zip(engineered_features, importances), key=lambda x: x[1], reverse=True):
    print(f'  {name:<20} {imp:.4f}')

print()
print('=' * 60)
print('5-FOLD CV OF BEST MODELS')
print('=' * 60)
for name, model in [('XGBoost', xgb_m), ('Random Forest', rf), ('ANN Deep', ann_wide)]:
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    for tr_idx, te_idx in kfold.split(X_eng):
        X_tr, X_te = X_eng[tr_idx], X_eng[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]
        s = StandardScaler()
        X_tr_s = s.fit_transform(X_tr)
        X_te_s = s.transform(X_te)
        if name == 'ANN Deep':
            m = MLPRegressor(hidden_layer_sizes=(64, 32, 16), activation='relu',
                            solver='adam', max_iter=3000, random_state=42)
        elif name == 'XGBoost':
            m = xgb.XGBRegressor(n_estimators=300, max_depth=8, learning_rate=0.08,
                                 subsample=0.8, colsample_bytree=0.8, random_state=42)
        else:
            m = RandomForestRegressor(n_estimators=300, max_depth=15, min_samples_leaf=3, random_state=42)
        m.fit(X_tr_s, y_tr)
        scores.append(r2_score(y_te, m.predict(X_te_s)))
    print(f'  {name:<25} R2 = {np.mean(scores):.4f} +/- {np.std(scores):.4f}')

print()
print('All models trained successfully!')
