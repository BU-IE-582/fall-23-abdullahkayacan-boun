# %%
import pandas as pd
import numpy as np
import os

# %%
# latest_from_clipboard = pd.read_clipboard()
# latest_from_clipboard.sort_values("timestamp",ascending=False).head(30)

# %%
old_date = "2024-01-11"
new_date = "2024-01-12"
sheet_name = "20231210_"+new_date.replace("-","")

latest_from_clipboard = pd.read_excel(f"https://docs.google.com/spreadsheets/d/1-YwqS8sBO6sMMRRJV9UIKelAcPtZO6obqEKepCZ9GSU/export?exportFormat=xlsx",sheet_name=sheet_name)
# print(latest_from_clipboard)

# %%
data_file_paths = ["data/"+e for e in os.listdir("data")]
# data_dict = {}
raw_data = []
for path in data_file_paths:
    # data_dict[path.split("_")[-2]] = pd.read_csv(path)
    raw_data.append(pd.read_csv(path))
raw_data = pd.concat(raw_data,axis=0)
raw_data = pd.concat([raw_data,latest_from_clipboard],axis=0)

# %%
df = raw_data.copy()
df["date"] = [e[:10] for e in df["timestamp"]]
df = df[["timestamp","date"]].drop_duplicates()
# df.groupby("date").count().sort_values("timestamp")[:50]

# %%
duplicates = raw_data.groupby(["short_name","timestamp"]).agg(["count","min","max"]).reset_index()#.sort_values("price",ascending=False).iloc[:50]

duplicates.columns = [e[0] if e[1] == "" else e[0]+"_"+e[1] for e in duplicates.columns]
duplicates = duplicates[duplicates["price_count"] > 1]
duplicates["price_dif"] = duplicates["price_max"]/duplicates["price_min"]
# duplicates.sort_values("price_dif",ascending=False)


# %%
data_all_indices = pd.DataFrame()
short_name = raw_data[["short_name"]].drop_duplicates()
short_name["key"] = 0
timestamp = raw_data[["timestamp"]].drop_duplicates()
timestamp["key"] = 0
data_all_indices = pd.merge(short_name,timestamp,how="outer").drop("key",1)

new_date_next = str(pd.to_datetime(new_date)+pd.DateOffset(days=1))[:10]
new_day_indices = data_all_indices[(data_all_indices["timestamp"]>new_date) & (data_all_indices["timestamp"]<new_date_next)].copy()
new_day_indices["timestamp"] = [e.replace(old_date,new_date) for e in new_day_indices["timestamp"]]

data_all_indices = data_all_indices[data_all_indices["timestamp"]<new_date]
data_all_indices = pd.concat([data_all_indices,new_day_indices],axis = 0)

# %%


# %%
data = raw_data.copy()
data = data.groupby(["short_name","timestamp"]).median().reset_index()
data = pd.merge(data_all_indices,data,"left")
data = data.sort_values(["timestamp","short_name"]).reset_index(drop=True)


data["timestamp"] = pd.to_datetime(data["timestamp"])
data["year"] = data["timestamp"].dt.year
data["monthday"] = data["timestamp"].dt.day
data["weekday"] = data["timestamp"].dt.weekday
data["month"] = data["timestamp"].dt.month
# data["hour"] = data["timestamp"].dt.hour
data["hour"] = data.groupby(["short_name","year","month","monthday"])['timestamp'].transform("rank").astype(int)

data["price"] = data.groupby(["short_name"])["price"].ffill()
data["prev_close"] = data.groupby(["short_name","year","month","monthday"])['price'].transform("last")#.shift(30)
data["prev_close"] = data.groupby(["short_name"])["prev_close"].shift()
data["prev_close"][data["hour"] != 1] = np.nan
data["prev_close"] = data.groupby(["short_name"])["prev_close"].ffill()
# data["prev_close"] = data.groupby(["short_name","year","month","monthday"])['price'].transform("last").shift(30)
# data["prev_close"][data["hour"] != 1] = np.nan
# data["prev_close"] = data.groupby(["short_name"])["prev_close"].ffill()
# data["change"] = data["price"]/data.groupby(["short_name"])["price"].shift()-1
data["change"] = data["price"]/data["prev_close"]-1
data["change"] = data["change"].fillna(0)

for i in range(20):
    data[f"change_lag_prev_{str(i+1).zfill(2)}"] = data.groupby(["short_name"])["change"].shift(11+10*i)
    data[f"change_lag_exact_{str(i+1).zfill(2)}"] = data.groupby(["short_name"])["change"].shift(10+10*i)
    data[f"change_lag_next_{str(i+1).zfill(2)}"] = data.groupby(["short_name"])["change"].shift(9+10*i)

# data = pd.concat([data,pd.get_dummies(data[["short_name"]],"short_name")],axis=1)
data["starting_price"] = data.groupby(["short_name"])['price'].transform("first")

data["change_mean"] = np.exp(data["change"]+1)
data["change_mean"] = np.log(data.groupby("timestamp")["change_mean"].transform(np.mean))-1
data["change_mean_lag"] = data["change_mean"].shift(300)

data[f"naive_forecast_1"] = data.groupby(["short_name"])["price"].shift(10)
data[f"naive_forecast_2"] = data["prev_close"]

data

# %%
target_col = "change"
feature_cols = ['monthday', 'weekday',
       'month', 'hour', 'change_lag_prev_01',
       'change_lag_exact_01', 'change_lag_next_01', 'change_lag_prev_02',
       'change_lag_exact_02', 'change_lag_next_02', 'change_lag_prev_03',
       'change_lag_exact_03', 'change_lag_next_03', 'change_lag_prev_04',
       'change_lag_exact_04', 'change_lag_next_04', 'change_lag_prev_05',
       'change_lag_exact_05', 'change_lag_next_05', 'change_lag_prev_06',
       'change_lag_exact_06', 'change_lag_next_06', 'change_lag_prev_07',
       'change_lag_exact_07', 'change_lag_next_07', 'change_lag_prev_08',
       'change_lag_exact_08', 'change_lag_next_08', 'change_lag_prev_09',
       'change_lag_exact_09', 'change_lag_next_09', 'change_lag_prev_10',
       'change_lag_exact_10', 'change_lag_next_10', 'change_lag_prev_11',
       'change_lag_exact_11', 'change_lag_next_11', 'change_lag_prev_12',
       'change_lag_exact_12', 'change_lag_next_12',
       'starting_price',"change_mean_lag"
       # 'short_name_AKBNK', 'short_name_ARCLK',
       # 'short_name_ASELS', 'short_name_BIMAS', 'short_name_DOHOL',
       # 'short_name_EKGYO', 'short_name_EREGL', 'short_name_FROTO',
       # 'short_name_GARAN', 'short_name_GUBRF', 'short_name_HALKB',
       # 'short_name_ISCTR', 'short_name_KCHOL', 'short_name_KOZAA',
       # 'short_name_KOZAL', 'short_name_KRDMD', 'short_name_PETKM',
       # 'short_name_PGSUS', 'short_name_SAHOL', 'short_name_SASA',
       # 'short_name_SISE', 'short_name_TAVHL', 'short_name_TCELL',
       # 'short_name_THYAO', 'short_name_TKFEN', 'short_name_TTKOM',
       # 'short_name_TUPRS', 'short_name_VAKBN', 'short_name_VESTL',
       # 'short_name_YKBNK'
       ]
print(feature_cols)
data_for_train_all = data.dropna().reset_index(drop=True)

# %%
data_for_train_all.iloc[-300:][["short_name","timestamp","price"]]#.drop_duplicates()

# %%
fold_size = data_for_train_all.shape[0]//11
fold_size = (fold_size - (fold_size%300))
fold_indices = np.array([[0,(e+1)*fold_size,(e+1)*fold_size,(e+2)*fold_size] for e in range(10)])
fold_indices[-1,-1] = data_for_train_all.shape[0]
fold_indices

# %%
# data_for_train_all.iloc[fold_end_indices[-1]:].shape

# %%
import lightgbm as lgb

# %%
# np.random.seed(1)
# models = []
# change_preds = []
# for i,(train_start,train_end,test_start,test_end) in enumerate(fold_indices):
#     X = data_for_train_all[feature_cols].iloc[train_start:train_end,:]
#     y = data_for_train_all[target_col].iloc[train_start:train_end]
#     model = lgb.LGBMRegressor()
#     model.fit(X,y)
#     X_test = data_for_train_all[feature_cols].iloc[test_start:test_end,:]
#     y_test_pred = model.predict(X_test)
#     models.append(model)
#     change_preds.append(y_test_pred)
#     print(i)

# %%
np.random.seed(1)
X = data_for_train_all[feature_cols].iloc[:-300,:]
y = data_for_train_all[target_col].iloc[:-300]
model = lgb.LGBMRegressor()
model.fit(X,y)
X_test = data_for_train_all[feature_cols].iloc[-300:,:]
y_test_pred = model.predict(X_test)

# %%
# change_preds_np = np.array([])
# for cp in change_preds:
#     change_preds_np = np.append(change_preds_np,cp)
# change_preds_np = np.append([np.nan]*fold_size,change_preds_np)
# change_preds_np.shape

# %%
change_preds_np = np.append([np.nan]*X.shape[0],y_test_pred)
change_preds_np.shape

# %%
data_for_train_all["model_change_pred"] = change_preds_np
data_for_train_all["model_price_pred"] = (data_for_train_all["model_change_pred"]+1)*data_for_train_all["prev_close"]

# %%
data_for_train_all.isna().sum()

# %%
data_for_train_all.iloc[-300:][["short_name","timestamp","model_price_pred"]]#.pivot(columns="short_name",index="timestamp")#.sort_index()

# %%
new_prices = latest_from_clipboard[(latest_from_clipboard["timestamp"] > new_date)&(latest_from_clipboard["timestamp"] < new_date_next)].sort_values(["short_name","timestamp"]).reset_index(drop=True)

new_day_pred = data_for_train_all.iloc[-300:][["short_name","timestamp","model_price_pred"]].sort_values(["short_name","timestamp"]).reset_index(drop=True)
new_day_pred["model_price_pred"] = new_day_pred["model_price_pred"].values.round(4)
new_day_pred = pd.concat([new_day_pred,new_prices["price"]],1)

# print(new_day_actual)
new_day_pred["abs_error"] = (new_day_pred["model_price_pred"]-new_day_pred["price"]).abs()
new_day_eval_agg = new_day_pred.groupby("short_name").sum()
new_day_eval_agg["wmape"] = (new_day_eval_agg["abs_error"]/new_day_eval_agg["price"])
new_day_eval_agg.mean()

# %%
model_pred_df = data_for_train_all.iloc[-300:][["short_name","timestamp","naive_forecast_2"]].pivot(columns="short_name",index="timestamp").sort_index()
model_pred_df.columns = [c[1] for c in model_pred_df.columns] 
model_pred_df
model_pred_dict = {}
for c in model_pred_df.columns:
    model_pred_dict[c] = list(model_pred_df[c].values.round(4))
print(model_pred_dict)

# %%
model_pred_df = data_for_train_all.iloc[-300:][["short_name","timestamp","model_price_pred"]].pivot(columns="short_name",index="timestamp").sort_index()
model_pred_df.columns = [c[1] for c in model_pred_df.columns] 
model_pred_df
model_pred_dict = {}
for c in model_pred_df.columns:
    model_pred_dict[c] = list(model_pred_df[c].values.round(4))
print(model_pred_dict)


