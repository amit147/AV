
import pandas as pd

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

file_path ='\\Analytis Vidya Competition\\'
file_name = 'train_NA17Sgz\\train.csv'

train_data = pd.read_csv(file_path + file_name,parse_dates=True,index_col=0)

train_data = pd.get_dummies(train_data, prefix_sep="_",
                              columns=['os_version'])


train_data['impression_time'] = pd.to_datetime(train_data['impression_time'],format="%Y-%m-%d %H:%M:%S")

train_data['weekday'] = train_data.impression_time.dt.dayofweek

features = ['app_code','os_version_old','os_version_latest','os_version_intermediate','is_4G','weekday']

X = train_data[features].values

y = train_data['is_click'].values

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=21)


dt = DecisionTreeRegressor(max_depth=7,min_samples_leaf=100,random_state=3)

dt.fit(X_train,y_train)

y_pred = dt.predict(X_test)

print(roc_auc_score(y_test,y_pred))

test_file_name = 'test_aq1FGdB\\test.csv'

test_data = pd.read_csv(file_path + test_file_name,parse_dates=True,index_col=0)

test_data = pd.get_dummies(test_data, prefix_sep="_",
                              columns=['os_version'])

test_data['impression_time'] = pd.to_datetime(test_data['impression_time'],format="%Y-%m-%d %H:%M:%S")

test_data['weekday'] = test_data.impression_time.dt.dayofweek

test_data = test_data[features]

pred = dt.predict(test_data.values)

df_result = pd.DataFrame(pred,index=test_data.index)

df_result.columns = ['is_click']

df_result.to_csv(file_path + 'result.csv')

