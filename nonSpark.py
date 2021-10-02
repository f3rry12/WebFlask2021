def linreg1_nonspark():
  import numpy as np
  import pandas as pd
  import os.path

  BASE_DIR = os.path.dirname(os.path.abspath(__file__))
  url = os.path.join(BASE_DIR, "Data_Generate_EA.csv")

  # Importing the dataset => ganti sesuai dengan case yg anda usulkan
  # a. Min. 30 Data dari case data simulasi dari yg Anda usulkan
  # b. Min. 30 Data dari real case, sesuai dgn yg Anda usulkan dari tugas minggu ke-3 (dari Kaggle/UCI Repository)
  # url = "./Salary_Data.csv"
  dataset = pd.read_csv(url)
  X = dataset['CountAll'].values.reshape(-1,1)
  y = dataset['StockOpen'].values

  # Splitting the dataset into the Training set and Test set
  # Lib-nya selain sklearn/ Tensorflow/ Keras/ PyTorch/ etc
  from sklearn.model_selection import train_test_split
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

  # Hitung Mape
  # from sklearn.metrics import mean_absolute_percentage_error

  # Feature Scaling
  """from sklearn.preprocessing import StandardScaler
  sc_X = StandardScaler()
  X_train = sc_X.fit_transform(X_train)
  X_test = sc_X.transform(X_test)
  sc_y = StandardScaler()
  y_train = sc_y.fit_transform(y_train)"""

  # Fitting Simple Linear Regression to the Training set
  from sklearn.linear_model import LinearRegression
  regressor = LinearRegression()
  regressor.fit(X_train, y_train)

  # Predicting the Test set results
  y_pred = regressor.predict(X_test)
  y_pred2 = regressor.predict(X_train)


  aktual, predict = y_train, y_pred2
  mape = np.sum(np.abs(((aktual - predict)/aktual)*100))/len(predict)

  thisdict =	{
  "y_aktual": list(y_train),
  "y_prediksi": list(y_pred2),
  "mape": mape
  }
  return thisdict

def linreg2_nonspark():
  import numpy as np
  import pandas as pd
  import os.path

  BASE_DIR = os.path.dirname(os.path.abspath(__file__))
  url = os.path.join(BASE_DIR, "Data_Generated_daily_adjusted_EA.csv")

  dataset = pd.read_csv(url)
  X = dataset['CountAll'].values.reshape(-1,1)
  y = dataset['StockOpen'].values

  from sklearn.model_selection import train_test_split
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)


  # Feature Scaling
  """from sklearn.preprocessing import StandardScaler
  sc_X = StandardScaler()
  X_train = sc_X.fit_transform(X_train)
  X_test = sc_X.transform(X_test)
  sc_y = StandardScaler()
  y_train = sc_y.fit_transform(y_train)"""

  # Fitting Simple Linear Regression to the Training set
  from sklearn.linear_model import LinearRegression
  regressor = LinearRegression()
  regressor.fit(X_train, y_train)

  # Predicting the Test set results
  y_pred = regressor.predict(X_test)
  y_pred2 = regressor.predict(X_train)


  aktual, predict = y_train, y_pred2
  mape = np.sum(np.abs(((aktual - predict)/aktual)*100))/len(predict)

  thisdict =	{
  "y_aktual": list(y_train),
  "y_prediksi": list(y_pred2),
  "mape": mape
  }
  return thisdict

def savepkl1():
  import numpy as np
  import pandas as pd
  import os.path

  BASE_DIR = os.path.dirname(os.path.abspath(__file__))
  url = os.path.join(BASE_DIR, "Data_Generate_EA.csv")

  dataset = pd.read_csv(url)
  X = dataset.iloc[:, :-1].values
  y = dataset.iloc[:, 1].values


  from sklearn.model_selection import train_test_split
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

  # Hitung Mape
  # from sklearn.metrics import mean_absolute_percentage_error

  # Feature Scaling
  """from sklearn.preprocessing import StandardScaler
  sc_X = StandardScaler()
  X_train = sc_X.fit_transform(X_train)
  X_test = sc_X.transform(X_test)
  sc_y = StandardScaler()
  y_train = sc_y.fit_transform(y_train)"""

  # Fitting Simple Linear Regression to the Training set
  from sklearn.linear_model import LinearRegression
  regressor = LinearRegression()
  myModelReg = regressor.fit(X_train, y_train)

  import joblib
  # Simpan hasil model fit
  with open(os.path.join(BASE_DIR, "ModelReg1.joblib.pkl"), 'wb') as f:
    joblib.dump(myModelReg, f, compress=9)

  # Load hasil model fit
  with open(os.path.join(BASE_DIR, "ModelReg1.joblib.pkl"), 'rb') as f:
    myModelReg_load = joblib.load(f)


  y_pred = myModelReg_load.predict(X_test)
  y_pred2 = myModelReg_load.predict(X_train)

  aktual, predict = y_train, y_pred2
  mape = np.sum(np.abs(((aktual - predict)/aktual)*100))/len(predict)

  thisdict =	{
  "y_aktual": list(y_train),
  "y_prediksi": list(y_pred2),
  "mape": mape
  }
  return thisdict

def savepkl2():
  import numpy as np
  import pandas as pd
  import os.path

  BASE_DIR = os.path.dirname(os.path.abspath(__file__))
  url = os.path.join(BASE_DIR, "Data_Generated_daily_adjusted_EA.csv")

  dataset = pd.read_csv(url)
  X = dataset.iloc[:, :-1].values
  y = dataset.iloc[:, 1].values


  from sklearn.model_selection import train_test_split
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

  # Hitung Mape
  # from sklearn.metrics import mean_absolute_percentage_error

  # Feature Scaling
  """from sklearn.preprocessing import StandardScaler
  sc_X = StandardScaler()
  X_train = sc_X.fit_transform(X_train)
  X_test = sc_X.transform(X_test)
  sc_y = StandardScaler()
  y_train = sc_y.fit_transform(y_train)"""

  # Fitting Simple Linear Regression to the Training set
  from sklearn.linear_model import LinearRegression
  regressor = LinearRegression()
  myModelReg = regressor.fit(X_train, y_train)

  import joblib
  # Simpan hasil model fit
  with open(os.path.join(BASE_DIR, "ModelReg2.joblib.pkl"), 'wb') as f:
    joblib.dump(myModelReg, f, compress=9)

  # Load hasil model fit
  with open(os.path.join(BASE_DIR, "ModelReg2.joblib.pkl"), 'rb') as f:
    myModelReg_load = joblib.load(f)


  y_pred = myModelReg_load.predict(X_test)
  y_pred2 = myModelReg_load.predict(X_train)

  aktual, predict = y_train, y_pred2
  mape = np.sum(np.abs(((aktual - predict)/aktual)*100))/len(predict)

  thisdict =	{
  "y_aktual": list(y_train),
  "y_prediksi": list(y_pred2),
  "mape": mape
  }
  return thisdict