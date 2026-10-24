import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# 1. Memuat Data
data = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML240EN-SkillsNetwork/labs/data/Ames_Housing_Sales.csv")

# 2. Pemrosesan Data
data_ohc = data.copy()
categorical_cols = data_ohc.select_dtypes(include=['object']).columns

le = LabelEncoder()
ohc = OneHotEncoder()

for col in categorical_cols:
    dat = le.fit_transform(data_ohc[col]).astype(int)
    data_ohc = data_ohc.drop(col, axis=1) 
    
    new_dat = ohc.fit_transform(dat.reshape(-1,1))
    n_cols = new_dat.shape[1]
    col_names = ['_'.join([col, str(x)]) for x in range(n_cols)]
    
    new_df = pd.DataFrame(new_dat.toarray(), index=data_ohc.index, columns=col_names)
    data_ohc = pd.concat([data_ohc, new_df], axis=1)

# 3. Pembagian dan Penskalaan Data
y_col = 'SalePrice'
feature_cols = [x for x in data_ohc.columns if x != y_col]
X_train, X_test, y_train, y_test = train_test_split(data_ohc[feature_cols], data_ohc[y_col], test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Pelatihan Model & Visualisasi
LR = LinearRegression()
LR.fit(X_train_scaled, y_train)

y_test_pred = LR.predict(X_test_scaled)
print("Mean Squared Error (Test):", mean_squared_error(y_test, y_test_pred))

plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_test_pred, alpha=.5)
plt.xlabel('Ground truth')
plt.ylabel('Predictions')
plt.title('House Price Predictions vs Truth')
plt.show()