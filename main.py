import pandas as pd
from sklearn.tree import DecisionTreeRegressor

home_file_path = '.../input/home-data/train.csv'
home_data = pd.read_csv(home_file_path)
y = home_data.SalePrice
feature_names = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = home_data[feature_names]

home_model = DecisionTreeRegressor(random_state = 1)
home_model.fit(X, y)

predictions = home_model.predict(X)
print("Predicted:")
print(predictions)
print("Actual")
y.head()