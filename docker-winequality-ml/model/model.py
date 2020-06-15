# Import dependencies
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib

# Import data
data = pd.read_csv('../data/winequality.csv',sep=';')
print(data.columns)
X_data = data[['fixed acidity', 'citric acid', 'sulphates', 'alcohol']]

# Create output classes 
WineQuality = []
for i in data['quality']:
    if i >= 1 and i <= 3:
        WineQuality.append('Bad')
    elif i >= 4 and i <= 7:
        WineQuality.append('Average')
    elif i >= 8 and i <= 10:
        WineQuality.append('Good')
data['WineQuality'] = WineQuality

X = X_data
y = data['WineQuality']

# Split train and test data
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

# Train Model
rf = RandomForestClassifier()
rf.fit(x_train, y_train)
rf_predict=rf.predict(x_test)

print('Accuracy of the model: {}%'.format(accuracy_score(y_test, rf_predict)*100))

joblib.dump(rf,"./RandomForest.pkl", protocol =2) # Save Model
print("Model dumped")
model_columns = list(X_data.columns)
joblib.dump(model_columns, "./model_columns.pkl", protocol =2) # Save column names
print("Model columns dumped")
