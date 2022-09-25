from statistics import correlation
import pandas as pd
import numpy as np
import plotly.express as px
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# read the transaction file
df = pd.read_csv('credit card.csv')

# checks if there is any null value
df[df.isnull().sum(axis=1) > 0]

# Exploring transaction types
type = df['type'].value_counts()
transactions = type.index
quantity = type.values

# creating transaction graphs
figure = px.pie(df, values=quantity, names=transactions, title="Distribution of transaction types")
figure.show()

# verifying correlation between data and column
correlation = df.corr()
correlation['isFraud'].sort_values(ascending=False)

# with map() function change 'type' from catagorical data to numeric data
df['type'] = df['type'].map({'CASH_OUT': 1, 'PAYMENT': 2,
                             'CASH_IN': 3, 'TRANSFER': 4,
                             'DEBIT': 5})

# if in the column 'isFraud' do the opposite
df['isFraud'] = df['isFraud'].map({0: 'No Fraud', 1: 'Fraud'})

df.head()

# separating the data into training data and test data

x = np.array(df[['type', 'amount', 'oldbalanceOrg', 'newbalanceOrig']])
y = np.array(df[['isFraud']])

# training the model
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.10, random_state=42)
model = DecisionTreeClassifier()
model.fit(xtrain, ytrain)
model.score(xtest, ytest)

# prediction
# features = [type, amount, oldbalanceOrg, newbalanceOrig]
features = np.array([[4, 9000.60, 9000.60, 0.0]])
model.predict(features)
