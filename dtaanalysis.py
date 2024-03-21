import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm

#Convert stata file to csv
data = pd.io.stata.read_stata('/Users/dbala/Downloads/UN Comtrade-1.dta')
data.to_csv('UN Comtrade-1.csv')

#Create DataFrame
df = pd.read_csv('UN Comtrade-1.csv')

#Find the unique number of countries and comodities in this dataset
unique_countries = df['reporter'].nunique()
unique_commodity = df['commodity'].nunique()

#Group the data by year
agg_data = df.groupby('year').sum()

#Based off the grouped data get the imports and exports for the time series graph
plt.plot(agg_data.index, agg_data['imports'], label='Import')
plt.plot(agg_data.index, agg_data['exports'], label='Export')

#Make the Time Series Graph
plt.title('Import and Export Over Years')
plt.xlabel('Year')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()

#Make the histogram
plt.figure(figsize=(10,6))
plt.hist(df['exports'], bins=10, color='blue', alpha=0.7)
plt.xticks(np.arange(0, 200000, step=50000))
plt.title('Histogram of Exports')
plt.xlabel('Export Value')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

#Find the biggest exporter and how much they exported for commodity code 84 and in the year 2008
biggest_exporter = df[(df['commoditycode'] == 84) & (df['year'] == 2008)].groupby('reporter')['exports'].sum().idxmax()
export_value = df[(df['commoditycode'] == 84) & (df['year'] == 2008)].groupby('reporter')['exports'].sum().max()
print(f"The biggest exporter in 2008 for commodity 84 is {biggest_exporter} with an export value of {export_value}")

#Rank the top 5 exporters and importers based off the data set
top_exporters = df.groupby('reporter')['exports'].sum().nlargest(5)
top_importers = df.groupby('reporter')['imports'].sum().nlargest(5)

#Create a Linear Regression Model based off the top exporters and top importers
X = top_exporters.values.reshape(-1, 1)
y = top_importers.values.reshape(-1, 1)

model = LinearRegression()
model.fit(X, y)

y_pred = model.predict(X)

#Plot the Linear Regression Model
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X, y_pred, color='red', linewidth=2, label='Fitted Line')
plt.title('Import vs. Export by Country')
plt.xlabel('Exports (in billion $)')
plt.ylabel('Imports (in billion $)')
plt.legend()
plt.grid(True)
plt.show()

#Find the value of r_squared
r_squared = r2_score(y, y_pred)

#Find the outliers
residuals = np.abs(y - y_pred)
outlier_threshold = np.mean(residuals) + 2 * np.std(residuals)
outliers = df[residuals.flatten() > outlier_threshold]

#Create a dataframe for the trade flow data
df1 = pd.DataFrame(pd.read_excel('/Users/dbala/Downloads/gravity_flow_no_zeros-1.xlsx'))

#Calculate the logarithm for all the following variable
df1['log_flow'] = np.log(df1['flow'])
df1['log_gdpo'] = np.log(df1['gdp_o'])
df1['log_gdpd'] = np.log(df1['gdp_d'])
df1['log_distw'] = np.log(df1['distw'])

#Regress the log variable on the trade flow
X = df1[['log_gdpo', 'log_gdpd', 'log_distw']]
X = sm.add_constant(X)

y = df1['log_flow']

model = sm.OLS(y, X).fit()

#Print out the table that was given
print(model.summary())

#Add more variables to the log table
X_extended = df1[['log_gdpo', 'log_gdpd', 'log_distw', 'contig', 'rta', 'comcur']]
X_extended = sm.add_constant(X_extended)

model_extended = sm.OLS(y, X_extended).fit()

#Print out the table that was given
print(model_extended.summary())

#Create a dataframe for the ricardian model
df2 = pd.DataFrame(pd.read_excel('/Users/dbala/Downloads/ricardian.xlsx'))

#Calculate the relative productivity
df2['relative_productivity'] = df2['labor unit required_foreign'] / df2['labor unit required_home']

#Create new variable called home produce
relative_wage = 2
df2['home_produce'] = np.where(df2['relative_productivity'] * relative_wage > 1, 1, 0)

#Regression Model for the first 150 products
X_ricardian = df2.iloc[:150][['relative_productivity']]
X_ricardian = sm.add_constant(X_ricardian)
y_ricardian = df2.iloc[:150]['home_produce']

model_ricardian = sm.OLS(y_ricardian, X_ricardian).fit()

#Prediction for the last 50 products
X_predict = sm.add_constant(df2.iloc[150:][['relative_productivity']])
predictions = model_ricardian.predict(X_predict)
predictions_binary = np.where(predictions > 0.5, 1, 0)

#Put the prediction back in the dataframe
df2.loc[150:, 'predicted_produce'] = predictions_binary

#Actual values for the last 50 products
actual = df2.iloc[150:]['home_produce']

#Calculate the mean squared error and print it out
mse = mean_squared_error(actual, predictions_binary)
print(f'Mean Squared Error: {mse}')