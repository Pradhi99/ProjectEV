

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

path = "C:\\Users\\craro\\.spyder-py3\\Pradhi\\"

# Read data from the input file
input_file = path + 'Combines_NJ_WA_Long_Form.csv'
data = pd.read_table(input_file, delimiter=',', header=[0])
data.info(verbose = True, show_counts=True)


# Filter the data to include only county rows
data2 = data[ data['Geography'] == "County" ]
data2.info(verbose = True, show_counts=True) 


# descriptive statistics
data2.describe()

data2.describe().to_csv(path + "Descriptive Stats.csv")


# correlations

df_subset = data2[["EV charging stations", "Unemployment Rate", 
                   "Population Estimate", "Per capita personal income",
                   "EV registered"]]
correlation_matrix = df_subset.corr()
print(correlation_matrix)



# density plot for EV Charging Stations
import seaborn as sns 
import matplotlib.pyplot as plt 

var_name = "EV charging stations"
var_title = "Density Plot for " + var_name
print(var_title, var_name)  

fig, ax = plt.subplots(figsize=(10, 6))  
plt.ticklabel_format(style='plain')
plt.xlabel(var_title)
plt.ylabel("Frequency")
plt.title(var_title)

# plotting density plot for carat using distplot() 
sns.kdeplot(np.array( data2[ var_name ] ), bw_method = 0.5)



# scatterplot with best fit regression line; y is dependent variable
import seaborn as sns   
x_var = data2['Unemployment Rate']
y_var = data2['EV charging stations']

x_name = 'Unemployment Rate'
y_name = 'EV charging stations'

plt.close()

fig, ax = plt.subplots(figsize=(10, 8))  
sns.set(font_scale=1.0)

ax = sns.regplot(data2, x = x_var, y = y_var, 
    scatter_kws={"color": "black"}, line_kws={"color": "red"})

plt.suptitle("Scatter plot of \n" + x_name + " vs " + y_name)
plt.xlabel(x_name)
plt.ylabel(y_name)
plt.ticklabel_format(style='plain')
plt.show()

plt.savefig(path + "ScatterPlot_Unemployment_Rate.png")

  

# Extract data from the data frame into arrays y and x

y = data2["EV charging stations"].values 

x = data2[['Unemployment Rate', 'Population Estimate', 
           'Per capita personal income', 'EV registered']].values.reshape(-1, 4)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()   
X_scaled = scaler.fit_transform(x)
x = X_scaled

print(y)

print(x)


# add y intercept
import statsmodels.api as sm
x = sm.add_constant(x)  # add y-intercept
print(x)

# create the train/test split    
from sklearn.model_selection import train_test_split
train_X, test_X, train_y, test_y = train_test_split(x, y, 
                test_size = 0.20, random_state = 11) 

# fit the linear regression model in sklearn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as MSE
skreg = LinearRegression().fit(train_X, train_y)

# set up the cross-validation 
from sklearn.model_selection import cross_val_predict, KFold
kfold = KFold(n_splits=5, shuffle=True, random_state=11)   


import numpy as np
from sklearn.metrics import mean_squared_error as MSE
from sklearn.model_selection import cross_val_predict

# Predict train from KFold cross-validation
y_pred = cross_val_predict(skreg, train_X, train_y, cv=kfold, method='predict')
# Compute RMSE and MSE
mse_kfold_train = MSE(train_y, y_pred)
rmse_kfold_train = np.sqrt(mse_kfold_train)
print("KFold MSE train: % f" %(mse_kfold_train))
print("KFold RMSE train : % f" %(rmse_kfold_train))

# Predict test from KFold cross-validation    
y_pred = cross_val_predict(skreg, test_X, test_y, cv=kfold, method='predict')
# Compute RMSE and MSE
mse_kfold_test = MSE(test_y, y_pred)
rmse_kfold_test = np.sqrt(mse_kfold_test)
print("KFold MSE test: % f" %(mse_kfold_test))
print("KFold RMSE test: % f" %(rmse_kfold_test))


# fit the linear regression model in stats models
import statsmodels.api as sm
model = sm.OLS(train_y, train_X)    
ols_fit = model.fit()
ols_fit.summary()

# Predict train from OLS
ols_pred_train = ols_fit.predict(train_X) 
mse_ols_train = ols_fit.mse_model
rmse_ols_train = np.sqrt(ols_fit.mse_model)
print("OLS MSE train: % f" %(mse_ols_train))
print("OLS RMSE train: % f" %(rmse_ols_train))

# Predict test from OLS
ols_pred_test = ols_fit.predict(test_X)
mse_ols_test = MSE(test_y, ols_pred_test)
rmse_ols_test = np.sqrt(mse_ols_test)
print("OLS MSE test: % f" %(mse_ols_test))
print("OLS RMSE test: % f" %(rmse_ols_test))
    
print( ols_fit.summary())



        
