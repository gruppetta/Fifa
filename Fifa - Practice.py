# -*- coding: utf-8 -*-
"""
Created on Mon May 25 09:14:53 2020
Fifa data set 
@author: RG
"""

#Importing the relevant libraries and data 

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder , OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.regression.linear_model as sm

df = pd.read_csv('data.csv')
df.info()

df.columns

# Checking for missing values 
def missing_values_table(df):
        # Total missing values
        mis_val = df.isnull().sum()
        
        # Percentage of missing values
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        
        # Make a table with the results
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        
        # Rename the columns
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        
        # Sort the table by percentage of missing descending
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
        
        # Print some summary information
        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
            "There are " + str(mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")
        
        # Return the dataframe with missing information
        return mis_val_table_ren_columns


missing_values_table(df)

#Drop Missing values given column 
df.dropna(subset=['Wage', 'Value','Skill Moves','International Reputation'],inplace=True)




missing_height = df[df['Height'].isnull()].index.tolist()
missing_weight = df[df['Weight'].isnull()].index.tolist()
if missing_height == missing_weight:
    print('They are same')
else:
    print('They are different')

#Same as above with missing value and wage
zero_value= df[df['Value']==0].index.tolist()
zero_wage = df[df['Wage']==0].index.tolist()  


#Removing data for which we do not have info     
df.drop(df.index[missing_height],inplace =True)
df.drop(['Loaned From','Release Clause','Joined'],axis=1,inplace=True)

df.drop(df.index[zero_value],inplace =True)
df.drop(df.index[zero_wage],inplace =True)



#We wont be using all of the columns - I will start looking at columns and their correlation to value and salary 
df.describe()

df['Age'].corr(df['Value'])

#Cleaning the data from M and K 
def value_to_int(df_value):
    try:
        value = float(df_value[1:-1])
        suffix = df_value[-1:]

        if suffix == 'M':
            value = value * 1000000
        elif suffix == 'K':
            value = value * 1000
    except ValueError:
        value = 0
    return value

df['Value'] = df['Value'].apply(value_to_int)
df['Wage'] = df['Wage'].apply(value_to_int)


#Finding the highest value player and the highest paid 
print('Most valued player : '+str(df.loc[df['Value'].idxmax()]['Name']))
print('Highest earner : '+str(df.loc[df['Wage'].idxmax()]['Name']))
print("--"*40)
print("\nTop Earners")


#DROP UNNECESSARY VALUES
df = pd.read_csv('data.csv')
drop_cols = df.columns[28:54]
df = df.drop(drop_cols, axis = 1)
df = df.drop(['Unnamed: 0','ID','Photo','Flag','Club Logo','Jersey Number','Special','Body Type', 
               'Weight','Height','Contract Valid Until','Name','Club'], axis = 1)
df = df.dropna()
df.head()

#Creating a simplified version of the player positions 
def simple_position(df):
    if (df['Position'] == 'GK'):
        return 'GK'
    elif ((df['Position'] == 'RB') | (df['Position'] == 'LB') | (df['Position'] == 'CB') | (df['Position'] == 'LCB') | (df['Position'] == 'RCB') | (df['Position'] == 'RWB') | (df['Position'] == 'LWB') ):
        return 'DF'
    elif ((df['Position'] == 'LDM') | (df['Position'] == 'CDM') | (df['Position'] == 'RDM')):
        return 'DM'
    elif ((df['Position'] == 'LM') | (df['Position'] == 'LCM') | (df['Position'] == 'CM') | (df['Position'] == 'RCM') | (df['Position'] == 'RM')):
        return 'MF'
    elif ((df['Position'] == 'LAM') | (df['Position'] == 'CAM') | (df['Position'] == 'RAM') | (df['Position'] == 'LW') | (df['Position'] == 'RW')):
        return 'AM'
    elif ((df['Position'] == 'RS') | (df['Position'] == 'ST') | (df['Position'] == 'LS') | (df['Position'] == 'CF') | (df['Position'] == 'LF') | (df['Position'] == 'RF')):
        return 'ST'
    else:
        return df.Position
    
#Turn Preferred Foot into a binary indicator variable
def right_footed(df):
    if (df['Preferred Foot'] == 'Right'):
        return 1
    else:
        return 0

#Create a copy of the index 
df1 = df.copy()
df1['Right_Foot'] = df1.apply(right_footed, axis=1)
df1['Simple_Position'] = df1.apply(simple_position,axis = 1)

#Individual relationships to value 
sns.jointplot(y='Value',x='Overall',data=df1,kind='scatter')
sns.jointplot(y='Wage',x='Overall',data=df1,kind='scatter')

sns.distplot(df['Wage'])

#Correlations 


print('Correlation between Age and value : {}' .format(df['Age'].corr(df['Value'])))
print('Correlation between Age and Wage : {}' .format(df['Wage'].corr(df['Wage'])))
print('Correlation between Overall and value : {}' .format(df['Overall'].corr(df['Value'])))
print('Correlation between Overall and Wage : {}' .format(df['Overall'].corr(df['Wage'])))
print('Correlation between Wage and Value : {}' .format(df['Wage'].corr(df['Value'])))

#Correlation matrix 
corr = df.corr()
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True
with sns.axes_style("white"):
    f, ax = plt.subplots(figsize=(15, 15))
    ax = sns.heatmap(corr,mask=mask,square=True,linewidths=.8,cmap="YlGnBu") 
    
    
plt.figure(figsize=(20,15))
sns.heatmap(df.corr(), annot=True, annot_kws={"size": 5})
sns.set_style('white')
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()

#List of remaining columns 
columns = list(df.columns) 


# Simple linear regression between some imoprtant values 
X = df[['Wage']]
y = df[['Value']]

# Splitting the dataset in test set and train set 
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)

#Linear Regression 
regressor = LinearRegression()
regressor.fit(X_train,y_train)

# Predicting the test set results 
y_pred = regressor.predict(X_train)

#Plotting and visualisation 
plt.scatter(X_train,y_train,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title('Salary vs Experience (training set)')
plt.xlabel('Years of Exp')
plt.ylabel('Salary')
plt.show()

#Plotting and visualisation 
plt.scatter(X_test,y_test,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title('Salary vs Experience (test set)')
plt.xlabel('Years of Exp')
plt.ylabel('Salary')
plt.show()



#Multiple linear regression 
X = df[['Overall','Wage','Skill Moves','International Reputation']].values
y = df[['Value']].values


# Splitting the dataset in test set and train set 
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)


#Multiple Linear Regression 
regressor = LinearRegression()
regressor.fit(X_train,y_train)

# Predicting the test set results 
y_pred = regressor.predict(X_train)

print('Training data r-squared:', regressor.score(X_train, y_train))
print('Test data r-squared:', regressor.score(X_test, y_test))


X_opt = X[:,[0, 1, 2, 3]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()

print(regressor_OLS.summary())


#Practice with value and wage 
sns.scatterplot(x=df['Wage'],y=df['Value'])


# Splitting the dataset in test set and train set 
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

#Linear Regression 
regressor = LinearRegression()
regressor.fit(X_train,y_train)

# Predicting the test set results 
y_pred = regressor.predict(X_train)

#Plotting and visualisation 
plt.scatter(X_train,y_train,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title('Wage vs Value(training set)')
plt.xlabel('Wage')
plt.ylabel('value')
plt.show()

print('Training data r-squared:', regressor.score(X_train, y_train))
print('Test data r-squared:', regressor.score(X_test, y_test))

print("predictions: %s" % regressor.predict(X_test))
print("accuracy: %.2f" % regressor.score(X_test, y_test))





