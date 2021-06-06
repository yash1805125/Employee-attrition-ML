#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import plotly.offline as py
import seaborn as sns
py.init_notebook_mode(connected=True)

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.linear_model import LogisticRegression


# In[6]:


# Data import
attrition = pd.read_csv('C:/Users/KIIT/Desktop/emp_att_f.csv')
attrition.head()
attritionnum = pd.read_csv('C:/Users/KIIT/Desktop/emp_att_f.csv')
attritionnum.head()


# In[7]:


# Dataframe definition
df = pd.DataFrame(data=attritionnum, columns = ['employee_number', 'age', 
        'business_travel_num', 'monthly_income', 	'department_num', 	'distance_home',
        'education', 'edu_field_num', 'environment_satisfaction', 'gender_num', 'job_involvement',
       'job_level',	'job_role_num', 'job_satisfaction', 'marital_status_num', 'num_comp_worked',
       'overtime_num', 'percent_salary_hike', 'performance_rating', 'relationship_satisfaction', 
       'stock_option_level',	'total_working_years', 'training_times_last_y', 'work_life_balance', 
       'years_at_company', 'years_in_current_role', 'years_since_last_promotion', 'years_with_curr_manager',
       'attrition_num'])
df.head()


# In[8]:


if df.empty:
    print('DataFrame is empty!')


# In[9]:


df.dtypes


# In[10]:


df.isnull().values.any()


# In[11]:


#Data Exploration
df.head()


# In[12]:


df.tail()


# In[13]:


df["attrition_num"].value_counts()


# In[14]:


df["attrition_num"].value_counts().plot(kind="bar", color=["Lime", "Red"]);


# In[15]:


pd.crosstab(df.attrition_num, df.job_satisfaction)


# In[16]:


pd.crosstab(df.attrition_num, df.job_satisfaction).plot(kind="bar", color=["green", "Red","blue","yellow"], figsize=(10,6));

plt.title("Job-satisfaction Vs. Attrition-number")
plt.xlabel("0.No Attrition,1.Attrition")
plt.ylabel("No. of persons")
plt.legend(["Level 1","Level 2","Level 3","Level 4",]);
plt.xticks(rotation=0);


# In[17]:


#Attrition & work travel frequency
pd.crosstab(attritionnum.business_travel, df.attrition_num).plot(kind="bar", color=["blue", "cyan"], figsize=(10,6));

plt.title("Work-travel-freq Vs. Attrition-number")
plt.xlabel(" ")
plt.ylabel("No. of persons")
plt.legend(["Attrition No","Attrition Yes"]);
plt.xticks(rotation=0);


# In[18]:


#Overtime Vs. Attrition
pd.crosstab(attritionnum.overtime, df.attrition_num).plot(kind="bar", color=["orange", "purple"], figsize=(10,6));

plt.title("Overtime Vs. Attrition-number")
plt.xlabel("No->Non overtime, Yes->Overtime")
plt.ylabel("No. of persons")
plt.legend(["Attrition No","Attrition Yes"]);
plt.xticks(rotation=0);


# In[19]:


print(attritionnum.business_travel)


# In[ ]:





# In[20]:


#Make a correlation matrix
df.corr()


# In[21]:


corr_matrix = df.corr()
fig,ax = plt.subplots(figsize=(22,18))
ax = sns.heatmap(corr_matrix,annot=True,
                linewidths=0.5,fmt=" .2f",cmap="YlGnBu");


# In[22]:


# Generating a random # between 1 and 0, if # <= 0.75 the observations goes to the group train
df['is_train'] = np.random.uniform(0, 1, len(df)) <= .75
df.head()


# In[23]:


# Delete employee number - not needed, could mess the method
del df['employee_number']
df.head()


# In[24]:


# Create two new dataframes, one with the training rows, one with the test rows
train, test = df[df['is_train']==True], df[df['is_train']==False]


# In[25]:


# Show the number of observations for the test and training dataframes
print('Number of observations in the training data:', len(train))
print('Number of observations in the test data:',len(test))


# In[26]:


# Create a list of the feature column's names - without 29th column attrition
features = df.columns[:27]

# Defining the variable to be predicted - the target
x_train = train
y_train = train['attrition_num']
train.head()
len(train)


# In[27]:


# Defining test
x_test = test
y_test = test['attrition_num']
test.head()


# In[28]:


# Create a random forest Classifier
clf = RandomForestClassifier(n_estimators=25, n_jobs=2, random_state=0)

# Train the Classifier to take the training features and learn how they relate to the training y (attrition_num)
clf.fit(train[features], y_train)

# Apply the Classifier we trained to the test data
clf.predict(test[features])

# View the predicted probabilities of the first 10 observations
clf.predict_proba(test[features])[0:10]


# In[29]:


clf.score(test[features],y_test)


# In[30]:


preds = clf.predict(test[features])


# In[31]:


# Create confusion matrix, WHICH REALLY IS CONFUSING AT FIRST
pd.crosstab(test['attrition_num'], preds, rownames=['Actual Attrition'], colnames=['Predicted Attrition'])


# In[32]:


log_reg = LogisticRegression(n_jobs=2, random_state=0)


# In[33]:


log_reg.fit(train[features], y_train)


# In[34]:


log_reg.fit(train[features], y_train)

# Apply the Classifier we trained to the test data
log_reg.predict(test[features])

# View the predicted probabilities of the first 10 observations
log_reg.predict_proba(test[features])[0:10]


# In[35]:


log_reg.score(test[features],y_test)


# In[ ]:





# In[36]:


#Create a Hyperparameter grid for Logistic Regression
log_reg_grid = {"C": np.logspace(-4,4,20),
               "solver": ["liblinear"]}

#Create Hyperparameter grid for Random Forest\
rf_grid = {"n_estimators": np.arange(100,1000,50),
          "max_depth": [None,3,5,10],
          "min_samples_split": np.arange(2,20,2),
          "min_samples_leaf": np.arange(1,20,2)}


# In[37]:


#Tune Logistic Regression

np.random.seed(42)

#Setup random Hyperparameter search for LogisticRegression
rs_log_reg = RandomizedSearchCV(LogisticRegression(),
                               param_distributions=log_reg_grid,
                               cv=5,
                               n_iter=20,
                               verbose=True)
#Fit random hyperparameter search model for LogisticReession
rs_log_reg.fit(train[features], y_train)


# In[38]:


rs_log_reg.best_params_


# In[39]:


rs_log_reg.score(test[features],y_test)


# In[40]:


# Apply the Classifier we trained to the test data
rs_log_reg.predict(test[features])

# View the predicted probabilities of the first 10 observations
rs_log_reg.predict_proba(test[features])[0:10]


# In[41]:


p = rs_log_reg.predict(test[features])


# In[42]:


# Create confusion matrix, WHICH REALLY IS CONFUSING AT FIRST
pd.crosstab(test['attrition_num'], p, rownames=['Actual Attrition'], colnames=['Predicted Attrition'])


# In[43]:


#Tuning RandomForestClassifier

np.random.seed(42)

#Setup Random Hyperparameter search for RandomForestClassifier
rs_rf = RandomizedSearchCV(RandomForestClassifier(),
                          param_distributions=rf_grid,
                          cv=5,
                          n_iter=20,
                          verbose=True)
#Fit the hyperparameter search model fo RF
rs_rf.fit(train[features], y_train)


# In[44]:


rs_rf.best_params_


# In[45]:


rs_rf.score(test[features],y_test)


# In[46]:


# Apply the Classifier we trained to the test data
rs_rf.predict(test[features])

# View the predicted probabilities of the first 10 observations
rs_rf.predict_proba(test[features])[0:10]


# In[47]:


ps = rs_rf.predict(test[features])


# In[48]:


# Create confusion matrix, WHICH REALLY IS CONFUSING AT FIRST
pd.crosstab(test['attrition_num'], ps, rownames=['Actual Attrition'], colnames=['Predicted Attrition'])


# In[49]:


#Different hyperparameters for Log Reg model
log_reg_grid={"C": np.logspace(-4,4,30),
             "solver": ["liblinear"]}
#Setup Grid hyperparameter search for Log Reg
gs_log_reg = GridSearchCV(LogisticRegression(),
                         param_grid=log_reg_grid,
                          cv=5,
                          verbose=True)
#Fit the grid hyperparameter
gs_log_reg.fit(train[features], y_train)


# In[50]:


gs_log_reg.best_params_


# In[51]:


gs_log_reg.score(test[features],y_test)


# In[52]:


# Apply the Classifier we trained to the test data
gs_log_reg.predict(test[features])

# View the predicted probabilities of the first 10 observations
gs_log_reg.predict_proba(test[features])[0:10]


# In[53]:


pz = gs_log_reg.predict(test[features])
# Create confusion matrix, WHICH REALLY IS CONFUSING AT FIRST
pd.crosstab(test['attrition_num'], pz, rownames=['Actual Attrition'], colnames=['Predicted Attrition'])


# In[56]:


# Scatter plot
y = ['age', 'business_travel_num', 'monthly_income', 'department_num', 'distance_home',
     'education', 'edu_field_num', 'environment_satisfation', 'gender_num',
     'job_involvement', 'job_level', 'job_role_num', 'job_satisfaction',
    'marital_status_num', 'num_comp_worked', 'overtime_num', 'percent_salary_hike',
    'performance_rating', 'relationship_satisfaction', 'stock_option_level', 
   'total_working_years', 'training_times_last_y', 'work_life_balance', 
    'years_at_company', 'years_in_current_role',     'years_since_last_promo',
    'years_with_curr_manager']
x = clf.feature_importances_
plt.figure(figsize=(15, 10))
plt.scatter(x, y, c=x, vmin=0, vmax=0.10, s=400, alpha = 0.75, cmap='plasma')
plt.colorbar()
#plt.ylabel('Attributes')
plt.xlabel('Feature Importance', fontsize=15)
plt.yticks([])
#plt.xticks(rotation=90)
plt.title('Feature Importance', fontsize=18)
labels = ['age', 'business travel', 'monthly income', 'department', 'distance from home', 'education level', 
          'education field', 'environment satisfation', 'gender', 'job involvement', 'job level', 'job role', 
         'job satisfaction', 'marital status', 'number companies worked', 'overtime', 'percent salary hike', 
         'performance rating', 'relationship satisfaction', 'stock option level', 'total working years', 
          'training times last year', 'work life balance', 'years at company', 'years in current role',
          'years since last promotion', 'years with current manager']
for label, x, y in zip(labels, x, y):
    plt.annotate(
        label,
        xy=(x, y), xytext=(-20, 20),
        textcoords='offset points', ha='left', va='top',
        bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.5),
        arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))
plt.show() 


# In[ ]:




