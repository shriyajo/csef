#!/usr/bin/env python
# coding: utf-8

# In[55]:


import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


# In[56]:


peptides=pd.read_csv(r"C:\Users\jshri\OneDrive\Desktop\train_peptides.csv")
proteins=pd.read_csv(r"C:\Users\jshri\OneDrive\Desktop\train_proteins.csv")
clinical_data=pd.read_csv(r"C:\Users\jshri\OneDrive\Desktop\train_clinical_data.csv")
supp_clinical=pd.read_csv(r"C:\Users\jshri\OneDrive\Desktop\supplemental_clinical_data.csv") 


# In[57]:


all(proteins[['visit_id', 'UniProt']].value_counts() == 1)
df_p = peptides.merge(proteins[['visit_id', 'UniProt', 'NPX']], on=['visit_id','UniProt'], how='left')
df_p.head()
#I've rewritten visit_id for supplemental clinical data since it seems that its visit_id was different from convention in other files
supp_clinical['visit_id'] = supp_clinical['patient_id'].astype(str) + "_"+ supp_clinical['visit_month'].astype(str)

#Here we combine both main and supplemental clinical data into a single dataframe
df_cd = pd.concat([clinical_data, supp_clinical], ignore_index=True)
display(df_cd.info())
df_cd.melt(id_vars=['visit_id', 'patient_id', 'visit_month', 'upd23b_clinical_state_on_medication'], 
                   var_name='updrs', value_name='rating')
df_all = df_p.merge(df_cd[['visit_id','updrs_1','updrs_2','updrs_3','updrs_4','upd23b_clinical_state_on_medication']], on=['visit_id'], how='left')
df_all.info()
df_all['Peptide'].str.extract(r"(.\(.*?\))", expand=False).value_counts()
df_all.head()


# In[58]:


options = ['SPQGLGAFTPVVR', 'ALEYIENLR', 'GMADQDGLKPTIDKPSEDSPPLEMLGPR', 'ESLQQMAEVTR', 'LEPGQQEEYYR', 'AYQGVAAPFPK', 'LQDLYSIVR',
          'SSGLVSNAPGVQIR', 'QALNTDYLDSDYQR', 'LVFFAEDVGSNK']
df_all = df_all[df_all['Peptide'].isin(options)]
df_all.head()


# In[59]:


import pandas as pd
df = df_all
# Create a new column that checks whether all UPDRS values are 0
df['all_UPDRS_0'] = ((df['updrs_1'] == 0) & (df['updrs_2'] == 0) &
                     (df['updrs_3'] == 0) & (df['updrs_4'] == 0))

# Group the control patients based on whether all UPDRS values are 0
control_groups = df[
                    (df['all_UPDRS_0'] == True)]

# Print the resulting groups
control_groups.head(40)
num_patients = control_groups['visit_month'].nunique()
print(num_patients)


# In[60]:


df_all = df_all.drop(['updrs_4', 'upd23b_clinical_state_on_medication'], axis=1)


# In[61]:


updrs_cutoffs = {'updrs_1': 1.5, 'updrs_2': 5, 'updrs_3': 13}


# In[62]:


# Create a new column indicating whether an individual is less likely to have Parkinson's disease
df['no_pd'] = (df_all['updrs_1'] <= updrs_cutoffs['updrs_1']) & (df_all['updrs_2'] <= updrs_cutoffs['updrs_2']) &  (df_all['updrs_3'] <= updrs_cutoffs['updrs_3'])


# In[63]:


# Separate no_pd == True (control) and no_pd == False (PD) into different dataframes
control_df = df[df['no_pd'] == True]
pd_df = df[df['no_pd'] == False]

# Drop updrs_4 and all_UPDRS_0 columns
control_df = control_df.drop(['updrs_4', 'all_UPDRS_0'], axis=1)
pd_df = pd_df.drop(['updrs_4', 'all_UPDRS_0'], axis=1)


# In[64]:


control_df = control_df.drop(['visit_id', 'visit_month', 'UniProt', 'NPX','updrs_1', 'updrs_2', 'updrs_3', 'upd23b_clinical_state_on_medication' ,'no_pd'  ], axis=1) 
control_df.head()


# In[65]:


pd_df = pd_df.drop(['visit_id', 'visit_month', 'UniProt', 'NPX','updrs_1', 'updrs_2', 'updrs_3', 'upd23b_clinical_state_on_medication' ,'no_pd'  ], axis=1)
pd_df.head()


# In[66]:


import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, ensemble
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


# In[67]:


from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[68]:


import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import ElasticNet
from sklearn.datasets import make_regression

# Define a list of hyperparameters to iterate over
learning_rates = [0.1, 0.2, 0.5]
max_depths = [3, 4, 5]

# Initialize an empty dictionary to store accuracy scores
accuracy_scores = {}

# Separate no_pd == True (control) and no_pd == False (PD) into different dataframes
control_df = df[df['no_pd'] == True]
pd_df = df[df['no_pd'] == False]
control_df.fillna(0, inplace=True)
pd_df.fillna(0, inplace=True)



# Drop updrs_4 and all_UPDRS_0 columns
control_df = control_df.drop(['updrs_4', 'all_UPDRS_0'], axis=1)
pd_df = pd_df.drop(['updrs_4', 'all_UPDRS_0'], axis=1) 

# Split the data into training and testing sets

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    pd.concat([control_df, pd_df])[['PeptideAbundance']],
    pd.concat([control_df, pd_df])['no_pd'].astype(int),
    test_size=0.2, 
    random_state=7)

for lr in learning_rates:
    for depth in max_depths:
        model = XGBClassifier(scale_pos_weight=1,
                              learning_rate=lr,  
                              colsample_bytree = 0.9,
                              subsample = 0.3,
                              objective='reg:logistic', 
                              n_estimators=1000, 
                              reg_alpha = 0.3,
                              max_depth=depth, 
                              gamma=1)
        model.fit(X_train,y_train)
        accuracy = model.score(X_test, y_test)
        accuracy_scores[(lr, depth)] = accuracy

# Test the model on the testing set
accuracy = model.score(X_test, y_test)
print(f"Accuracy: {accuracy}")


# In[69]:


accuracy = model.score(X_test, y_test)
accuracy_scores[(lr, depth)] = accuracy


# In[77]:


plt.figure(figsize=(8,6))
plt.title("Accuracy Scores for Different Hyperparameters")
plt.xlabel("Learning Rate")
plt.ylabel("Max Depth")
for lr in learning_rates:
    for depth in max_depths:
        score = accuracy_scores[(lr, depth)]
        plt.text(lr, depth, "{:.3f}".format(score), ha='center', va='center')
plt.imshow([[accuracy_scores[(lr, depth)] for depth in max_depths] for lr in learning_rates],
           cmap='Blues', interpolation='nearest', origin='lower')
plt.xticks(range(len(learning_rates)), learning_rates)
plt.yticks(range(len(max_depths)), max_depths)
plt.colorbar()
plt.show()


# In[78]:


from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import plot_roc_curve

# Predict classes for the testing set
y_pred = model.predict(X_test)

# Create the classification residual plot
fig, ax = plt.subplots(figsize=(8, 8))
plot_confusion_matrix(model, X_test, y_test, ax=ax, cmap='Blues')
ax.set_title("Classification Residual Plot")
plot_roc_curve(model, X_test, y_test, ax=ax)
plt.show()


# In[ ]:




