
# coding: utf-8

# # Data Analysis

# In[7]:


import math
import IPython
import time

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import preprocessing, decomposition, discriminant_analysis, tree

pd.options.display.max_columns = None


# In[3]:


raw_train_credit_application = pd.read_csv('data/application_train.csv')
raw_test_credit_application = pd.read_csv('data/application_test.csv')

raw_bureau = pd.read_csv('data/bureau.csv')
# raw_bureau_balance = pd.read_csv('data/bureau_balance.csv')
# raw_credit_card_balance = pd.read_csv('data/credit_card_balance.csv')
# raw_installments_payments = pd.read_csv('data/installments_payments.csv')
# raw_pos_cash_balance = pd.read_csv('data/pos_cash_balance.csv')
# raw_previous_application = pd.read_csv('data/previous_application.csv')


# In[4]:


all_features = list(set(raw_train_credit_application.columns.values.tolist()) - set(['SK_ID_CURR', 'TARGET']))

non_features = ['SK_ID_CURR', 'TARGET']

categorical_features = ['NAME_CONTRACT_TYPE', 'NAME_TYPE_SUITE', 'NAME_INCOME_TYPE', 
                        'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 
                        'OCCUPATION_TYPE', 'WEEKDAY_APPR_PROCESS_START', 'ORGANIZATION_TYPE', 
                        'FONDKAPREMONT_MODE', 'HOUSETYPE_MODE', 'WALLSMATERIAL_MODE', 
                        'EMERGENCYSTATE_MODE',
                       ]

flag_features = ['CODE_GENDER', 
                 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'FLAG_MOBIL', 'FLAG_EMP_PHONE', 
                 'FLAG_WORK_PHONE', 'FLAG_CONT_MOBILE', 'FLAG_PHONE', 'FLAG_EMAIL', 
                 'FLAG_DOCUMENT_2', 'FLAG_DOCUMENT_3', 'FLAG_DOCUMENT_4', 'FLAG_DOCUMENT_5', 
                 'FLAG_DOCUMENT_6', 'FLAG_DOCUMENT_7', 'FLAG_DOCUMENT_8', 'FLAG_DOCUMENT_9', 
                 'FLAG_DOCUMENT_10', 'FLAG_DOCUMENT_11', 'FLAG_DOCUMENT_12', 'FLAG_DOCUMENT_13', 
                 'FLAG_DOCUMENT_14', 'FLAG_DOCUMENT_15', 'FLAG_DOCUMENT_16', 'FLAG_DOCUMENT_17', 
                 'FLAG_DOCUMENT_18', 'FLAG_DOCUMENT_19', 'FLAG_DOCUMENT_20', 'FLAG_DOCUMENT_21',
                 'REG_REGION_NOT_LIVE_REGIONREG_REGION_NOT_LIVE_REGION', 'REG_REGION_NOT_WORK_REGION', 
                 'LIVE_REGION_NOT_WORK_REGION', 'REG_CITY_NOT_LIVE_CITY', 'REG_CITY_NOT_WORK_CITY', 
                 'LIVE_CITY_NOT_WORK_CITY',
                ]

numerical_features = list(set(all_features) - set(non_features) - set(categorical_features) - set(flag_features))


# # Feature Engineering I

# In[5]:


train_SK_ID_CURR = raw_train_credit_application['SK_ID_CURR'].unique()
test_SK_ID_CURR = raw_test_credit_application['SK_ID_CURR'].unique()


# In[ ]:


add_columns = [
    'SK_ID_CURR',
    
    'BUR_credit_active',
    'BUR_credit_closed',
    'BUR_credit_sold',
    'BUR_credit_bad_debt',
]

add_train_credit_application = pd.DataFrame(columns=add_columns)
index = 0
for SK_ID_CURR in train_SK_ID_CURR:
    if index % 1000 == 0:
        print(index, time.time())
    bureau_SK_ID_CURR = raw_bureau.loc[raw_bureau['SK_ID_CURR'] == SK_ID_CURR]
    credit_active_value_counts = bureau_SK_ID_CURR['CREDIT_ACTIVE'].value_counts()
        
    BUR_credit_active = credit_active_value_counts['Active'] if 'Active' in credit_active_value_counts else 0
    BUR_credit_closed = credit_active_value_counts['Closed'] if 'Closed' in credit_active_value_counts else 0
    BUR_credit_sold = credit_active_value_counts['Sold'] if 'Sold' in credit_active_value_counts else 0
    BUR_credit_bad_debt = credit_active_value_counts['Bad Debt'] if 'Bad Debt' in credit_active_value_counts else 0
    
    add_train_credit_application.loc[index] = [SK_ID_CURR, 
                                               BUR_credit_active,
                                               BUR_credit_closed,
                                               BUR_credit_sold,
                                               BUR_credit_bad_debt,]
    
    index = index + 1


# In[ ]:



add_test_credit_application = pd.DataFrame(columns=add_columns)
index = 0
for SK_ID_CURR in test_SK_ID_CURR:
    if index % 1000 == 0:
        print(index)
    bureau_SK_ID_CURR = raw_bureau.loc[raw_bureau['SK_ID_CURR'] == SK_ID_CURR]
    credit_active_value_counts = bureau_SK_ID_CURR['CREDIT_ACTIVE'].value_counts()
        
    BUR_credit_active = credit_active_value_counts['Active'] if 'Active' in credit_active_value_counts else 0
    BUR_credit_closed = credit_active_value_counts['Closed'] if 'Closed' in credit_active_value_counts else 0
    BUR_credit_sold = credit_active_value_counts['Sold'] if 'Sold' in credit_active_value_counts else 0
    BUR_credit_bad_debt = credit_active_value_counts['Bad Debt'] if 'Bad Debt' in credit_active_value_counts else 0
    
    add_test_credit_application.loc[index] = [SK_ID_CURR, 
                                              BUR_credit_active,
                                              BUR_credit_closed,
                                              BUR_credit_sold,
                                              BUR_credit_bad_debt,]
    
    index = index + 1


# In[ ]:


train_credit_application_1 = pd.merge(left=raw_train_credit_application, right=add_train_credit_application, left_on='SK_ID_CURR', right_on='SK_ID_CURR', how='outer')
test_credit_application_1 = pd.merge(left=raw_test_credit_application, right=add_test_credit_application, left_on='SK_ID_CURR', right_on='SK_ID_CURR', how='outer')


# In[ ]:


train_credit_application_1.to_pickle('train_credit_application_1')
test_credit_application_1.to_pickle('test_credit_application_1')
