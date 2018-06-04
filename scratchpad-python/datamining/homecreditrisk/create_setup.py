from datamining.homecreditrisk.library.Setup import Setup

import math
import os

import numpy as np
import pandas as pd

from sklearn import preprocessing
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import optimizers
import keras.backend as K


def clean_data(df, train_df=None):
    new_df = df.copy()

    if train_df is None:
        for col in new_df.columns.values:
            if new_df[col].dtype == np.float or new_df[col].dtype == np.int:
                new_df[col] = new_df[col].fillna(new_df[col].median())
            elif new_df[col].dtype == np.object:
                new_df[col] = new_df[col].fillna(new_df[col].value_counts().idxmax())
    else:
        for col in new_df.columns.values:
            if new_df[col].dtype == np.float64 or new_df[col].dtype == np.int64:
                new_df[col] = new_df[col].fillna(train_df[col].median())
            elif new_df[col].dtype == np.object:
                new_df[col] = new_df[col].fillna(train_df[col].value_counts().idxmax())

    new_df['SK_ID_CURR'] = new_df['SK_ID_CURR']
    if 'TARGET' in new_df:
        new_df['TARGET'] = new_df['TARGET']
    new_df['NAME_CONTRACT_TYPE'] = new_df['NAME_CONTRACT_TYPE']
    new_df['CODE_GENDER'] = new_df['CODE_GENDER'].apply(lambda val: 'F' if val == 'XNA' else val)
    new_df['FLAG_OWN_CAR'] = new_df['FLAG_OWN_CAR'].apply(lambda val: '0' if val == 'N' else 1)
    new_df['FLAG_OWN_REALTY'] = new_df['FLAG_OWN_REALTY'].apply(lambda val: '0' if val == 'N' else 1)
    new_df['CNT_CHILDREN'] = new_df['CNT_CHILDREN'].apply(lambda val: 3 if val > 3 else val)
    new_df['AMT_INCOME_TOTAL'] = new_df['AMT_INCOME_TOTAL'].apply(lambda val: 500000 if val > 500000 else val)
    new_df['AMT_CREDIT'] = new_df['AMT_CREDIT'].apply(lambda val: 1800000 if val > 1800000 else val)
    new_df['AMT_ANNUITY'] = new_df['AMT_ANNUITY'].apply(lambda val: 100000 if val > 100000 else val)
    new_df['AMT_GOODS_PRICE'] = new_df['AMT_GOODS_PRICE'].apply(lambda val: 2500000 if val > 2500000 else val)
    new_df['NAME_TYPE_SUITE'] = new_df['NAME_TYPE_SUITE']
    new_df['NAME_INCOME_TYPE'] = new_df['NAME_INCOME_TYPE']
    new_df['NAME_EDUCATION_TYPE'] = new_df['NAME_EDUCATION_TYPE']
    new_df['NAME_FAMILY_STATUS'] = new_df['NAME_FAMILY_STATUS'].apply(
        lambda val: 'Married' if val == 'Unknown' else val)
    new_df['NAME_HOUSING_TYPE'] = new_df['NAME_HOUSING_TYPE']
    new_df['REGION_POPULATION_RELATIVE'] = new_df['REGION_POPULATION_RELATIVE']
    new_df['DAYS_BIRTH'] = new_df['DAYS_BIRTH']
    new_df['DAYS_EMPLOYED'] = new_df['DAYS_EMPLOYED'].apply(lambda val: 0 if val > 0 else val)
    new_df['DAYS_REGISTRATION'] = new_df['DAYS_REGISTRATION'].apply(lambda val: -18000 if val < -18000 else val)
    new_df['DAYS_ID_PUBLISH'] = new_df['DAYS_ID_PUBLISH'].apply(lambda val: -6300 if val < -6300 else val)
    new_df['OWN_CAR_AGE'] = new_df['OWN_CAR_AGE'].apply(lambda val: 65 if val > 65 else val)
    new_df['FLAG_MOBIL'] = new_df['FLAG_MOBIL']
    new_df['FLAG_EMP_PHONE'] = new_df['FLAG_EMP_PHONE']
    new_df['FLAG_WORK_PHONE'] = new_df['FLAG_WORK_PHONE']
    new_df['FLAG_CONT_MOBILE'] = new_df['FLAG_CONT_MOBILE']
    new_df['FLAG_PHONE'] = new_df['FLAG_PHONE']
    new_df['FLAG_EMAIL'] = new_df['FLAG_EMAIL']
    new_df['OCCUPATION_TYPE'] = new_df['OCCUPATION_TYPE']
    new_df['CNT_FAM_MEMBERS'] = new_df['CNT_FAM_MEMBERS']
    new_df['REGION_RATING_CLIENT'] = new_df['REGION_RATING_CLIENT']
    new_df['REGION_RATING_CLIENT_W_CITY'] = new_df['REGION_RATING_CLIENT_W_CITY']
    new_df['WEEKDAY_APPR_PROCESS_START'] = new_df['WEEKDAY_APPR_PROCESS_START']
    new_df['HOUR_APPR_PROCESS_START'] = new_df['HOUR_APPR_PROCESS_START']
    new_df['REG_REGION_NOT_LIVE_REGION'] = new_df['REG_REGION_NOT_LIVE_REGION']
    new_df['REG_REGION_NOT_WORK_REGION'] = new_df['REG_REGION_NOT_WORK_REGION']
    new_df['LIVE_REGION_NOT_WORK_REGION'] = new_df['LIVE_REGION_NOT_WORK_REGION']
    new_df['REG_CITY_NOT_LIVE_CITY'] = new_df['REG_CITY_NOT_LIVE_CITY']
    new_df['REG_CITY_NOT_WORK_CITY'] = new_df['REG_CITY_NOT_WORK_CITY']
    new_df['LIVE_CITY_NOT_WORK_CITY'] = new_df['LIVE_CITY_NOT_WORK_CITY']
    new_df['ORGANIZATION_TYPE'] = new_df['ORGANIZATION_TYPE'].apply(lambda val: 'Unknown' if val == 'XNA' else val)
    new_df['EXT_SOURCE_1'] = new_df['EXT_SOURCE_1']
    new_df['EXT_SOURCE_2'] = new_df['EXT_SOURCE_2']
    new_df['EXT_SOURCE_3'] = new_df['EXT_SOURCE_3']
    new_df['APARTMENTS_AVG'] = new_df['APARTMENTS_AVG']
    new_df['BASEMENTAREA_AVG'] = new_df['BASEMENTAREA_AVG'].apply(lambda val: 0.5 if val > 0.5 else val)
    new_df['YEARS_BEGINEXPLUATATION_AVG'] = new_df['YEARS_BEGINEXPLUATATION_AVG'].apply(
        lambda val: 0.9 if val < 0.9 else val)
    new_df['YEARS_BUILD_AVG'] = new_df['YEARS_BUILD_AVG']
    new_df['COMMONAREA_AVG'] = new_df['COMMONAREA_AVG'].apply(lambda val: 0.4 if val > 0.4 else val)
    new_df['ELEVATORS_AVG'] = new_df['ELEVATORS_AVG'].apply(lambda val: 0.4 if val > 0.4 else val)
    new_df['ENTRANCES_AVG'] = new_df['ENTRANCES_AVG'].apply(lambda val: 0.5 if val > 0.5 else val)
    new_df['FLOORSMAX_AVG'] = new_df['FLOORSMAX_AVG']
    new_df['FLOORSMIN_AVG'] = new_df['FLOORSMIN_AVG']
    new_df['LANDAREA_AVG'] = new_df['LANDAREA_AVG'].apply(lambda val: 0.4 if val > 0.4 else val)
    new_df['LIVINGAPARTMENTS_AVG'] = new_df['LIVINGAPARTMENTS_AVG'].apply(lambda val: 0.6 if val > 0.6 else val)
    new_df['LIVINGAREA_AVG'] = new_df['LIVINGAREA_AVG'].apply(lambda val: 0.75 if val > 0.75 else val)
    new_df['NONLIVINGAPARTMENTS_AVG'] = new_df['NONLIVINGAPARTMENTS_AVG'].apply(
        lambda val: 0.075 if val > 0.075 else val)
    new_df['NONLIVINGAREA_AVG'] = new_df['NONLIVINGAREA_AVG'].apply(lambda val: 0.3 if val > 0.3 else val)
    new_df['APARTMENTS_MODE'] = new_df['APARTMENTS_MODE'].apply(lambda val: 0.6 if val > 0.6 else val)
    new_df['BASEMENTAREA_MODE'] = new_df['BASEMENTAREA_MODE'].apply(lambda val: 0.4 if val > 0.4 else val)
    new_df['YEARS_BEGINEXPLUATATION_MODE'] = new_df['YEARS_BEGINEXPLUATATION_MODE'].apply(
        lambda val: 0.95 if val < 0.95 else val)
    new_df['YEARS_BUILD_MODE'] = new_df['YEARS_BUILD_MODE'].apply(lambda val: 0.3 if val < 0.3 else val)
    new_df['COMMONAREA_MODE'] = new_df['COMMONAREA_MODE'].apply(lambda val: 0.25 if val > 0.25 else val)
    new_df['ELEVATORS_MODE'] = new_df['ELEVATORS_MODE'].apply(lambda val: 0.4 if val > 0.4 else val)
    new_df['ENTRANCES_MODE'] = new_df['ENTRANCES_MODE'].apply(lambda val: 0.4 if val > 0.4 else val)
    new_df['FLOORSMAX_MODE'] = new_df['FLOORSMAX_MODE'].apply(lambda val: 0.6 if val > 0.6 else val)
    new_df['FLOORSMIN_MODE'] = new_df['FLOORSMIN_MODE'].apply(lambda val: 0.6 if val > 0.6 else val)
    new_df['LANDAREA_MODE'] = new_df['LANDAREA_MODE'].apply(lambda val: 0.4 if val > 0.4 else val)
    new_df['LIVINGAPARTMENTS_MODE'] = new_df['LIVINGAPARTMENTS_MODE'].apply(lambda val: 0.6 if val > 0.6 else val)
    new_df['LIVINGAREA_MODE'] = new_df['LIVINGAREA_MODE'].apply(lambda val: 0.75 if val > 0.75 else val)
    new_df['NONLIVINGAPARTMENTS_MODE'] = new_df['NONLIVINGAPARTMENTS_MODE'].apply(
        lambda val: 0.075 if val > 0.075 else val)
    new_df['NONLIVINGAREA_MODE'] = new_df['NONLIVINGAREA_MODE'].apply(lambda val: 0.2 if val > 0.2 else val)
    new_df['APARTMENTS_MEDI'] = new_df['APARTMENTS_MEDI'].apply(lambda val: 0.6 if val > 0.6 else val)
    new_df['BASEMENTAREA_MEDI'] = new_df['BASEMENTAREA_MEDI'].apply(lambda val: 0.4 if val > 0.4 else val)
    new_df['YEARS_BEGINEXPLUATATION_MEDI'] = new_df['YEARS_BEGINEXPLUATATION_MEDI'].apply(
        lambda val: 0.95 if val < 0.95 else val)
    new_df['YEARS_BUILD_MEDI'] = new_df['YEARS_BUILD_MEDI'].apply(lambda val: 0.3 if val < 0.3 else val)
    new_df['COMMONAREA_MEDI'] = new_df['COMMONAREA_MEDI'].apply(lambda val: 0.25 if val > 0.25 else val)
    new_df['ELEVATORS_MEDI'] = new_df['ELEVATORS_MEDI'].apply(lambda val: 0.4 if val > 0.4 else val)
    new_df['ENTRANCES_MEDI'] = new_df['ENTRANCES_MEDI'].apply(lambda val: 0.4 if val > 0.4 else val)
    new_df['FLOORSMAX_MEDI'] = new_df['FLOORSMAX_MEDI'].apply(lambda val: 0.6 if val > 0.6 else val)
    new_df['FLOORSMIN_MEDI'] = new_df['FLOORSMIN_MEDI'].apply(lambda val: 0.6 if val > 0.6 else val)
    new_df['LANDAREA_MEDI'] = new_df['LANDAREA_MEDI'].apply(lambda val: 0.4 if val > 0.4 else val)
    new_df['LIVINGAPARTMENTS_MEDI'] = new_df['LIVINGAPARTMENTS_MEDI'].apply(lambda val: 0.6 if val > 0.6 else val)
    new_df['LIVINGAREA_MEDI'] = new_df['LIVINGAREA_MEDI'].apply(lambda val: 0.75 if val > 0.75 else val)
    new_df['NONLIVINGAPARTMENTS_MEDI'] = new_df['NONLIVINGAPARTMENTS_MEDI'].apply(
        lambda val: 0.075 if val > 0.075 else val)
    new_df['NONLIVINGAREA_MEDI'] = new_df['NONLIVINGAREA_MEDI'].apply(lambda val: 0.2 if val > 0.2 else val)
    new_df['FONDKAPREMONT_MODE'] = new_df['FONDKAPREMONT_MODE']
    new_df['HOUSETYPE_MODE'] = new_df['HOUSETYPE_MODE']
    new_df['TOTALAREA_MODE'] = new_df['TOTALAREA_MODE'].apply(lambda val: 0.60 if val > 0.60 else val)
    new_df['WALLSMATERIAL_MODE'] = new_df['WALLSMATERIAL_MODE']
    new_df['EMERGENCYSTATE_MODE'] = new_df['EMERGENCYSTATE_MODE']
    new_df['OBS_30_CNT_SOCIAL_CIRCLE'] = new_df['OBS_30_CNT_SOCIAL_CIRCLE'].apply(lambda val: 25 if val > 25 else val)
    new_df['DEF_30_CNT_SOCIAL_CIRCLE'] = new_df['DEF_30_CNT_SOCIAL_CIRCLE'].apply(lambda val: 5 if val > 5 else val)
    new_df['OBS_60_CNT_SOCIAL_CIRCLE'] = new_df['OBS_60_CNT_SOCIAL_CIRCLE'].apply(lambda val: 15 if val > 15 else val)
    new_df['DEF_60_CNT_SOCIAL_CIRCLE'] = new_df['DEF_60_CNT_SOCIAL_CIRCLE'].apply(lambda val: 3 if val > 3 else val)
    new_df['DAYS_LAST_PHONE_CHANGE'] = new_df['DAYS_LAST_PHONE_CHANGE'].apply(lambda val: -3200 if val < -3200 else val)
    new_df['FLAG_DOCUMENT_2'] = new_df['FLAG_DOCUMENT_2']
    new_df['FLAG_DOCUMENT_3'] = new_df['FLAG_DOCUMENT_3']
    new_df['FLAG_DOCUMENT_4'] = new_df['FLAG_DOCUMENT_4']
    new_df['FLAG_DOCUMENT_5'] = new_df['FLAG_DOCUMENT_5']
    new_df['FLAG_DOCUMENT_6'] = new_df['FLAG_DOCUMENT_6']
    new_df['FLAG_DOCUMENT_7'] = new_df['FLAG_DOCUMENT_7']
    new_df['FLAG_DOCUMENT_8'] = new_df['FLAG_DOCUMENT_8']
    new_df['FLAG_DOCUMENT_9'] = new_df['FLAG_DOCUMENT_9']
    new_df['FLAG_DOCUMENT_10'] = new_df['FLAG_DOCUMENT_10']
    new_df['FLAG_DOCUMENT_11'] = new_df['FLAG_DOCUMENT_11']
    new_df['FLAG_DOCUMENT_12'] = new_df['FLAG_DOCUMENT_12']
    new_df['FLAG_DOCUMENT_13'] = new_df['FLAG_DOCUMENT_13']
    new_df['FLAG_DOCUMENT_14'] = new_df['FLAG_DOCUMENT_14']
    new_df['FLAG_DOCUMENT_15'] = new_df['FLAG_DOCUMENT_15']
    new_df['FLAG_DOCUMENT_16'] = new_df['FLAG_DOCUMENT_16']
    new_df['FLAG_DOCUMENT_17'] = new_df['FLAG_DOCUMENT_17']
    new_df['FLAG_DOCUMENT_18'] = new_df['FLAG_DOCUMENT_18']
    new_df['FLAG_DOCUMENT_19'] = new_df['FLAG_DOCUMENT_19']
    new_df['FLAG_DOCUMENT_20'] = new_df['FLAG_DOCUMENT_20']
    new_df['FLAG_DOCUMENT_21'] = new_df['FLAG_DOCUMENT_21']
    new_df['AMT_REQ_CREDIT_BUREAU_HOUR'] = new_df['AMT_REQ_CREDIT_BUREAU_HOUR'].apply(
        lambda val: 2.0 if val > 2.0 else val)
    new_df['AMT_REQ_CREDIT_BUREAU_DAY'] = new_df['AMT_REQ_CREDIT_BUREAU_DAY'].apply(
        lambda val: 4.0 if val > 4.0 else val)
    new_df['AMT_REQ_CREDIT_BUREAU_WEEK'] = new_df['AMT_REQ_CREDIT_BUREAU_WEEK'].apply(
        lambda val: 3.0 if val > 3.0 else val)
    new_df['AMT_REQ_CREDIT_BUREAU_MON'] = new_df['AMT_REQ_CREDIT_BUREAU_MON'].apply(
        lambda val: 17.0 if val > 17.0 else val)
    new_df['AMT_REQ_CREDIT_BUREAU_QRT'] = new_df['AMT_REQ_CREDIT_BUREAU_QRT'].apply(
        lambda val: 6.0 if val > 6.0 else val)
    new_df['AMT_REQ_CREDIT_BUREAU_YEAR'] = new_df['AMT_REQ_CREDIT_BUREAU_YEAR'].apply(
        lambda val: 14.0 if val > 14.0 else val)

    return new_df


def select_feature(df):
    columns = ['CODE_GENDER', 'CNT_CHILDREN',
               'AMT_GOODS_PRICE', 'REGION_POPULATION_RELATIVE', 'DAYS_EMPLOYED',
               'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3',
               'FLOORSMAX_AVG', 'FLOORSMAX_MODE', 'FLOORSMAX_MEDI'
               ]

    new_df = pd.DataFrame()

    new_df['SK_ID_CURR'] = df['SK_ID_CURR']
    if 'TARGET' in df:
        new_df['TARGET'] = df['TARGET']

    for col in columns:
        if col in df:
            new_df[col] = df[col]

    return new_df


def engineer_feature(df):
    categorical_columns = ['NAME_CONTRACT_TYPE', 'CODE_GENDER', 'FLAG_OWN_CAR',
                           'FLAG_OWN_REALTY', 'NAME_TYPE_SUITE', 'NAME_INCOME_TYPE',
                           'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE',
                           'OCCUPATION_TYPE', 'WEEKDAY_APPR_PROCESS_START', 'ORGANIZATION_TYPE',
                           'FONDKAPREMONT_MODE', 'HOUSETYPE_MODE', 'WALLSMATERIAL_MODE',
                           'EMERGENCYSTATE_MODE']
    for col in categorical_columns:
        if col in df:
            df = pd.get_dummies(df, columns=[col])

    return df


def split_ids_data_targets(df, dataframe=False):
    new_df = df.copy()

    new_df_ids = new_df['SK_ID_CURR']
    if 'TARGET' in new_df:
        new_df_targets = new_df['TARGET']
        new_df_data = new_df.drop(columns=['SK_ID_CURR', 'TARGET'])
    else:
        new_df_targets = None
        new_df_data = new_df.drop(columns=['SK_ID_CURR'])

    if not dataframe:
        new_df_ids = new_df_ids.values
        new_df_data = new_df_data.values
        if new_df_targets is not None:
            new_df_targets = new_df_targets.values

    return new_df_ids, new_df_data, new_df_targets


print('Loading Data ...')
raw_train_credit_application = pd.read_csv('data/application_train.csv')
raw_test_credit_application = pd.read_csv('data/application_test.csv')

print('Transforming Data ...')
train_credit_application = clean_data(df=raw_train_credit_application)
test_credit_application = clean_data(df=raw_test_credit_application, train_df=raw_train_credit_application)
train_credit_application = select_feature(train_credit_application)
test_credit_application = select_feature(test_credit_application)
train_credit_application = engineer_feature(train_credit_application)
test_credit_application = engineer_feature(test_credit_application)

train_credit_application = train_credit_application.reindex(np.random.permutation(train_credit_application.index))
total_count = train_credit_application['SK_ID_CURR'].count()
training_count = math.ceil(total_count * 0.75)
validation_count = math.floor(total_count * 0.25)

training_credit_application = train_credit_application.head(training_count)
validation_credit_application = train_credit_application.tail(validation_count)
testing_credit_application = test_credit_application

training_ids, training_data, training_targets = split_ids_data_targets(training_credit_application)
validation_ids, validation_data, validation_targets = split_ids_data_targets(validation_credit_application)
testing_ids, testing_data, testing_targets = split_ids_data_targets(testing_credit_application)
training_targets_onehot = (preprocessing.OneHotEncoder().fit_transform(training_targets.reshape(-1, 1))).toarray()
validation_targets_onehot = (preprocessing.OneHotEncoder().fit_transform(validation_targets.reshape(-1, 1))).toarray()

print('Preparing Settings to create ...')
def_epochs = 50
def_bs = 1000
def_act = 'sigmoid'
def_last_act = 'sigmoid'
def_dropout = False
def_dropout_rate = 0.25
def_loss = 'binary_crossentropy'
def_add_name = None

dnn_configs = []

default_dnn_configs = [
]

potential_configs = [
    # Adam
    {'optimizer': optimizers.Adam(), 'bs': def_bs, 'epochs': def_epochs, 'loss': def_loss,
     'act': def_act, 'last_act': def_last_act, 'dropout': def_dropout, 'dropout_rate': def_dropout_rate,
     'add_name': def_add_name},

    # SGD
    {'optimizer': optimizers.SGD(momentum=0.1), 'bs': def_bs, 'epochs': def_epochs * 10, 'loss': def_loss,
     'act': def_act, 'last_act': def_last_act, 'dropout': def_dropout, 'dropout_rate': def_dropout_rate,
     'add_name': def_add_name},

    # Adagrad
    {'optimizer': optimizers.Adagrad(), 'bs': def_bs, 'epochs': def_epochs, 'loss': def_loss,
     'act': def_act, 'last_act': def_last_act, 'dropout': def_dropout, 'dropout_rate': def_dropout_rate,
     'add_name': def_add_name},

    # RMSprop
    {'optimizer': optimizers.RMSprop(), 'bs': def_bs, 'epochs': def_epochs, 'loss': def_loss,
     'act': def_act, 'last_act': def_last_act, 'dropout': def_dropout, 'dropout_rate': def_dropout_rate,
     'add_name': def_add_name},

    # Adamax
    {'optimizer': optimizers.Adamax(), 'bs': def_bs, 'epochs': def_epochs, 'loss': def_loss,
     'act': def_act, 'last_act': def_last_act, 'dropout': def_dropout, 'dropout_rate': def_dropout_rate,
     'add_name': def_add_name},

    # Nadam
    {'optimizer': optimizers.Nadam(), 'bs': def_bs, 'epochs': def_epochs, 'loss': def_loss,
     'act': def_act, 'last_act': def_last_act, 'dropout': def_dropout, 'dropout_rate': def_dropout_rate,
     'add_name': def_add_name}
]

if len(default_dnn_configs) > 0:
    dnn_configs.extend(default_dnn_configs)
if len(potential_configs) > 0:
    dnn_configs.extend(potential_configs)

for config in dnn_configs:
    print('Creating Model ...')
    optimizer = config['optimizer']
    activation = config['act']
    last_activation = config['last_act']
    dropout = config['dropout']
    dropout_rate = config['dropout_rate']
    loss = config['loss']
    expected_epochs = config['epochs']
    batch_size = config['bs']
    additional_name = config['add_name']

    classifier = Sequential()
    input_shape = (training_data.shape[1], )
    classifier.add(Dense(256, activation=activation, input_shape=input_shape))
    if dropout:
        classifier.add(Dropout(rate=dropout_rate))
    classifier.add(Dense(128, activation=activation))
    if dropout:
        classifier.add(Dropout(rate=dropout_rate))
    classifier.add(Dense(64, activation=activation))
    if dropout:
        classifier.add(Dropout(rate=dropout_rate))
    classifier.add(Dense(2, activation=last_activation))
    classifier.compile(loss=loss,
                       optimizer=optimizer,
                       metrics=['acc'])

    print('Creating Setup ...')

    save_path = 'setup'
    setup_numbers = []
    max_setup_number = None
    if os.path.isdir(os.path.join(os.getcwd(), save_path)):
        for setup_name in os.listdir(os.path.join(os.getcwd(), save_path)):
            setup_number = int(setup_name.split('.')[0])
            setup_numbers.append(setup_number)
        max_setup_number = max(setup_numbers) if len(setup_numbers) > 0 else 0
    else:
        max_setup_number = 0

    new_setup_number = max_setup_number + 1
    setup_name = ('%d. dnn_classifier-%d' % (new_setup_number, new_setup_number)) if additional_name is None \
        else ('%d. dnn_classifier-%d-%s' % (new_setup_number, new_setup_number, additional_name))
    setup = Setup(setup_name)
    setup.setModel(classifier)
    setup.setData(training_ids=training_ids, training_data=training_data, training_targets=training_targets,
                  validation_ids=validation_ids, validation_data=validation_data, validation_targets=validation_targets,
                  testing_ids=testing_ids, testing_data=testing_data, testing_targets=testing_targets)
    setup.setOthers({'expected_epochs': expected_epochs,
                     'batch_size': batch_size})
    setup.setOthers({'optimizer': optimizer.__class__.__name__,
                     'lr': float(K.get_value(optimizer.lr)),
                     'loss': loss,
                     'activation': activation,
                     'last_activation': last_activation,
                     'columns': str(training_credit_application.columns.values.tolist()),
                     'dropout': dropout,
                     'dropout_rate': dropout_rate,
                     })
    setup.save(save_path)
