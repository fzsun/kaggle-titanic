# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 15:00:37 2018

@author: fangzhou
"""

# Handle table-like data and matrices
import numpy as np
import pandas as pd

# Modelling Algorithms
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier,
                              GradientBoostingClassifier, ExtraTreesClassifier,
                              VotingClassifier)


# Modelling Helpers
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import Imputer , Normalizer, scale
from sklearn.feature_selection import RFECV

# Visualisation
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns

# Configure visualisations
mpl.style.use( 'ggplot' )
sns.set_style( 'white' )
pylab.rcParams[ 'figure.figsize' ] = 8 , 6

#%% Load data
train = pd.read_csv("inputs/train.csv")
test = pd.read_csv("inputs/test.csv")
full = train.append(test, ignore_index = True)
Y_train = full[:891].Survived
del train, test
del full['Survived']

#%% Title
full['Title'] = full.Name.map(lambda n:n.split(',')[1].split('.')[0].strip())
#full.Title.value_counts()
#group_median = full.groupby('Title').agg('median')
#%% Transform title
title_dict = {
            'Mr':'Mr',
            'Mrs':'Mrs',
            'Miss':'Miss',
            'Master':'Master',
            'Don':'Royalty',
            'Rev':'Officer',
            'Dr':'Officer',
            'Mme':'Miss',
            'Ms':'Mrs',
            'Major':'Officer',
            'Lady':'Royalty',
            'Sir':'Royalty',
            'Mlle':'Mrs',
            'Col':'Officer',
            'Capt':'Officer',
            'the Countess':'Royalty',
            'Jonkheer':'Royalty',
            'Dona':'Royalty'}
full['Title'] = full['Title'].map(title_dict)
#full.Title.value_counts()
#group_median = full.groupby('Title').agg('median')

#%% Delete redundant features
del full['Name'], full['PassengerId']
#%% Missing values
full.Cabin.fillna('U', inplace=True)
full.Cabin = full.Cabin.map(lambda c:c[0])
full.Embarked = full.Embarked.fillna('C')
train = full[:891]
group_median = train.groupby(['Pclass','Title']).agg('median')
group_median = group_median[['Age','Fare']].reset_index()

for i, row in full.loc[full.Age.isnull()].iterrows():
    p,  t = row[['Pclass','Title']]
    full.loc[i,'Age'] = group_median[(group_median.Pclass==p) &
                           (group_median.Title==t)].Age.values[0]
#group_count = full.groupby(['Pclass','Title']).agg('count')

for i, row in full.loc[full.Fare.isnull()].iterrows():
    p,  t = row[['Pclass','Title']]
    full.loc[i,'Fare'] = group_median[(group_median.Pclass==p) &
                           (group_median.Title==t)].Fare.values[0]

#%% Transform Family related features
full['Fare'] = np.log(full.Fare+1).astype(int)
full['Family'] = full.SibSp + full.Parch + 1

for i, row in full.iterrows():
    if row.Age <= 16:
        full.loc[i,'Age'] = 'Child'
    elif row.Age <= 32:
        full.loc[i,'Age'] = 'Youth'
    elif row.Age <= 48:
        full.loc[i,'Age'] = 'Middle1'
    elif row.Age <= 64:
        full.loc[i,'Age'] = 'Middle2'
    else:
        full.loc[i,'Age'] = 'Old'

    if row.Parch == 0:
        full.loc[i,'Parch'] = '0'
    elif row.Parch <=2:
        full.loc[i,'Parch'] = 'Small'
    else:
        full.loc[i,'Parch'] = 'Big'

    if row.SibSp == 0:
        full.loc[i,'SibSp'] = '0'
    elif row.SibSp <=1:
        full.loc[i,'SibSp'] = 'Small'
    else:
        full.loc[i,'SibSp'] = 'Big'

    if row.Family <= 1:
        full.loc[i,'Family'] = 'Single'
    elif row.Family <=2:
        full.loc[i,'Family'] = 'Small'
    elif row.Family <=5:
        full.loc[i,'Family'] = 'Middle'
    else:
        full.loc[i,'Family'] = 'Big'


#%% Data Preparation
age = pd.get_dummies(full.Age,  prefix='Age')
cabin = pd.get_dummies(full.Cabin, prefix='Cabin')
embarked = pd.get_dummies(full.Embarked, prefix='Embarked')
fare = pd.get_dummies(full.Fare,  prefix='Fare')
family = pd.get_dummies(full.Family,  prefix='Family')
parch = pd.get_dummies(full.Parch,  prefix='Parch')
pclass = pd.get_dummies(full.Pclass, prefix='Pclass')
sex = pd.Series(np.where(full.Sex=='male', 1, 0), name='Sex' )
sibsp = pd.get_dummies(full.SibSp,  prefix='SibSp')
title = pd.get_dummies(full.Title, prefix='Title')

X_full = pd.concat([age,
                    cabin,
                    embarked,
                    fare,
                    family,
                    parch,
                    pclass,
                    sex,
                    sibsp,
                    title
                    ], axis=1)

#%% Select best models
X_train = X_full[:891]

models = [LogisticRegression(), KNeighborsClassifier(), GaussianNB(),
          SVC(), LinearSVC(), DecisionTreeClassifier(),
          RandomForestClassifier(), AdaBoostClassifier(),
          GradientBoostingClassifier(), ExtraTreesClassifier()]
for model in models:
    model_name = str(model.__class__).split('.')[-1].split("'")[0]
    print(f'Cross-validation of : {model_name}')
    score = cross_val_score(model, X_train, Y_train, cv=5, scoring='f1').mean()
    print (f'CV score = {score}')
    print('****')

#%% Tuning hyperparameters

tree_clfs = [RandomForestClassifier(),
             ExtraTreesClassifier()]

LR_param_grid = {'penalty': ['l1', 'l2'],
                 'C': [0.001, 0.01, 0.1, 1, 5, 10, 50, 100]}
AdaB_param_grid = {'n_estimators': [20, 50, 100, 200],
                   'learning_rate': [.01, 0.1, 0.3, 0.7, 1, 2],
                   'algorithm': ['SAMME', 'SAMME.R']}
tree_param_grid = {'max_depth': [4, 6, 8, 10],
                   'n_estimators': [50, 100, 200],
                   'min_samples_split': [2, 3, 10],
                   'min_samples_leaf': [1, 3, 10]}
GBst_param_grid = {**tree_param_grid,
                   'loss': ['deviance', 'exponential'],
                   'learning_rate': [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3],
                   'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9]}

cross_validation = StratifiedKFold(n_splits=5)

setups = [{'estimator': LogisticRegression(), 'param_grid': LR_param_grid},
          {'estimator': AdaBoostClassifier(), 'param_grid': AdaB_param_grid},
          {'estimator': GradientBoostingClassifier(), 'param_grid': GBst_param_grid},
          {'estimator': RandomForestClassifier(), 'param_grid': tree_param_grid},
          {'estimator': ExtraTreesClassifier(), 'param_grid': tree_param_grid}]

tuned_models = []
for clf in setups:
    grid_search = GridSearchCV(**clf, scoring='f1', cv=cross_validation,
                               verbose=2, n_jobs=8)
    grid_search.fit(X_train, Y_train)
    tuned_model = grid_search.best_estimator_
    tuned_models.append(tuned_model)
    model_name = str(tuned_model.__class__).split('.')[-1].split("'")[0]
    print(f'{model_name} best score: {grid_search.best_score_}')
    print(f'Best parameters: {grid_search.best_params_}\n')

#%% Model Ensemble: Voting classifier
name_model_pairs = [(str(clf.__class__), clf) for clf in tuned_models]

eclf = VotingClassifier(estimators = name_model_pairs, voting = 'soft',
                         weights=[1,1,1,1,1])
eclf.fit(X_train, Y_train)

#%% Tuning hyperparameters

parameters = {'bootstrap': False, 'max_depth': 6, 'max_features': 'sqrt',
              'min_samples_leaf': 3, 'min_samples_split': 3,
              'n_estimators': 1000}

model = RandomForestClassifier(**parameters)
model.fit(X_train, Y_train)

#%% Submission
print(f'Cross-validation of : {model.__class__}')
score = cross_val_score(model, X_train, Y_train, cv=5, scoring='accuracy').mean()
print (f'CV score = {score}')

X_test = X_full[891:]
my_prediction = model.predict(X_test).astype(int)
df_output = pd.DataFrame()
aux = pd.read_csv('inputs/test.csv')
df_output['PassengerId'] = aux['PassengerId']
df_output['Survived'] = my_prediction
model_name = str(model.__class__).split('.')[-1].split("'")[0]
from datetime import datetime
file_name = f"{score:.5f}_{model_name}_{datetime.now().strftime('%y%m%d-%H%M')}"
df_output.to_csv(f'outputs/{file_name}.csv', index=False)





