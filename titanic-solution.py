# Generated from: titanic-solution.ipynb
# Converted at: 2026-01-09T18:38:31.618Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

# # Titanic - Machine Learning from Disaster                     ..... ![icons8-spaceship-64.png](attachment:7f9d0319-9a53-44ba-88a9-044a52ff7f43.png)  
# 
# ## Author: RIDDY MAZUMDER
# ## 🔗 Connect with Me
# > [![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/riddymazumder)
# > [![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/RiddyMazumder)
# > [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/riddy-mazumder-7bab46338/)
# > [![Gmail](https://img.shields.io/badge/Gmail-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:riddymazumder1971@gmail.com)
# 
# ## Description 
# **This notebook follows a complete end-to-end data science workflow, from loading data to model evaluation and final submission.**  
# ****Each section is clearly explained and well-structured for learning and presentation.****
# 


# ## 1. Libraries Required
# 
# ****In this section, we import all the necessary Python libraries used throughout the project.****  
# **These include libraries for**:
# - **Data manipulation**  
# - **Visualization** 
# - **Machine learning**


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# ## 2. Load Dataset


df_train=pd.read_csv('/kaggle/input/titanic/train.csv')
df_test=pd.read_csv('/kaggle/input/titanic/test.csv')
df_test['Survived']=False
df=pd.concat([df_train,df_test],sort=True)

df.head()

# ## 3. Data Exploration & Cleaning
# 
# ## 3.1 Overview
# 
# **Check shape, missing values, data types.**


df.info()

# ## 3.2 Visualization


sns.heatmap(df.isnull(), yticklabels=False, cbar=True, cmap='viridis')
plt.title("Missing Values Heatmap")
plt.show()
missing_data = pd.DataFrame({
    'Missing_Values': df.isnull().sum(),
    'Percentage': (df.isnull().sum() / len(df)) * 100,
    'Data_Type': df.dtypes
})
missing_data = missing_data[missing_data['Missing_Values'] > 0]

missing_data = missing_data.sort_values(by='Missing_Values', ascending=False)

print("Missing Data Summary:\n")
display(missing_data)

# ## 3.3 Survival Rate
# **Survival Rate of Women**,**Survival Rate of men**


print("Rate of women survived:", df[df['Sex'] == 'female']['Survived'].mean())
print('Rate of man survived', df[df['Sex'] == 'male']['Survived'].mean())

# # 3.4 Filling missing values


df['Age']=df['Age'].fillna(df['Age'].mean())
df['Fare']=df['Fare'].fillna(df['Fare'].mean())

df['Embarked']=df['Embarked'].fillna('S')
df['Sex']=df['Sex'].fillna('unknown')

df.drop(['Cabin'],axis=1,inplace=True)

# # 3.5 Encoding 


df['Embarked']=df['Embarked'].map({'C':0,'Q':1,'S':2})

df['Ticket']=df['Ticket'].astype('category').cat.codes
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1,'unknown': -1}).astype('int64')


# # 3.6 Remove irrelevant columns


df.drop(['Name'],axis=1,inplace=True)

def check_missing(df):
    # Summary table
    missing_data = pd.DataFrame({
        'Missing_Values': df.isnull().sum(),
        'Percentage': (df.isnull().sum() / len(df)) * 100,
        'Data_Type': df.dtypes
    })
    missing_data = missing_data[missing_data['Missing_Values'] > 0]
    missing_data = missing_data.sort_values(by='Missing_Values', ascending=False)
    
    # Print + display
    print("🔍 Missing Data Summary:\n")
    display(missing_data)
    
    # Heatmap
    plt.figure(figsize=(10,6))
    sns.heatmap(df.isnull(), yticklabels=False, cbar=True, cmap='viridis')
    plt.title("Missing Values Heatmap")
    plt.show()
check_missing(df)

df_train, df_test = df[:df_train.shape[0]], df[df_train.shape[0]:]
df_test = df_test.drop(columns = 'Survived')
df_train.shape, df_test.shape

# ## 4. Model Building
# **Libraries Required**


from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier


# # Split Data


X=df_train.drop(columns='Survived')
y=df_train['Survived']

# # Train Model,Evaluate Model


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
model = XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss',
    verbosity=0   # <-- set to 0 to suppress info/warnings
)
model.fit(X_train, y_train)

# === . Evaluate ===
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# === . Feature Importance ===
importances = pd.Series(model.feature_importances_, index=X.columns)
print(importances.sort_values(ascending=False).head(10))


# # Submission File


submission = pd.DataFrame({
    'PassengerId': df_test['PassengerId'],
    'Survived': model.predict(df_test[X.columns])  # prediction on test set
})


submission.to_csv('submission.csv', index=False)
print(" Submission file created: submission.csv")

# ## 4.1 Model Accuracy_Score
# **Predictions on training data**


from sklearn.model_selection import cross_val_score

# === Cross-validation RMSE ===
scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')  # regression
rmse_scores = np.sqrt(-scores)
print(f"CV RMSE: {rmse_scores.mean():.4f} ± {rmse_scores.std():.4f}")


# For demonstration, using the mean RMSE as "local score"
local_score = rmse_scores.mean()
print(f"Local Score (simulating leaderboard): {local_score:.4f}")


acc = accuracy_score(y_test, y_pred)  # note: using X_test & y_test as local validation
print(f"Local Accuracy (simulating Kaggle leaderboard): {acc:.4f}")