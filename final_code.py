import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier



df = pd.read_csv('healthcare-dataset-stroke-data-selected-columns.csv')

#Data cleaning and preprocessing
df['gender'].value_counts()

df = df[df['gender'] != 'Other']
df['gender'].value_counts() 

sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.show()

print(df.isnull().sum())
print(df.info())
print(df.describe())

df['bmi'] = df['bmi'].fillna(df['bmi'].mean())
print(df.isnull().sum())

df = df.drop(['id'], axis=1)
le = LabelEncoder()
for col in ['gender','ever_married','work_type','Residence_type','smoking_status']:
    df[col] = le.fit_transform(df[col])

# Clip outliers only for numeric columns
for i in df.drop(columns=['stroke']):    
    q1 = df[i].quantile(0.25)
    q3 = df[i].quantile(0.75)
    iqr = q3 - q1
    min_val = q1 - 1.5 * iqr    
    max_val = q3 + 1.5 * iqr
    df[i] = df[i].clip(lower=min_val, upper=max_val)


for i in df.select_dtypes(include=[np.number]).columns:
    sns.boxplot(x=df[i])
    plt.title(i)
    plt.show()
#-----------------------------------------------------------------------------------------------------------------------
from imblearn.over_sampling import RandomOverSampler

X = df.drop("stroke", axis=1)
y = df["stroke"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42,stratify=y
)

ros = RandomOverSampler(random_state=42)
X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)

