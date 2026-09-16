import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns
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
import joblib



df = pd.read_csv('healthcare-stroke-balanced-real-oversampled.csv')

#Data cleaning and preprocessing
df['gender'].value_counts()

df = df[df['gender'] != 'Other']
df['gender'].value_counts() 

# sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
# plt.show()

print(df.isnull().sum())
print(df.info())
print(df.describe())

df['bmi'] = df['bmi'].fillna(df['bmi'].median())
print(df.isnull().sum())

df = df.drop(['id'], axis=1)
le = LabelEncoder()
for col in ['gender','ever_married','work_type','Residence_type','smoking_status']:
    df[col] = le.fit_transform(df[col])
joblib.dump(le, "label_encoder.pkl")    

# Clip outliers only for numeric columns
for i in df.drop(columns=['stroke']):    
    q1 = df[i].quantile(0.25)
    q3 = df[i].quantile(0.75)
    iqr = q3 - q1
    min_val = q1 - 1.5 * iqr    
    max_val = q3 + 1.5 * iqr
    df[i] = df[i].clip(lower=min_val, upper=max_val)



#-----------------------------------------------------------------------------------------------------------------------
from imblearn.over_sampling import RandomOverSampler

X = df.drop("stroke", axis=1)
y = df["stroke"]

X_train_resampled, X_test, y_train_resampled, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42,stratify=y
)

# ros = RandomOverSampler(random_state=42)
# X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)
feature_columns = X.columns.tolist()
joblib.dump(feature_columns, "feature_columns.pkl")
#-----------------------------------------------------------------------------------------------------------------------
#logestic regression Model
model = LogisticRegression()
model.fit(X_train_resampled,y_train_resampled)
y_pred = model.predict(X_test)
y_pred_train=model.predict(X_train_resampled)
print("\nLogistic Regression Model:\n")
cm = confusion_matrix(y_test, y_pred)
print("Accuracy: ", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, average='macro'))
print("Recall: ", recall_score(y_test, y_pred, average='macro'))
print("F1 Score: ", f1_score(y_test, y_pred, average='macro'))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
#XGBoost Classifier
xgb=XGBClassifier(n_estimators=100,learning_rate=0.05,max_depth=3,random_state=42,eval_metric="rmse")
xgb.fit(X_train_resampled,y_train_resampled)
y_pred = xgb.predict(X_test)
y_pred_train=xgb.predict(X_train_resampled)
joblib.dump(xgb, "stroke_xgb_model.pkl")
print("\nXGBoost Classifier Model:\n")
cm = confusion_matrix(y_test, y_pred)
print("Accuracy: ", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, average='macro'))
print("Recall: ", recall_score(y_test, y_pred, average='macro'))
print("F1 Score: ", f1_score(y_test, y_pred, average='macro'))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

#Random Forest Classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_resampled, y_train_resampled)
y_pred = rf.predict(X_test)
print("\nRandom Forest Classifier Model:\n")
cm = confusion_matrix(y_test, y_pred)
print("Accuracy: ", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, average='macro'))
print("Recall: ", recall_score(y_test, y_pred, average='macro'))
print("F1 Score: ", f1_score(y_test, y_pred, average='macro'))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
#Decision Tree Classifier
DT = DecisionTreeClassifier()
DT.fit(X_train_resampled, y_train_resampled)
y_pred = DT.predict(X_test)
print("\nDecision Tree Classifier Model:\n")
cm = confusion_matrix(y_test, y_pred)
print("Accuracy: ", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, average='macro'))
print("Recall: ", recall_score(y_test, y_pred, average='macro'))
print("F1 Score: ", f1_score(y_test, y_pred, average='macro'))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
#adaBoost Classifier
ada=AdaBoostClassifier(n_estimators=100,random_state=42)
ada.fit(X_train_resampled, y_train_resampled)
y_pred = ada.predict(X_test)
print("\nAda Boost Classifier Model:\n")
cm = confusion_matrix(y_test, y_pred)
print("Accuracy: ", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, average='macro'))
print("Recall: ", recall_score(y_test, y_pred, average='macro'))
print("F1 Score: ", f1_score(y_test, y_pred, average='macro'))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
#Gradient boosting classifier
gb=GradientBoostingClassifier(n_estimators=100,learning_rate=0.02,max_depth=3,random_state=42)
gb.fit(X_train_resampled, y_train_resampled)
y_pred = gb.predict(X_test)
print("\nGradientBoost Classifier Model:\n")
cm = confusion_matrix(y_test, y_pred)
print("Accuracy: ", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, average='macro'))
print("Recall: ", recall_score(y_test, y_pred, average='macro'))
print("F1 Score: ", f1_score(y_test, y_pred, average='macro'))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
#support vector classifier
svc = SVC(kernel='linear', C=1, gamma='scale')
svc.fit(X_train_resampled, y_train_resampled)
y_pred = svc.predict(X_test)
print("\nSupport vector Classifier Model:\n")
cm = confusion_matrix(y_test, y_pred)
print("Accuracy: ", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, average='macro'))
print("Recall: ", recall_score(y_test, y_pred, average='macro'))
print("F1 Score: ", f1_score(y_test, y_pred, average='macro'))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
#scaling the data 
scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train_resampled)
X_test_scaled=scaler.transform(X_test)
#kNeighbours classifier
knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled,y_train_resampled)
y_pred = knn.predict(X_test_scaled)
print("\nkNeighbours Classifier Model:\n")
cm = confusion_matrix(y_test, y_pred)
print("Accuracy: ", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, average='macro'))
print("Recall: ", recall_score(y_test, y_pred, average='macro'))
print("F1 Score: ", f1_score(y_test, y_pred, average='macro'))
print("\nClassification Report:\n", classification_report(y_test, y_pred))