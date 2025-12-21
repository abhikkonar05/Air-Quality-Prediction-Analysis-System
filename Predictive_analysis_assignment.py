import pandas as pd
import numpy as np
import seaborn as sns
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,LogisticRegression,Ridge,Lasso
from sklearn.metrics import mean_squared_error,root_mean_squared_error,mean_absolute_error,f1_score,accuracy_score,precision_score,recall_score
from sklearn.preprocessing import MinMaxScaler,StandardScaler,LabelEncoder,OneHotEncoder
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier,GradientBoostingRegressor,GradientBoostingClassifier,AdaBoostClassifier
import warnings
warnings.filterwarnings("ignore")
data=pd.read_csv("C:/Users/Abhik Konar/Downloads/Air_Quality_Purple_Air_Sensors.csv")
df=pd.DataFrame(data)
print(df['REPORTED_DATETIME'])
print(df.isnull().sum())
#print(df.describe())
df['HUMIDITY']=df['HUMIDITY'].fillna(df['HUMIDITY'].mean())
df['TEMPERATURE']=df['TEMPERATURE'].fillna(df['TEMPERATURE'].median())
df['PRESSURE']=df['PRESSURE'].fillna(df['PRESSURE'].median())
df['PM1']=df['PM1'].fillna(df['PM1'].mean())
df['PM2_5']=df['PM2_5'].fillna(df['PM2_5'].mean())
df['PM10']=df['PM10'].fillna(df['PM10'].mean())
df['PM2_5_CF_1']=df['PM2_5_CF_1'].fillna(df['PM2_5_CF_1'].mean())
df['PM2_5_ALT']=df['PM2_5_ALT'].fillna(df['PM2_5_ALT'].mean())
df['CONFIDENCE']=df['CONFIDENCE'].fillna(df['CONFIDENCE'].median())
df['VOC']=df['VOC'].fillna(df['VOC'].median())
print(df)
#print(df.isnull().sum())
df['REPORTED_DATETIME'] = pd.to_datetime(df['REPORTED_DATETIME'])
df['DATE'] = df['REPORTED_DATETIME'].dt.date
df['HOUR'] = df['REPORTED_DATETIME'].dt.hour
df['MONTH'] = df['REPORTED_DATETIME'].dt.month
df['DAY_OF_WEEK'] = df['REPORTED_DATETIME'].dt.dayofweek
df['IS_WEEKEND'] = df['DAY_OF_WEEK'].apply(lambda x: 1 if x >= 5 else 0)
print(df['IS_WEEKEND'])
df['LAT_ROUND'] = df['LATITUDE'].round(2)
df['LON_ROUND'] = df['LONGITUDE'].round(2)
col=['HUMIDITY','TEMPERATURE','PRESSURE','PM1','PM10','VOC','HOUR','MONTH','DAY_OF_WEEK','IS_WEEKEND','LAT_ROUND','LON_ROUND']
X=df[col]
y_reg=df['PM2_5']
X_train,X_test,y_reg_train,y_reg_test=train_test_split(X,y_reg,test_size=0.2,random_state=42)
scaler=StandardScaler()
X_train_s=scaler.fit_transform(X_train)
X_test_s=scaler.fit_transform(X_test)
#regression_models
reg_models={
    "Linear Regression":LinearRegression(),
    "Ridge Regression":Ridge(alpha=1.0),
    "Lasso Regression":Lasso(alpha=0.001),
    "KNN Regression":KNeighborsRegressor(n_neighbors=5),
    "Decision Tree Regressor":DecisionTreeRegressor(),
    "Random Forest Regressor":RandomForestRegressor(n_estimators=200),
    "Gradient Boosting Regressor":GradientBoostingRegressor()
}
reg_results={}
for name,model in reg_models.items():
    model.fit(X_train_s,y_reg_train)
    pred=model.predict(X_test_s)
    mae=mean_absolute_error(y_reg_test,pred)
    mse=mean_squared_error(y_reg_test,pred)
    rmse=np.sqrt(mse)
    reg_results[name]=[mae,mse,rmse]
    print("Prediction:",pred)
    print("MAE:",mae)
    print("MSE:",mse)
    print("RMSE:",rmse)
    reg_result_df=pd.DataFrame(reg_results,index=["MAE","MSE","RMSE"]).T
    print("REGRESSION MODEL")
    print(reg_result_df)
##classification model
X_clas = df[col]
df['AIR_QUALITY_LABEL']=df['PM2_5'].apply(lambda x:1 if x>50 else 0)
y_clas=df['AIR_QUALITY_LABEL']
scaler=StandardScaler()
X_train_c,X_test_c,y_train_c,y_test_c=train_test_split(X_clas,y_clas,test_size=0.2,random_state=42)
X_train_c_s=scaler.fit_transform(X_train_c)
X_test_c_s=scaler.transform(X_test_c)
class_models={
    "Logistic Regression":LogisticRegression(),
    "KNN Classifier":KNeighborsClassifier(),
    "Decision Tree Classifier":DecisionTreeClassifier(),
    "Random Forest Classifier":RandomForestClassifier(n_estimators=200),
    "Gradient Boosting Classifier":GradientBoostingClassifier(),
    "AdaBoost Classifier":AdaBoostClassifier()
}
class_results={}
for names, model in class_models.items():
    model.fit(X_train_c_s,y_train_c)
    predic=model.predict(X_test_c_s)
    acc=accuracy_score(y_test_c,predic)
    f1=f1_score(y_test_c,predic)
    preci=precision_score(y_test_c,predic)
    rec=recall_score(y_test_c,predic)
    class_results[names]=[acc,f1,preci,rec]
    print("Prediction:",predic)
    print("Accuracy:",acc)
    print("F1 Score:",f1)
    print("Precision_score:",preci)
    print("Recall_Score",recall_score)
    class_results_df=pd.DataFrame(class_results,index=["Accuracy","F1 Score","Precision Score","Recall Score"]).T
    print("Classification Model")
    print(class_results_df)

good_model=RandomForestClassifier()
good_model.fit(X_train_c,y_train_c)
pred_good=good_model.predict(X_test_c)
print("Confusion Matrix:",confusion_matrix(y_test_c,pred_good))
plt.figure(figsize=(6,4))
sns.heatmap(confusion_matrix(y_test_c,pred_good),annot=True,fmt="d",cmap="Blues")
plt.title("Confusion Matrix-Good Classifier")
plt.show()

plt.figure(figsize=(7,5))
plt.scatter(y_reg_test,pred, alpha=0.6)
plt.plot([y_reg_test.min(), y_reg_test.max()],
         [y_reg_test.min(), y_reg_test.max()],
         'r--', label="Perfect Prediction")
plt.title("Actual vs Predicted (Regression)")
plt.xlabel("Actual PM2.5")
plt.ylabel("Predicted PM2.5")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10,5))
plt.bar(class_results_df.index, class_results_df['Accuracy'])
plt.title("Classification Model Accuracy Comparison")
plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.xticks(rotation=45)
plt.grid(True)
plt.show()

plt.figure(figsize=(7,5))
plt.scatter(y_test_c,predic, alpha=0.6)
plt.plot([y_test_c.min(), y_test_c.max()],
         [y_test_c.min(), y_test_c.max()],
         'r--', label="Perfect Prediction")
plt.title("Actual vs Predicted (Classification)")
plt.xlabel("Actual PM2.5")
plt.ylabel("Predicted PM2.5")
plt.legend()
plt.grid(True)
plt.show()

pickle.dump(reg_models, open("regression_model.pkl", "wb"))
pickle.dump(class_models, open("classification_model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))

print("All models saved successfully!")



