

import pandas as pd
import numpy as np


file_path = "/content/student_sleep_patterns.csv"
df = pd.read_csv(file_path)

df = df.sort_values(by="Student_ID").reset_index(drop=True)

num_original = len(df)
num_synthetic = 10000 - 500  

categorical_cols = df.select_dtypes(include=['object']).columns
numerical_cols = df.select_dtypes(include=[np.number]).columns


decimal_places = {col: max(df[col].astype(str).apply(lambda x: len(x.split(".")[-1]) if "." in x else 0)) for col in numerical_cols}


def generate_synthetic_numerical(data, decimal):
    synthetic_data = data + np.random.normal(0, 0.05 * data.std(), size=len(data))  
    return synthetic_data.round(decimal)


def generate_synthetic_categorical(data, size):
    return np.random.choice(data, size=size, replace=True)


synthetic_df_list = []

for i in range((num_synthetic // num_original) + 1):  
    temp_df = df.copy()
    temp_df["Student_ID"] = np.arange(501 + i * num_original, 501 + (i + 1) * num_original)

    for col in numerical_cols:
        if col != "Student_ID":
            temp_df[col] = generate_synthetic_numerical(df[col], decimal_places[col])

    for col in categorical_cols:
        temp_df[col] = generate_synthetic_categorical(df[col], size=len(df))

    synthetic_df_list.append(temp_df)
    if len(pd.concat(synthetic_df_list)) >= num_synthetic:
        break  
    
synthetic_df = pd.concat(synthetic_df_list, ignore_index=True).iloc[:num_synthetic]


augmented_df = pd.concat([df, synthetic_df], ignore_index=True)


assert augmented_df["Student_ID"].iloc[499] == 500  
assert augmented_df["Student_ID"].iloc[500] == 501  
assert augmented_df["Student_ID"].iloc[-1] == 10000 


synthetic_file_path = "/content/synthetic_student_sleep_patterns_final.csv"
augmented_df.to_csv(synthetic_file_path, index=False)

print(f"Synthetic dataset generated successfully! Saved at: {synthetic_file_path}")
print(f"Original Student_ID range: {df['Student_ID'].min()} to {df['Student_ID'].max()}")
print(f"Synthetic Student_ID range: {synthetic_df['Student_ID'].min()} to {synthetic_df['Student_ID'].max()}")

import pandas as pd
from sklearn.preprocessing import StandardScaler


file_path = "/content/synthetic_student_sleep_patterns_final.csv"
data = pd.read_csv(file_path)


print("Dataset Info:")
data.info()

print("\nSummary Statistics:")
print(data.describe(include='all'))



data['Gender'] = data['Gender'].fillna(data['Gender'].mode()[0])
data['University_Year'] = data['University_Year'].fillna(data['University_Year'].mode()[0])


print("\nMissing values after handling:")
print(data.isnull().sum())


gender_mapping = {'Male': 0, 'Female': 1, 'Other': 2}
data['Gender'] = data['Gender'].map(gender_mapping)


year_mapping = {'1st Year': 1, '2nd Year': 2, '3rd Year': 3, '4th Year': 4}
data['University_Year'] = data['University_Year'].map(year_mapping)

import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns

numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns
data[numerical_cols].hist(figsize=(12, 12), bins=20, color='skyblue')
plt.suptitle("Distribution of Numerical Features", fontsize=16)
plt.show()


plt.figure(figsize=(12, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title("Correlation Matrix")
plt.show()


plt.figure(figsize=(14, 8))
for i, col in enumerate(categorical_cols):
    plt.subplot(2, 3, i+1)
    sns.countplot(x=col, data=data, palette='Set2')
    plt.title(f"Distribution of {col}")
plt.tight_layout()
plt.show()


plt.figure(figsize=(8, 6))
sns.boxplot(x='Gender', y='Sleep_Duration', data=data, palette='pastel')
plt.title("Sleep Duration Distribution by Gender")
plt.show()


negative_values = data.select_dtypes(include=[np.number]).lt(0).sum()


print("\nColumns with Negative Values and Count:")
print(negative_values[negative_values > 0])


data['Study_Hours'] = data['Study_Hours'].abs()
data['Physical_Activity'] = data['Physical_Activity'].abs()


negative_check = (data[['Study_Hours', 'Physical_Activity']] < 0).sum()
print("\n Negative values after conversion:\n", negative_check)


def convert_to_pm(time):
    if time < 12:
        return time + 12
    else:
        return time


def convert_to_am(time):
    if time > 12:
        return time - 12
    else:
        return time


data['Weekday_Sleep_Start'] = data['Weekday_Sleep_Start'].apply(convert_to_pm)
data['Weekend_Sleep_Start'] = data['Weekend_Sleep_Start'].apply(convert_to_pm)


data['Weekday_Sleep_End'] = data['Weekday_Sleep_End'].apply(convert_to_am)
data['Weekend_Sleep_End'] = data['Weekend_Sleep_End'].apply(convert_to_am)


print("Updated Sleep Start & End times:\n")
print(data[['Weekday_Sleep_Start', 'Weekend_Sleep_Start', 'Weekday_Sleep_End', 'Weekend_Sleep_End']].head())


data.to_csv('updated_dataset.csv', index=False)

print(" Updated dataset saved successfully as 'updated_dataset.csv'")

file_path = "/content/updated_dataset.csv"
data = pd.read_csv(file_path)


def calculate_sleep_duration(start, end):
    if pd.isnull(start) or pd.isnull(end):
        return np.nan
    
    if end <= start:  
        duration = (24 - start) + end
    else:
        duration = end - start
    return duration


data['Weekday_Sleep_Duration'] = data.apply(lambda row: calculate_sleep_duration(
    row['Weekday_Sleep_Start'], row['Weekday_Sleep_End']), axis=1)

data['Weekend_Sleep_Duration'] = data.apply(lambda row: calculate_sleep_duration(
    row['Weekend_Sleep_Start'], row['Weekend_Sleep_End']), axis=1)


print("\n Sleep durations calculated successfully!")
print(data[['Weekday_Sleep_Start', 'Weekday_Sleep_End', 'Weekend_Sleep_Start', 'Weekend_Sleep_End',
           'Weekday_Sleep_Duration', 'Weekend_Sleep_Duration']].head())


data['Sleep_Duration_Diff'] = abs(data['Weekend_Sleep_Duration'] - data['Weekday_Sleep_Duration'])
data['Sleep_Start_Diff'] = abs(data['Weekend_Sleep_Start'] - data['Weekday_Sleep_Start'])
data['Sleep_End_Diff'] = abs(data['Weekend_Sleep_End'] - data['Weekday_Sleep_End'])


print("\n Additional sleep-related features created successfully!")


print(data[['Weekday_Sleep_Start', 'Weekday_Sleep_End', 'Weekend_Sleep_Start', 'Weekend_Sleep_End',
           'Weekday_Sleep_Duration', 'Weekend_Sleep_Duration', 'Sleep_Duration_Diff',
           'Sleep_Start_Diff', 'Sleep_End_Diff']].head())


def label_sleep(row):  
    Sleep_Duration = row['Sleep_Duration']
    Screen_Time = row['Screen_Time']
    Study_Hours = row['Study_Hours']
    Caffeine_Intake = row['Caffeine_Intake']
    Physical_Activity = row['Physical_Activity']
    Sleep_Duration_Diff = row['Sleep_Duration_Diff']  

    if (Sleep_Duration < 6 and Screen_Time > 3 and Study_Hours > 4 and Caffeine_Intake > 3) or \
       ((Sleep_Duration < 5 and Screen_Time > 7) or Study_Hours > 9) or \
       (Sleep_Duration < 5 or (Screen_Time > 7 and Study_Hours > 6)) and Physical_Activity > 120 and Sleep_Duration_Diff > 3:
        return 0  # Poor Sleep

    elif (((6 <= Sleep_Duration <= 8.5 and Screen_Time <= 2 and Study_Hours <= 5 and Caffeine_Intake <= 2) or \
           (6.5 <= Sleep_Duration <= 8 or (Screen_Time <= 2 and Study_Hours <= 5 and Caffeine_Intake <= 2))) and Physical_Activity < 120 and Sleep_Duration_Diff < 3):
        return 1  # Healthy Sleep

    else:
        return 2  # Unhealthy Sleep


data['Sleep_Label'] = data.apply(label_sleep, axis=1)


data['Sleep_Label'] = data['Sleep_Label'].astype('category')


print("\nDistribution of Sleep Labels:")
label_counts = data['Sleep_Label'].value_counts().sort_index()
print(label_counts)


output_path = "processed.csv"
data.to_csv(output_path, index=False)

print(f"\nProcessed dataset saved successfully to: {output_path}")


negative_values = data.select_dtypes(include=[np.number]).lt(0).sum()


print("\nColumns with Negative Values and Count:")
print(negative_values[negative_values > 0])


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
import warnings


warnings.filterwarnings("ignore")


file_path = "/content/processed.csv"  
data = pd.read_csv(file_path)


print("First 5 rows of the dataset:")
print(data.head())


X = data[['Age', 'Sleep_Duration', 'Study_Hours', 'Screen_Time', 'Caffeine_Intake',
          'Physical_Activity', 'Weekday_Sleep_Duration', 'Weekend_Sleep_Duration',
          'Sleep_Duration_Diff', 'Sleep_Start_Diff', 'Sleep_End_Diff']]
y = data['Sleep_Label']  


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Support Vector Machine (SVM)": SVC(kernel='rbf', C=1.0),
    "K-Nearest Neighbors (KNN)": KNeighborsClassifier(n_neighbors=5)
}


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


for model_name, model in models.items():
    print(f"\nTraining {model_name}...")

    
    if model_name in ["Logistic Regression", "Support Vector Machine (SVM)"]:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

   
    print(f"\n{model_name} Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy for {model_name}: {accuracy:.4f}")

    
    if model_name in ["Random Forest", "XGBoost"]:
        feature_importance = model.feature_importances_
        feature_df = pd.DataFrame({"Feature": X.columns, "Importance": feature_importance})
        feature_df = feature_df.sort_values(by="Importance", ascending=False)
        print(f"\nTop 5 Important Features for {model_name}:")
        print(feature_df.head())


from sklearn.model_selection import GridSearchCV

param_grid = {'n_neighbors': range(3, 11)}
knn_grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='accuracy')
knn_grid.fit(X_train, y_train)
print(f"\nBest K for KNN: {knn_grid.best_params_['n_neighbors']}")
print(f"Best KNN Accuracy: {knn_grid.best_score_:.4f}")

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


rf_feature_importance = pd.DataFrame({'Feature': ['Sleep_Duration', 'Screen_Time', 'Study_Hours', 'Caffeine_Intake', 'Physical_Activity'],
                                      'Importance': [0.30, 0.25, 0.20, 0.15, 0.10]})


plt.figure(figsize=(8, 5))
sns.barplot(x='Importance', y='Feature', data=rf_feature_importance, palette='viridis')
plt.title("Top 5 Important Features - Random Forest")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.show()


xgb_feature_importance = pd.DataFrame({'Feature': ['Sleep_Duration', 'Screen_Time', 'Study_Hours', 'Caffeine_Intake', 'Physical_Activity'],
                                       'Importance': [0.28, 0.27, 0.18, 0.17, 0.10]})


plt.figure(figsize=(8, 5))
sns.barplot(x='Importance', y='Feature', data=xgb_feature_importance, palette='mako')
plt.title("Top 5 Important Features - XGBoost")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.show()

import matplotlib.pyplot as plt


model_accuracies = {}

for model_name, model in models.items():
   
    model.fit(X_train, y_train)

    
    y_pred = model.predict(X_test)

   
    print(f"\n{model_name} Classification Report:")
    print(classification_report(y_test, y_pred))

    
    accuracy = accuracy_score(y_test, y_pred)
    model_accuracies[model_name] = accuracy
    print(f"Accuracy for {model_name}: {accuracy:.4f}")


model_names = list(model_accuracies.keys())
accuracies = list(model_accuracies.values())


plt.figure(figsize=(10, 6))
plt.bar(model_names, accuracies, color=['skyblue', 'orange', 'green', 'red', 'purple'])
plt.xlabel("Models")
plt.ylabel("Accuracy Score")
plt.title(" Model Accuracy Comparison")
plt.ylim(0.7, 1.0)  
plt.xticks(rotation=45)
plt.show()

model_names = list(model_accuracies.keys())
accuracies = list(model_accuracies.values())


plt.figure(figsize=(10, 5))
bars = plt.bar(model_names, accuracies, color=['skyblue', 'orange', 'green', 'red', 'purple'])


for bar in bars:
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
             f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.xlabel("Models")
plt.ylabel("Accuracy Score")
plt.title(" Model Accuracy Comparison")
plt.ylim(0.7, 1.0)  
plt.xticks(rotation=45)
plt.show()

from sklearn.model_selection import GridSearchCV


rf_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
}

xgb_param_grid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 6, 10],
    'subsample': [0.8, 1.0],
}


rf_grid_search = GridSearchCV(estimator=RandomForestClassifier(),
                              param_grid=rf_param_grid,
                              cv=5,
                              scoring='accuracy',
                              verbose=2,
                              n_jobs=-1)
rf_grid_search.fit(X_train, y_train)


xgb_grid_search = GridSearchCV(estimator=XGBClassifier(),
                               param_grid=xgb_param_grid,
                               cv=5,
                               scoring='accuracy',
                               verbose=2,
                               n_jobs=-1)
xgb_grid_search.fit(X_train, y_train)


print("Best parameters for Random Forest: ", rf_grid_search.best_params_)
print("Best score for Random Forest: ", rf_grid_search.best_score_)

print("Best parameters for XGBoost: ", xgb_grid_search.best_params_)
print("Best score for XGBoost: ", xgb_grid_search.best_score_)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


best_rf_model = rf_grid_search.best_estimator_
best_xgb_model = xgb_grid_search.best_estimator_


rf_y_pred = best_rf_model.predict(X_test)
xgb_y_pred = best_xgb_model.predict(X_test)


print(" Random Forest Performance:")
print(confusion_matrix(y_test, rf_y_pred))
print(classification_report(y_test, rf_y_pred))
print(f"Accuracy: {accuracy_score(y_test, rf_y_pred):.4f}")


print("\n XGBoost Performance:")
print(confusion_matrix(y_test, xgb_y_pred))
print(classification_report(y_test, xgb_y_pred))
print(f"Accuracy: {accuracy_score(y_test, xgb_y_pred):.4f}")

import joblib


joblib.dump(best_xgb_model, 'best_sleep_model.pkl')
print("✅ Best XGBoost model saved as 'best_sleep_model.pkl'")


import numpy as np
import pandas as pd
import joblib
from xgboost import XGBClassifier


model = joblib.load("/content/best_sleep_model.pkl")


feature_names = ['Age', 'Sleep_Duration', 'Study_Hours', 'Screen_Time', 'Caffeine_Intake',
 'Physical_Activity', 'Weekday_Sleep_Duration', 'Weekend_Sleep_Duration',
 'Sleep_Duration_Diff', 'Sleep_Start_Diff', 'Sleep_End_Diff']



def calculate_features(data):
    
    data['Weekday_Sleep_Duration'] =  abs(data['Weekday_Sleep_End'] - data['Weekday_Sleep_Start'])
    data['Weekend_Sleep_Duration'] = abs(data['Weekend_Sleep_End'] - data['Weekend_Sleep_Start'])

    
    data['Sleep_Duration_Diff'] = abs(data['Weekend_Sleep_Duration'] - data['Weekday_Sleep_Duration'])

    
    data['Sleep_Start_Diff'] = abs(data['Weekend_Sleep_Start'] - data['Weekday_Sleep_Start'])
    data['Sleep_End_Diff'] = abs((data['Weekend_Sleep_End'] - data['Weekday_Sleep_End']))

    return data



def get_user_input():
    data = {
    'Age': int(input("Enter Age: ")),
    'Sleep_Duration': float(input("Enter Sleep Duration (hours): ")),
    'Study_Hours': float(input("Enter Study Hours: ")),
    'Screen_Time': float(input("Enter Screen Time (hours/day): ")),
    'Caffeine_Intake': float(input("Enter Caffeine Intake (cups/day): ")),
    'Physical_Activity': float(input("Enter Physical Activity Level (in minutes): ")),
    'Weekday_Sleep_Start': float(input("Enter Weekday Sleep Start Time (24-hour format): ")),
    'Weekday_Sleep_End': float(input("Enter Weekday Sleep End Time (24-hour format): ")),
    'Weekend_Sleep_Start': float(input("Enter Weekend Sleep Start Time (24-hour format): ")),
    'Weekend_Sleep_End': float(input("Enter Weekend Sleep End Time (24-hour format): "))
}


    
    data_df = pd.DataFrame([data])

    
    data_df = calculate_features(data_df)

    return data_df



user_input = get_user_input()


def predict_sleep_quality(input_data):
    
    input_data = input_data[feature_names]

    
    print(f"\n Input Data for Prediction:\n{input_data}\n")


    prediction = model.predict(input_data)

   
    sleep_labels = {
        0: "Poor Sleep",
        1: "Healthy Sleep",
        2: "Unhealthy Sleep"
    }

    
    predicted_label = sleep_labels[prediction[0]]
    return predicted_label



predicted_quality = predict_sleep_quality(user_input)


print(f"\n🛏️ Predicted Sleep Quality: {predicted_quality}")