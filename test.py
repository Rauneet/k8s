import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('path_to_csv_file.csv')
categorical_features = ['assignee', 'status', 'priority', 'resolution']
df = pd.get_dummies(df, columns=categorical_features)
#Assuming pull request is a binary 
df['pull request'] = pd.get_dummies(df['pull request'])
#define feature and target 
X = df.drop(['task_name', 'bug_present'], axis=1)  # 'bug_present' is your target variable
y = df['bug_present']
#Splitting the dataset 
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=42,test_size=0.2)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train[['comments,assigned_comments']])
X_test_scaled = scaler.fit_transform(X_test[['comments', 'assigned_comments']])

model = RandomForestClassifier(random_state=42)

model.fit(X_train,y_train)

y_pred = model.predict(X_test)

#model evalulation 
print("Accuracy:" ,accuracy_score(y_test,y_pred))
print("Classification report: \n", classification_report(y_test,y_pred))
