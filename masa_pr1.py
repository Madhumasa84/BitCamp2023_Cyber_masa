import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer  # Import SimpleImputer

data = pd.read_csv('C:/Users/smkp8/Downloads/ML model for machine images/2 nd ml/phishing_email.csv/Phishing_Email.csv')


data['Email Text'].fillna('', inplace=True)

X = data['Email Text'] 
y = data['Email Type']  
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)


clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_vectorized, y_train)

y_pred = clf.predict(X_test_vectorized)


accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
