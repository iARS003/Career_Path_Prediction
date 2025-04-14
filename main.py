import pandas as pd
import os
from datasets import load_dataset
import joblib

# Importing libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset from Hugging Face
dataset = load_dataset("NxtGenIntern/IT_Job_Roles_Skills_Certifications_Dataset")

# Load dataset to df
df = dataset['train'].to_pandas()

# Clean Data
df.dropna(subset=['Skills', 'Job Title'], inplace=True)

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(df['Skills'], df['Job Title'], test_size=0.2, random_state=42)

# Build Pipeline with TF-IDF and Naive Bayes
model = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_features=5000, stop_words='english')),
    ('clf', MultinomialNB())
])

# Hyperparameter tuning for Naive Bayes
parameters = {
    'tfidf__max_df': [0.85, 0.9, 0.95],
    'clf__alpha': [0.5, 1.0, 2.0]
}

# Use StratifiedKFold with 3 splits for cross-validation
stratified_kfold = StratifiedKFold(n_splits=2)

# Grid Search with StratifiedKFold
grid_search = GridSearchCV(model, parameters, cv=stratified_kfold, n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

# Best model and parameters
print("Best Parameters:", grid_search.best_params_)

# Predict and evaluate the model
y_pred = grid_search.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save the model
joblib.dump(grid_search.best_estimator_, "Career_Prediction.pk1")
print("Improved model is saved")

def predict_career(skills_input):
    prediction = grid_search.best_estimator_.predict([skills_input])
    return prediction[0]
