import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from flask import  Flask,render_template,request
app = Flask(__name__, static_url_path='/static')

# Step 2: Generate Synthetic Student Data (Simulating Web Behavior)
def generate_student_data(num_students=10000):
    data = []
    for i in range(num_students):
        time_spent = random.randint(5, 60)  # Time spent on the website in minutes
        num_pages_visited = random.randint(1, 10)  # Number of pages visited
        clicks_on_admission_page = random.randint(0, 1)  # Whether the student clicked on admission info
        student_score = random.randint(600, 800)  # Hypothetical entrance exam score
        application_submitted = random.randint(0, 1)  # 1 if the student submitted an application, 0 otherwise
        data.append([time_spent, num_pages_visited, clicks_on_admission_page, student_score, application_submitted])

    df = pd.DataFrame(data, columns=['time_spent', 'num_pages_visited', 'clicked_admission_page', 'student_score',
                                     'admitted'])
    return df


# Step 3: Preprocess the Data (Split, Scale)
student_data = generate_student_data()

X = student_data.drop('admitted', axis=1)
y = student_data['admitted']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 4: Build the Machine Learning Model (Random Forest)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
with open ('stock.pkl','wb') as file:
    pickle.dump(model,file)
# Step 5: Predict and Evaluate the Model
y_pred = model.predict(X_test)
with open ('stock.pkl','rb') as file:
    model1 = pickle.load(file)
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

# Display results
print(f"Accuracy: {accuracy:.2f}")
print("Confusion Matrix:")
print(cm)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Step 6: Visualize the Results
# Confusion Matrix Heatmap
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Admitted', 'Admitted'],
            yticklabels=['Not Admitted', 'Admitted'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()

# Feature Importance Plot
feature_importance = model.feature_importances_
plt.barh(X.columns, feature_importance)
plt.title('Feature Importance')
plt.show()
@app.route('/')
def Home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
     time_spent=float(request.form['time_spent'])
     num_pages_visited = float(request.form['num_pages_visited'])
     clicked_admission_page = float(request.form['clicked_admission_page'])
     student_score= float(request.form['student_score'])

     result= model.predict([[time_spent,num_pages_visited,clicked_admission_page ,student_score]])[0]
     return render_template('index.html',result="{}".format(result))
if __name__=="__main__":
    app.run(debug=True)