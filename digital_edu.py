#создай здесь свой индивидуальный проект!
import pandas as pd
df = pd.read_csv('train.csv')

df['has_mobile'] = df['has_mobile'].apply(int)
df.drop(['has_photo','bdate','followers_count', 'graduation', 'relation', 'id', 'langs', 'last_seen', 'occupation_name', 'life_main', 'people_main', 'career_start', 'career_end', 'city'], axis = 1, inplace = True)



df[list(pd.get_dummies(df['education_form']).columns)] = pd.get_dummies(df['education_form'])
df['education_form'].fillna('Full-time', inplace = True)
df.drop('education_form', axis = 1, inplace = True)

def sex_a(sex):
    if sex == 2:
        return 1
    return 0
df['sex'] = df['sex'].apply(sex_a)

def status_a(status):
    if status == 'Undergraduate applicant':
        return 1
    elif status == "Student (Master's)" or status == "Student (Specialist)" or status == "Student (Bachelor's)":
        return 2
    elif status == "Alumnus (Master's)" or status == "Alumnus (Specialist)" or status == "Alumnus (Bachelor's)":
        return 3
    else:
        return 0
df['education_status'] = df['education_status'].apply(status_a)

def type_a(type):
    if type == 'university':
        return 1
    else:
        return 0
df['occupation_type'] = df['occupation_type'].apply(type_a)

temp = df['occupation_type'].value_counts()

print(df.info())
print(temp)



from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score


X = df.drop('result', axis = 1)
y = df['result']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
 
classifier = KNeighborsClassifier(n_neighbors = 5)
classifier.fit(X_train, y_train)
 
y_pred = classifier.predict(X_test)
print('Процент правильно предсказанных покупок:', round(accuracy_score(y_test, y_pred) * 100, 2))