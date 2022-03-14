import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import joblib
from sklearn.metrics import f1_score


df = pd.read_csv('clean_data.csv')

x = df['tweets']
y = df['dialect']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.9, random_state = 42)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)



Tfidf_vect = TfidfVectorizer(max_features=5000)
x_train_tfidf= Tfidf_vect.fit_transform(x_train)
x_test_tfidf = Tfidf_vect.transform(x_test)


clf = LogisticRegression(max_iter =700)
clf.fit(x_train_tfidf, y_train)
y_pred = clf.predict(x_test_tfidf)
print('accuracy:',clf.score(x_test_tfidf,y_test))
print('f1_score',f1_score(y_test, y_pred, average='macro'))
#print(metrics.classification_report(y_test, y_pred))



ml_file = 'ml_model.pkl'
tf_file = 'tfidf.pkl'
joblib.dump(clf, ml_file)
joblib.dump(Tfidf_vect, open(tf_file, "wb"))