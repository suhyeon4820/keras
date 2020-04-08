import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

# 1. 데이터
wine = pd.read_csv('./data/winequality-white.csv', sep=';', encoding='utf-8')

# 데이터를 레이블과 데이터로 분리하기
y = wine['quality']
x = wine.drop('quality', axis=1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# 2. 모델구성하기
model = RandomForestClassifier()

# 3. 훈련
model.fit(x_train, y_train)

# 4. 평가
aaa = model.score(x_test, y_test)
print(aaa)

y_pred = model.predict(x_test)
print('정답률:', accuracy_score(y_test, y_pred))

print(classification_report(y_test, y_pred))

# 에러 : Classification metrics can't handle a mix of continuous-multioutput and multiclass targets