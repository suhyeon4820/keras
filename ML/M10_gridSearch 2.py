import pandas as pd
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import warnings
from sklearn.utils.testing import all_estimators
# sklearn 0.20.3에서 31개
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

import warnings
warnings.filterwarnings('ignore')

# 붓꽃 데이터 읽어 들이기
iris_data = pd.read_csv('./data/iris2.csv', encoding='utf-8')

# 붗꽃 데이터를 레이블과 입력 데이터로 분리하기
y = iris_data.loc[:, 'Name']
x = iris_data.loc[:, ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']]

#학습전용과 테이스 전용 분리하기
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, train_size = 0.8, shuffle= True)

# 그리드 서치에서 사용할 매개 변수
parameters = [{'n_estimators': [1, 10, 100, 1000], 'min_samples_split':[2, 3]}]

# 그리드 서치
kfold_cv = KFold(n_splits=5, shuffle=True)
model = GridSearchCV(RandomForestClassifier(), parameters, cv=kfold_cv)
model.fit(x_train, y_train)
print('최적의 매개 변수 :', model.best_estimator_)

# 최적의 매개 변수로 평가하기
y_pred = model.predict(x_test)
print('최종 정답률:', accuracy_score(y_test, y_pred))