import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import warnings
from sklearn.utils.testing import all_estimators
# sklearn 0.20.3 에서 31개
# sklearn 0.21.2 에서 40개중 4개만 됨

warnings.filterwarnings('ignore')

# 붓꽃 데이터 읽어 들이기
iris_data = pd.read_csv('./data/iris2.csv', encoding='utf-8')

# 붓꽃 데이터를 레이블과 입력 데이터로 분리하기
y = iris_data.loc[:,'Name']
x = iris_data.loc[:,['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']]

# 학습 전용과 테스트 전용 분리하기
warnings.filterwarnings('ignore')
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,
                                                    train_size=0.8, shuffle = True)
parameters = {'n_estimators': [3,5,7,10],
     'max_depth': [3,5,7,9,11],
     'min_samples_leaf' :[2,4,6,8]}   
 # n_estimator :  결정트리의 개수, max_depth : 트리의 깊이, min_samples_leaf : 리프노드가 되기 위한 최소한의 샘플 데이터 수
# min_samples_split : 노드를 분할하기 위한 최소한의 데이터 수 

# 랜덤그리드 서치
kfold_cv = KFold(n_splits = 5, shuffle = True)
model = RandomizedSearchCV(RandomForestClassifier(), parameters, cv = kfold_cv)
model.fit(x_train, y_train)
print("최적의 매개 변수 =", model.best_estimator_)

# 최적의 매개 변수로 평가하기
y_pred = model.predict(x_test)
print("최종 정답률 =", accuracy_score(y_test, y_pred))