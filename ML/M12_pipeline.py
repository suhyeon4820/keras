import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import warnings
from sklearn.utils.testing import all_estimators
from sklearn.svm import SVC

# sklearn 0.20.3 에서 31개
# sklearn 0.21.2 에서 40개중 4개만 됨

warnings.filterwarnings('ignore')

# 붓꽃 데이터 읽어 들이기
iris_data = pd.read_csv('./data/iris2.csv', encoding='utf-8')

# 붓꽃 데이터를 레이블과 입력 데이터로 분리하기
y = iris_data.loc[:,'Name']
x = iris_data.loc[:,['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']]
print(y.shape) #(150, )
print(x.shape) #(150, 4)

# 학습 전용과 테스트 전용 분리하기
warnings.filterwarnings('ignore')
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, train_size=0.7, shuffle = True)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

# 모델과 전처리 동시에 진행
model = Pipeline( [('scalar', MinMaxScaler()), ('svm', SVC())] )

# 훈련
model.fit(x_train, y_train)

print('테스트 점수 :', model.score(x_test, y_test))
