import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
import warnings
from sklearn.utils.testing import all_estimators
# sklearn 0.20.3에서 31개

import warnings
warnings.filterwarnings('ignore')

# 붓꽃 데이터 읽어 들이기
iris_data = pd.read_csv('./data/iris2.csv', encoding='utf-8')

# 붗꽃 데이터를 레이블과 입력 데이터로 분리하기
y = iris_data.loc[:, 'Name']
x = iris_data.loc[:, ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']]

# classifier 알고리즘 모두 추출하기(1)
warnings.filterwarnings('ignore')
allAlgorithms = all_estimators(type_filter='classifier')

kfold_cv = KFold(n_splits=10, shuffle=True)
for(name, algorithm) in allAlgorithms:
    # 각 알고리즘 객체 생성하기 (2)
    clf = algorithm()
    if hasattr(clf, 'score'):
        scores = cross_val_score(clf, x, y, cv=kfold_cv) #cv=자르는 수(5개)
        # scores.fit()
        print(name, '의 정답률:')
        print(scores)

# hasattr