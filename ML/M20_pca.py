from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
cancer = load_breast_cancer()

scaler = StandardScaler()
scaler.fit(cancer.data)
X_scaled = scaler.transform(cancer.data)

from sklearn.decomposition import PCA
pca = PCA(n_components=10)
pca.fit(X_scaled)

X_pca = pca.transform(X_scaled)
print('원본 데이터 형태 :', X_scaled.shape) #(569, 30)
print('축소된 데이터 형태 :', X_pca.shape)  #(569, 10) 차원이 n_components값으로 축소(압축)된거지 실제 컬럼이 아님
