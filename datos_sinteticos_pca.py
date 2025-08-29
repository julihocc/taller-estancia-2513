import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

# Fijar la semilla para reproducibilidad
np.random.seed(42)

# Generar datos sintéticos en 5D
n_samples = 200
mean = np.zeros(5)
cov = np.array([
	[3, 1.5, 1, 0.5, 0.2],
	[1.5, 2, 0.8, 0.3, 0.1],
	[1, 0.8, 1.5, 0.2, 0.1],
	[0.5, 0.3, 0.2, 1, 0.05],
	[0.2, 0.1, 0.1, 0.05, 0.5]
])
X = np.random.multivariate_normal(mean, cov, n_samples)

# Aplicar PCA y reducir a 3 componentes
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X)

# Visualizar los 3 componentes principales en 3D
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], alpha=0.7)
ax.set_title('Datos proyectados en los 3 componentes principales')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
plt.tight_layout()
plt.show()
