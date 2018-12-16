import numpy as np 
import matplotlib.pyplot as plt


data = np.array([[2.5,0.5,2.2, 1.9,3.1,2.3,2,1,1.5,1.1],
                 [2.4,0.7,2.9,2.2,3,2.7,1.6,1.1,1.6,0.9]]).T

plt.plot(data[:,0], data[:,1], '.', alpha = 0.2)
plt.title("Original Data")
plt.show()

# normalizing before performing PCA
data_norm = data - data.mean(axis=0)


# number of observations/rows
m = data_norm.shape[0]

# vectorized covariance calculation
cov = data_norm.T.dot(data_norm) / (m - 1)

eig_vals, eig_vects = np.linalg.eig(cov)

# fixing column order to mirror Lindsay Smith's output
eig_vals = np.array([eig_vals[1], eig_vals[0]])
eig_vects = np.array([[eig_vects[:,1]], [eig_vects[:,0]]])
print("Eigenvalues:\n",eig_vals)
print("Eigenvectors:\n:", eig_vects, '\n')

# keeping both eigenvectors
final_data_all = eig_vects.T.dot(data_norm.T).reshape(data.shape)
# keeping only the first eigenvector
final_data_one = eig_vects[0].dot(data_norm.T)

print(final_data_all)
print()
print(final_data_one.T)
plt.title("Original Data Rotated about principal Components")
plt.plot(final_data_all[:,0], final_data_all[:,1], '+')
plt.show()


# calculating the projection onto first principal component
RowOriginalData = (eig_vects[0].T.dot(final_data_one)) +\
                  data.mean(axis=0)[:,None]
RowOriginalData = RowOriginalData.T

plt.plot(data[:,0], data[:,1], '.', alpha = 0.2, label = "Original Data")
plt.plot(RowOriginalData[:,0], RowOriginalData[:,1], '-b')
plt.plot(RowOriginalData[:,0], RowOriginalData[:,1], '+', 
         label = "Projection onto 1st PC")
plt.legend()
plt.show()

