import numpy as np
import matplotlib.pyplot as plt

#Criação de 1000 pontos aleatórios
N_train = 1000

def rnd(n):
    return np.random.uniform(-1, 1, size = n)

#Vetor
X_train = np.transpose(np.array([np.ones(N_train), rnd(N_train), rnd(N_train)]))
y_f_train = np.sign(np.multiply(X_train[:,1], X_train[:,1]) + np.multiply(X_train[:,2], X_train[:,2]) - 0.6)

#Escolha 10% = 100 índices aleatórios
indices = list(range(N_train))
np.random.shuffle(indices)
random_indices = indices[:(N_train // 10)]

#Sinal de aleta no vetor y_f_train
for idx in random_indices:
    y_f_train[idx] = (-1) * y_f_train[idx]

#REGRESSÃO LINEAR
X_dagger = np.dot(np.linalg.inv(np.dot(X_train.T, X_train)), X_train.T)
w_lr_train = np.dot(X_dagger, y_f_train)

#Calculo do E_in
y_lr_train = np.sign(np.dot(X_train, w_lr_train))
E_in = sum(y_lr_train != y_f_train)  / N_train

#Gráfico dos pontos classificados
plt.plot(X_train[:,1][y_f_train == 1], X_train[:,2][y_f_train == 1], 'ro')
plt.plot(X_train[:,1][y_f_train == -1], X_train[:,2][y_f_train == -1], 'bo')
plt.xlim(-1,1)
plt.ylim(-1,1)
plt.show()