import numpy as np
import matplotlib.pyplot as plt

def rnd(n):
    return np.random.uniform(-1, 1, size = n)

#Escolha de dois pontos aleatórios A, B em [-1,1] x [-1,1]
A = rnd(2)
B = rnd(2)

#A linha pode ser descrita por y = m * x + b onde m é a inclinação
m = (B[1] - A[1]) / (B[0] - A[0])
b = B[1] - m * B[0]
w_f = np.array([b, m, -1])

#Escolha N pontos de dados (x, y) uniformemente da caixa [-1,1] x [-1,1]
N = 100
X = np.transpose(np.array([np.ones(N), rnd(N), rnd(N)])) #input

#Classificação dos pontos
y_f = np.sign(np.dot(X, w_f))

#Traçar pontos e colorir de acordo com sua classificação
plt.plot(X[:,1][y_f == 1], X[:,2][y_f == 1], 'ro')
plt.plot(X[:,1][y_f == -1], X[:,2][y_f == -1], 'bo')

#Plot line
#Criação de alguns pontos de dados na linha (para o gráfico) usando a forma vetorial paramétrica de uma linha
#Linha (t) = A + t * d, onde A é um ponto na linha, d o vetor direcional e t o parâmetro
d = B - A
line_x = [A[0] + t * d[0] for t in range(-10,10)]
line_y = [A[1] + t * d[1] for t in range(-10,10)]
plt.plot(line_x, line_y)

#Traçar os dois pontos que definem a linha
plt.plot(A[0], A[1], 'go')
plt.plot(B[0], B[1], 'go')

#Definição dos intervalos para o eixo xey para exibir a caixa [-1,1] x [-1,1]
plt.ylim(-1,1)
plt.xlim(-1,1)
plt.show()

#REGRESSÃO LINEAR
X_dagger = np.dot(np.linalg.inv(np.dot(X.T, X)), X.T)
w_lr = np.dot(X_dagger, y_f)

#Classificação do enredo segundo w encontrado por regressão linear
#Mostra que alguns dos pontos são classificados erroneamente
y_lr = np.sign(np.dot(X, w_lr))

#Traçar pontos e colori-los de acordo com sua classificação
plt.plot(X[:,1][y_lr == 1], X[:,2][y_lr == 1], 'ro')
plt.plot(X[:,1][y_lr == -1], X[:,2][y_lr == -1], 'bo')

#Traçar a linha de classificação correta (função de destino)
plt.plot(line_x, line_y, 'g')
plt.ylim(-1,1)
plt.xlim(-1,1)
plt.show()