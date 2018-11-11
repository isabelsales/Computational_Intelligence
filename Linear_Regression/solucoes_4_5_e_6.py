import numpy as np
import matplotlib.pyplot as plt

def rnd(n):
    return np.random.uniform(-1, 1, size=n)

#Início Problema 4
RUNS = 1000
N_train = 1000
E_in_total = 0

for run in range(RUNS):

    #Criação de 1000 pontos aleatórios
    X_train = np.transpose(np.array([np.ones(N_train), rnd(N_train), rnd(N_train)]))
    y_f_train = np.sign(X_train[:, 1] * X_train[:, 1] + X_train[:, 2] * X_train[:, 2] - 0.6)

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
    E_in = sum((y_lr_train != y_f_train)) / N_train
    E_in_total += E_in

E_in_avg = E_in_total / RUNS
print("\n---------------------------------- Resultado Problema 4 ------------------------------------------------")
print("Erro médio de E_in por ", RUNS, "execuções = ", E_in_avg)


#Início Problema 5
#Vetor: (1, x1, x2, x1*x2, x1*x1, x2*x2)
X = X_train

#Nova matriz
X_trans = np.transpose(np.array([np.ones(N_train), X[:,1], X[:,2], X[:,1]*X[:,2], X[:,1]*X[:,1], X[:,2]*X[:,2]]))

#Regressão linear na nova "matriz característica"
X_dagger_trans = np.dot(np.linalg.inv(np.dot(X_trans.T, X_trans)), X_trans.T)
w_lr_trans = np.dot(X_dagger_trans, y_f_train)

#Diferentes hipóteses que são dadas
w_a = np.array([-1, -0.05, 0.08, 0.13, 1.5, 1.5])
w_b = np.array([-1, -0.05, 0.08, 0.13, 1.5, 15])
w_c = np.array([-1, -0.05, 0.08, 0.13, 15, 1.5])
w_d = np.array([-1, -1.5, 0.08, 0.13, 0.05, 0.05])
w_e = np.array([-1, -0.05, 0.08, 1.5, 0.15, 0.15])

#Computar classificações feitas por cada hipótese
y_lr_trans = np.sign(np.dot(X_trans, w_lr_trans))
y_a = np.sign(np.dot(X_trans, w_a))
y_b = np.sign(np.dot(X_trans, w_b))
y_c = np.sign(np.dot(X_trans, w_c))
y_d = np.sign(np.dot(X_trans, w_d))
y_e = np.sign(np.dot(X_trans, w_e))

mismatch_lr_and_a = sum(y_a != y_lr_trans) / N_train                 #Sempre dar restart no Kernel
mismatch_lr_and_b = sum(y_b != y_lr_trans) / N_train
mismatch_lr_and_c = sum(y_c != y_lr_trans) / N_train
mismatch_lr_and_d = sum(y_d != y_lr_trans) / N_train
mismatch_lr_and_e = sum(y_e != y_lr_trans) / N_train


print("\n---------------------------------- Resultado Problema 5 ------------------------------------------------")
print("Incompatibilidade entre LR e a = ", mismatch_lr_and_a)
print("Incompatibilidade entre LR e b = ", mismatch_lr_and_b)
print("Incompatibilidade entre LR e c = ", mismatch_lr_and_c)
print("Incompatibilidade entre LR e d = ", mismatch_lr_and_d)
print("Incompatibilidade entre LR e e = ", mismatch_lr_and_e)

print("O vetor de peso da hipótese = ", w_lr_trans)

#Comparação das previsões feitas por w_lr_trans com aquelas feitas pela função target
print("E_in = ", sum(y_f_train != y_lr_trans) / N_train)


#Início Problema 6
RUNS = 1000
N_test = 1000
E_out_total = 0

for run in range(RUNS):

    #Criação de 1000 pontos aleatórios
    #Vetor
    X_test = np.transpose(np.array([np.ones(N_train), rnd(N_train), rnd(N_train)]))
    y_f_test = np.sign(X_test[:, 1] * X_test[:, 1] + X_test[:, 2] * X_test[:, 2] - 0.6)

    #Escolha 10% = 100 índices aleatórios
    indices = list(range(N_test))
    np.random.shuffle(indices)
    random_indices = indices[:(N_test // 10)]

    #Sinal de aleta no vetor y_f_train
    for idx in random_indices:
        y_f_test[idx] = (-1) * y_f_test[idx]

    #Calcular classificação feita por minha hipótese do problema 5
    #Primeiro criar matriz característica transformada
    X = X_test
    X_trans_test = np.transpose(
        np.array([np.ones(N_test), X[:, 1], X[:, 2], X[:, 1] * X[:, 2], X[:, 1] * X[:, 1], X[:, 2] * X[:, 2]]))
    y_lr_trans_test = np.sign(np.dot(X_trans_test, w_lr_trans))

    #Computar a discordância entre a hipótese e a função alvo
    E_out = sum(y_lr_trans_test != y_f_test) / N_train
    E_out_total += E_out

E_out_avg = E_out_total / RUNS
print("\n---------------------------------- Resultado Problema 6 ------------------------------------------------")
print("Erro médio E_out por", RUNS, "execuções = ", E_out_avg)
