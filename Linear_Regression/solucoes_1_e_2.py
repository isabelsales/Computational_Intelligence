import numpy as np
import matplotlib.pyplot as plt

def rnd(n):
    return np.random.uniform(-1, 1, size=n)

#Repita o experimento 1000 vezes
RUNS = 1000
N_amostra = 100
E_in_total = 0
E_out_total = 0
N_test = 1000

for run in range(RUNS):
    #Escolha dois pontos aleatórios A, B em [-1,1] x [-1,1]
    A = rnd(2)
    B = rnd(2)

    #A linha pode ser descrita por y = m * x + b onde m é a inclinação
    m = (B[1] - A[1]) / (B[0] - A[0])
    b = B[1] - m * B[0]
    w_f = np.array([b, m, -1])

    #Criar N pontos de dados (x, y) da função de destino
    X = np.transpose(np.array([np.ones(N_amostra), rnd(N_amostra), rnd(N_amostra)]))  # input
    y_f = np.sign(np.dot(X, w_f))

    #REGRESSÃO LINEAR
    X_dagger = np.dot(np.linalg.inv(np.dot(X.T, X)), X.T)
    w_lr = np.dot(X_dagger, y_f)

    #Classificação de acordo com w encontrada por regressão linear
    y_lr = np.sign(np.dot(X, w_lr))

    #Erro E_in
    E_in = sum(y_lr != y_f) / N_amostra
    E_in_total += E_in

    #1000 pontos de teste (fora dos pontos de amostra) e contar o desacordo
    #entre y_f_test e y_lr_test
    X_test = np.transpose(np.array([np.ones(N_test), rnd(N_test), rnd(N_test)]))
    y_f_test = np.sign(np.dot(X_test, w_f))
    y_lr_test = np.sign(np.dot(X_test, w_lr))

    E_out = sum(y_lr_test != y_f_test) / N_test
    E_out_total += E_out

#Média de execuções de E_in
E_in_avg = E_in_total / RUNS
print("\n---------------------------------- Resultado Problema 1 ------------------------------------------------")
print("Média E_in", RUNS, " execuções:", E_in_avg)

#Média de execuções de E_out
E_out_avg = E_out_total / RUNS
print("\n---------------------------------- Resultado Problema 2 ------------------------------------------------")
print("Média E_out", RUNS, " execuções:", E_out_avg)