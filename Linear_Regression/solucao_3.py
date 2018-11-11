import numpy as np
import matplotlib.pyplot as plt

def rnd(n):
    return np.random.uniform(-1, 1, size=n)

N_amostra = 10
RUNS = 1000
iterations_total = 0

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

    #Usar o PLA até que todos os pontos sejam separados
    t = 0  #contar o numero de ietracoes do PLA
    w_h = np.copy(w_lr)

    while True:
        #Iniciar PLA
        y_h = np.sign(np.dot(X, w_h))  #Classificação da hipóteses
        comp = (y_h != y_f)  #Comparar a classificação com dados reais da função alvo
        wrong = np.where(comp)[0]  #Índices de pontos com classificação errada pela hipótese h

        if wrong.size == 0:
            break

        rnd_choice = np.random.choice(wrong)  #Escolha um ponto aleatório mal classificado

        #Escolha um vetor de peso pontual aleatoriamente classificado erroneamente (nova hipótese):
        w_h = w_h + y_f[rnd_choice] * np.transpose(X[rnd_choice])
        t += 1

    iterations_total += t

iterations_avg = iterations_total / RUNS
print("\n---------------------------------- Resultado Problema 3 ------------------------------------------------")
print("Média de Iterações do PLA por", RUNS, "execuções: ", iterations_avg)
