import random
import numpy as np
import math

def problem_4_e_5():
    execucoes = 100
    E_out_total = 0
    epoca_total = 0

    for n in range(execucoes):
        #Conjunto de treino com N = 100 pontos através da linha de separação

        #Escolher dois pontos aleatórios A, B em [-1,1] x [-1,1]
        A = np.random.uniform(-1, 1, 2)
        B = np.random.uniform(-1, 1, 2)

        #A linha descrita por y = m * x + b onde m é a inclinação
        m = (B[1] - A[1]) / (B[0] - A[0])
        b = B[1] - m * B[0]
        w_f = np.array([b, m, -1])

        # ----------------------------------------------------------------------------------------

        #Escolher N pontos de dados (x, y) uniformemente de [-1,1] x [-1,1]
        N = 100
        x1 = np.random.uniform(-1, 1, N)
        x2 = np.random.uniform(-1, 1, N)

        X = np.transpose(np.array([np.ones(N), x1, x2]))  # input

        #Classificação dos pontos
        y_f = np.sign(np.dot(X, w_f))

        # ----------------------------------------------------------------------------------------

        #Executa a Regressão Logística
        #Inicializar pesos para hipóteses com zeros
        eta = 0.01
        w_g = np.zeros(3)  #Vetor de peso para a hipótese g

        #Inicializa as iterações
        for t in range(10 ** 5):

            #Criação de permutação de pontos de dados
            indices = list(range(N))
            random.shuffle(indices)
            w_old = w_g

            #Laço de repetição: para cada época
            for i in indices:
                xn = X[i, :]  #escolha um ponto
                yn = y_f[i]
                delta_w = -yn * xn / (1 + math.exp(yn * np.dot(w_g.T, xn)))

                #Atualizar w
                w_g = w_g - eta * delta_w

            #Verificar quanto w_g mudou
            #print("t = ", t, "    Diferença de w = ", np.linalg.norm(w_g - w_old))
            if np.linalg.norm(w_g - w_old) < 0.01:
                break

        epoca_total += t

        #1000 pontos de teste para calcular o E_out
        N_test = 1000
        x1_test = np.random.uniform(-1, 1, N_test)  #1000 pontos
        x2_test = np.random.uniform(-1, 1, N_test)
        X_test = np.array([np.ones(N_test), x1_test, x2_test]).T

        y_f_test = np.sign(np.dot(X_test, w_f))  #classificação

        #Calculo do E_out (entropia cruzada)
        E_out = 0
        for i in range(N_test):
            E_out += math.log(1 + math.exp(-y_f_test[i] * np.dot(X_test[i, :], w_g)))

        E_out_total += (E_out / N_test)

    E_out_avg = E_out_total / execucoes
    epoca_avg = epoca_total / execucoes

    return (E_out_avg, epoca_avg)

E_out_avg, epoca_avg = problem_4_e_5()

print("\n---------------------------------- Resultado Problema 4 ------------------------------------------------")
print("Erro médio de entropia cruzada E_out para 100 execuções: ", E_out_avg)

print("\n---------------------------------- Resultado Problema 5 ------------------------------------------------")
print("Número médio de épocas: ", epoca_avg)
