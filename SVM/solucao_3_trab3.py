import math
import numpy as np

e = math.e

def E(u, v):
    return (u * e ** v - 2 * v * e ** (-u)) ** 2

def problem_3_trab3():
    #Iterações exigidas
    num_iteracoes = 15

    #Inicia (u,v) = (1,1)
    x = [1, 1]
    eta = 0.1

    #Criação de laço de repetição
    for t in range(num_iteracoes):
        #Primeiro passo: Calcular o gradiente na direção u
        u, v = x
        dE_du = 2 * (u * e ** v - 2 * v * e ** (-u)) * (e ** v + 2 * v * e ** (-u))
        grad = np.array([dE_du, 0])

        #Atualiza a posição somente na direção u
        x = x - eta * grad

        #Segundo passo: Calcular o gradiente na direção v
        u, v = x
        dE_dv = 2 * (u * e ** v - 2 * v * e ** (-u)) * (u * e ** v - 2 * e ** (-u))
        grad = np.array([0, dE_dv])

        #Atualiza a posição somente na direção u
        x = x - eta * grad

        #Iterações necessárias
        iteracoes = t

    #Erro final
    erro_final = E(x[0], x[1])
    return erro_final

print("\n---------------------------------- Resultado Problema 3 ------------------------------------------------")
print("Coordenada descendente:")
print("Valor de erro E(u, v) mais próximo após 15 iterações completas:", problem_3_trab3())

# -----------------------------------------------------------------------------------------------------------------
