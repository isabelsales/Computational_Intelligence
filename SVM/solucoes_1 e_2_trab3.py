import math
import numpy as np

e = math.e

def E(u, v):
    return (u * e ** v - 2 * v * e ** (-u)) ** 2

def problem_1_2_trab3():
    #Iterações exigidas
    iteracoes = -1

    #Inicia (u, v) = (1,1)
    x = [1, 1]
    eta = 0.1

    #Criação de um laço de repetição
    for t in range(1, 10 ** 5):

        #Descompactar os valores na lista x
        u, v = x

        #Cálculo de gradiente
        dE_du = 2 * (u * e ** v - 2 * v * e ** (-u)) * (e ** v + 2 * v * e ** (-u))
        dE_dv = 2 * (u * e ** v - 2 * v * e ** (-u)) * (u * e ** v - 2 * e ** (-u))
        grad = np.array([dE_du, dE_dv])

        #Atualiza posições
        x = x - eta * grad

        #Iterações necessárias
        iteracoes = t

        #Armazenar posição atual x como final_uv
        final_uv = x

        #Parar se E cair abaixo de 10 ^ (- 14)
        if E(x[0], x[1]) < 10 ** (-14):
            break

    return iteracoes, final_uv

iteracoes, final_uv = problem_1_2_trab3()
print("\n---------------------------------- Resultado Problema 1 ------------------------------------------------")
print("Quantidade de iterações que encontra o erro E(u, v) abaixo de (10)^14 pela primeira vez: ", iteracoes)
print("Final(u,v) = ", final_uv)

# ----------------------------------------------------------------------------------------------------------------

#Calcular qual dos seguintes pontos é o mais próximo do final (u, v)
L = [(1.000, 1.000), (0.713, 0.045), (0.016, 0.112), (-0.083, 0.029), (0.045, 0.024)]

min_dist = 2 ** 64
min_ponto = None

print("\n---------------------------------- Resultado Problema 2 ------------------------------------------------")
print("Distância dos pontos dados até a final (u, v):")

for ponto in L:
    x = np.array(ponto)
    distance = np.linalg.norm(final_uv - x)
    print("Ponto x = ", x, " => distância = ", distance)
    if distance < min_dist:
        min_dist = distance
        min_ponto = x

print("\nO ponto com distância mínima até a final (u, v) é: ", min_ponto)
