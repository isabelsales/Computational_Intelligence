import numpy as np
import matplotlib.pyplot as plt


def rnd(n):
    return np.random.uniform(-1, 1, size=n)


def get_E_out_iteracoes(N_pontos):
    execucoes = 1000
    total_iteracoes = 0
    incompatibilidade_total = 0
    N_teste = 1000
    N = N_pontos
    d = 2

    # ------------------------------------------------------------------------------------------------------------------------

    for exe in range(execucoes):

        #Escolha de dois pontos aleatórios A, B em [-1,1] x [-1,1]/ para d=2
        A = rnd(d)
        B = rnd(d)

        #A reta pode ser descrita por y = m * x + b onde m é a inclinação
        m = (B[1] - A[1]) / (B[0] - A[0])
        b = B[1] - m * B[0]
        w_f = np.array([b, m, -1])

        # ------------------------------------------------------------------------------------------------------------------------

        #Criação de N pontos de dados (x, y) da função de destino
        X = np.transpose(np.array([np.ones(N), rnd(N), rnd(N)]))  #input
        y_f = np.sign(np.dot(X, w_f))  #output

        # ------------------------------------------------------------------------------------------------------------------------

        #Escolha da a hipótese h
        w_h = np.zeros(3)  #Inicializa o vetor de peso para a hipótese h
        t = 0  #Conta o número de iterações no PLA


        while True:
            #Inicia o PLA
            y_h = np.sign(np.dot(X, w_h))  #Classificação por hipótese
            comp = (y_h != y_f)  #Comparação da classificação com dados reais da função alvo
            erros = np.where(comp)[0]  #Índices de pontos com classificação errada pela hipótese h

            if erros.size == 0:
                break

            ponto_aleatorio = np.random.choice(erros)  #Escolha de um ponto aleatório mal classificado

            #Atualização do vetor de peso (nova hipótese):
            w_h = w_h + y_f[ponto_aleatorio] * np.transpose(X[ponto_aleatorio])
            t += 1

        total_iteracoes += t

        # ------------------------------------------------------------------------------------------------------------------------

        #Cálculo do erro
        #Criação de dados "fora" de dados de treinamento
        test_x0 = np.random.uniform(-1, 1, N_teste)
        test_x1 = np.random.uniform(-1, 1, N_teste)

        X_teste = np.array([np.ones(N_teste), test_x0, test_x1]).T

        y_target = np.sign(np.dot(X_teste, w_f))
        y_hipotese = np.sign(np.dot(X_teste, w_h))

        relacao_incompatibilidade = ((y_target != y_hipotese).sum()) / N_teste
        incompatibilidade_total += relacao_incompatibilidade

    # ------------------------------------------------------------------------------------------------------------------------

    print( "\n----------------------------------------- Relatório Detalhado -----------------------------------------------\n")

    print("Tamanho dos dados de treinamento: N = ", N, "pontos")

    iteracoes = total_iteracoes / execucoes
    print("\nNúmero médio de iterações de PLA durante", execucoes, "Iterações: t_avg = ", iteracoes)

    incompatibilidade = incompatibilidade_total / execucoes
    print("\nRazão média para a incompatibilidade entre f (x) e (x) fora dos dados de treinamento:")
    print("P(f(x)!=h(x)) = E_out = ", incompatibilidade)


    return (incompatibilidade, iteracoes)

dados = [10,100]
E_out, iteracoes_dados = [], []

for tamanho in dados:
    incompatibilidade, iteracoes = get_E_out_iteracoes(tamanho)
    E_out.append(incompatibilidade)
    iteracoes_dados.append(iteracoes)


plt.figure(1)
plt.plot(dados, E_out, 'ro')
plt.ylabel("E_out")
plt.xlabel("Dados de treinamento")
plt.savefig('E_out_vs_dados_treinamento.png')

plt.figure(2)
plt.plot(dados, iteracoes_dados, 'bo')
plt.ylabel("Iterações PLA")
plt.xlabel("Dados de treinamento")
plt.savefig('iteracoes_PLA_vs_dados_treinamento.png')

plt.show()

#Relatório
print("\n----------------------------------------------- Relatório Final ---------------------------------------------\n")
print("Dados de treinamento ", dados)
print("Iterações do PLA: ", iteracoes_dados)
print("P(f(x)!=h(x)) = E_out = ", E_out)
