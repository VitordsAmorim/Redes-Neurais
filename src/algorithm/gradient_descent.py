# Call the functions in the Descida_de_gradiante file
import numpy as np
from sympy import diff, Symbol

def gradient_descent(problem):

    """ Dada a função f (x) = exp(−x) ∗ x ∗ (x ∗ ∗2 − x − 1), o valor inicial x0 = 1, e a taxa
    de aprendizagem alfa= 0.1, calcule o próximo valor x1 pela descida de gradiente, usando
    uma aproximação da derivada pelo método de diferenças finitas (eq. 3.1), com um passo
    h = 0.01, diferenciação numérica. """
    x1 = problem.task1(x0=1, alfa=0.1, h=0.01)
    print("Tarefa 1\nx0: %.4f, x1: %.4f\n" % (1, x1))


    """"    Calcule x1 , usando a derivada f′(x) 
    Ajuda: Para calcular a derivada analiticamente, use a função ’diff’ do pacote ’sympy’ do Python.
    A função ’lambdify’ converte de uma função simbólica para uma função numérica. """
    dx1, _ = problem.task2(x0=1, alfa=0.1)
    print("Tarefa 2\nx0: %.4f, x1: %.4f\n" % (1, dx1))


    """   
        Usando o método da descida de gradiente, tente aproximar o mínimo de f,
        com o valor inicial x_0 = 3 e taxa de aprendizagem alpha = 0.1. 
        Defina as duas condições de parada do Algoritmo 3, com gmin = 0.1, k_max = 20
        Qual é o valor de um mínimo obtido?        
        Usando um gráfico bidimensional, eixo x = x, eixo y = f(x), marque os valores da aproximação
        x_0 --> x_1 --> x_2 --> x_3 --> x_4 --> x_5 ...
        e f(x_0) --> f(x_1) --> f(x_2) --> f(x_3) --> f(x_4) --> f(x_5) ...
        e f'(x_0) --> f'(x_1) --> f'(x_2) --> f'(x_3) --> f'(x_4) --> f'(x_5) ..."""
    # problem.task3(x0=3, alfa=0.1, gmin=0.1, kmax=100)


    """
        Plot the graph representing the function in question 1 and 2.
        It was not requested by the activity, but I believe it is interesting
         to know a little more about the feature of the function"""

    """ 4. Determine pelo menos dois mínimos da função bidimensional:
        f(x1, x2) = (4 - 2.1 * x1**2 + x1**3 / 3 ) * x1**3 + x1 * x2 + (-4 + 4 * x2**2) * x2**2
        
        (a) Use o método de diferenças finitas para aproximar o gradiente.
    """

    print("Questão 4")
    print("f(x1, x2) = (4 - 2.1 * x1**2 + x1**3 / 3 ) * x1**3 + x1 * x2 + (-4 + 4 * x2**2) * x2**2\n")

    point = [1, 1]  # Starting points
    fx1x2_mdf, fx1x2_dliteral = problem.gradiente_duas_variaveis(point, alpha=0.05, gmin=0.1, kmax=100, h=0.01)
    print("       X1           X2        f(x1,x2)  Ponto Inicial")
    print(fx1x2_mdf, point, ": Método das diferenças finitas")
    print(fx1x2_dliteral, point, ": Método Literal para o cálculo da derivada\n")

    point = [-0.5, -0.5]
    fx1x2_mdf, fx1x2_dliteral = problem.gradiente_duas_variaveis(point, alpha=0.05, gmin=0.1, kmax=100, h=0.01)
    print(fx1x2_mdf, point, ": Método das diferenças finitas")
    print(fx1x2_dliteral, point, ": Método Literal para o cálculo da derivada")

    # colocar o código acima num for, para ir fazendo o calculo de acordo
    # com novas posições iniciais
    # conseguir salvar os valores de casa iteração
    # depois plotar um gráfico com vários pontos iniciasi percorrendo ao mesmo tempo.


    # (c) Desenhe a trajetória de x k no plano (x 1 , x 2 ),
    # e o valor da função correspondente de f(x1 , x2) no gráfico 3-D.
    #problem.task4_c()

    """Essa função é pesada, vale a pena deixá-la como comentário"""
    #problem.make_gif(frame_folder='Image/gif/')

    return [print("Fim")]
