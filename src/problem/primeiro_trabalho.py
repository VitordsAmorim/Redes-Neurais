import numpy as np
from src.problem.problem_interface import ProblemInterface
# import matplotlib.pyplot as plt
import math



class PrimeiroTrabalho(ProblemInterface):

    def __init__(self, fname):
        # load dataset
        """self.filename = fname
        with open(fname, "r") as f:
            lines = f.readlines()

        lines = [l.rstrip().rsplit() for l in lines]

        # Convert the list of list into a numpy matrix of integers.
        lines = np.array(lines).astype(np.int)
        self.x = lines[:, :-1]
        self.y = lines[:, -1:]"""
        pass

    # """   Dada a função f (x) = exp(−x) ∗ x ∗ (x ∗ ∗2 − x − 1), o valor inicial x0 = 1, e a taxa
    # de aprendizagem alfa= 0.1, calcule o próximo valor x1 pela descida de gradiente, usando
    # uma aproximação da derivada pelo método de diferenças finitas (eq. 3.1), com um passo
    # h = 0.01, diferenciação numérica. """
    def task1(self, x0, alfa, h):
        xk = x0
        for k in range(1):
            fx = self.function_exercise(xk)
            df = self.dfx(xk, h)
            xk = xk + (alfa * -df)
        return xk

    def function_exercise(self, xk):
        fx = np.exp(xk) * xk * ((xk ** 2) - xk - 1)
        return fx

    def dfx(self, x, h):
        """ Aproximação da derivada pelo método de diferenças finitas """
        df = (self.function_exercise(x + h) - self.function_exercise(x))/h
        return df


    # """"    Calcule x1 , usando a derivada f′(x)
    #
    #     Ajuda: Para calcular a derivada analiticamente, use a função ’diff’ do pacote ’sympy’ do Python.
    #     A função ’lambdify’ converte de uma função simbólica para uma função numérica."""
    def task2(self):
        # TODO
        return print('task2')


    # """   Usando o método da descida de gradiente, tente aproximar o mínimo de f,
    #           com o valor inicial x_0=3 e taxa de aprendizagem alpha=0.1.
    #
    #           Defina as duas condições de parada do Algoritmo 3, com gmin = 0.1, k_max = 20
    #
    #           Qual é o valor de um mínimo obtido?
    #
    #           Usando um gráfico bidimensional, eixo x = x, eixo y = f(x), marque os valores da aproximação
    #           x_0 --> x_1 --> x_2 --> x_3 --> x_4 --> x_5 ...
    #           e f(x_0) --> f(x_1) --> f(x_2) --> f(x_3) --> f(x_4) --> f(x_5) ...
    #           e f'(x_0) --> f'(x_1) --> f'(x_2) --> f'(x_3) --> f'(x_4) --> f'(x_5) ..."""
    def task3(self):
        # TODO
        return print('task3')


    # (a) Use o método de diferenças finitas para aproximar o gradiente.
    def task4_a(self):
        # TODO
        return print('task4_a')

    # (b) Use o gradiente explícito no algoritmo da descida de gradiente
    def task4_b(self):
        # TODO
        return print('task4_b')

    # (c) Desenhe a trajetória de x k no plano (x 1 , x 2 ),
    # e o valor da função correspondente de f(x1 , x2) no gráfico 3-D.
    def task4_c(self):
        # TODO
        return print('task4_c')

    def plot(self):
        # TODO
        pass

    def plot_bestfit(self):
        # TODO
        pass
