import numpy as np
from src.problem.problem_interface import ProblemInterface
from sympy import diff, Symbol
from sympy import lambdify
from sympy import sin, cos, exp


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


    def task2(self, x0, alfa):
        x = Symbol('x')
        f = exp(x) * x * ((x ** 2) - x - 1)
        difx = diff(f, x)
        # Apresenta a função derivada
        # print(difx)

        """Aqui já se trata da função derivada"""
        # Converte de uma função simbólica para uma função numérica
        lam_f = lambdify(x, difx)
        # Resolve a função para um dado valor de x
        xk = x0 + (alfa * -lam_f(x0))
        return xk


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
