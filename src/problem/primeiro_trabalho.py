import numpy as np
from src.problem.problem_interface import ProblemInterface
from sympy import diff, Symbol
from sympy import lambdify
from sympy import sin, cos, exp
import matplotlib.pyplot as plt
import pandas as pd





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
            # fx = self.function_exercise(xk)
            df = self.dfx(xk, h)
            xk = xk - (alfa * df)
        return xk

    def function_exercise(self, xk):
        fx = np.exp(-xk) * xk * ((xk ** 2) - xk - 1)
        return fx

    def dfx(self, x, h):
        """ Approximation of the derivative by the finite difference method """
        df = (self.function_exercise(x + h) - self.function_exercise(x))/h
        return df


    def task2(self, x0, alfa):
        x = Symbol('x')
        f = exp(-x) * x * ((x ** 2) - x - 1)
        difx = diff(f, x)  # dfx is the derivative of the function

        """ Convert from a symbolic function to a numeric function """
        lam_f = lambdify(x, difx)
        dfdx = lam_f(x0) # Solve the function for a given value of x
        xk = x0 - (alfa * dfdx)  # x_(k+1) <-  xk - alfa * f'(xk)
        return xk, dfdx


    def task3(self, x0, alfa, gmin, kmax):
        xk = x0
        k, dfdx  = 0, 0
        novo = []

        _, dfdx = self.task2(xk, gmin)
        """ math.fabs is used to get the absolute value of the derivative """
        while k < kmax and math.fabs(dfdx) > gmin:
            f = math.exp(-xk) * xk * ((xk ** 2) - xk - 1)
            novo.append([dfdx,xk,f])
            xk, dfdx = self.task2(xk, alfa)
            k += 1

        # ATENÇÃO: revisar o valor de xk, que não está batendo com o valor da tabela apresentada
        # pode ser que o xk aqui, seja o próximo valor de x
        self.plot_3(database=novo)
        messenger = "The minimum value found for the function f(x) is y= %0.4f x= %0.4f"
        return print( messenger % (f, xk))


    # (a) Use o método de diferenças finitas para aproximar o gradiente.
    def task4_a(self):
        # TODO
        return print('task4_a')

    def task4_function(self, x1, x2):
        fx1x2 = (4 - 2.1 * x1 ** 2 + x1 ** 3 / 3) * x1 ** 3 + x1 * x2 + (-4 + 4 * x2 ** 2) * x2 ** 2
        return fx1x2

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
        y = []
        xaxis = []
        for x in np.arange(-10.0, 40.0, 0.1):
            f = exp(-x) * x * ((x ** 2) - x - 1)
            y.append(f)
            xaxis.append(x)

        plt.plot(xaxis, y)
        plt.xlim(-10, 10)
        plt.ylim(-20, 10)
        plt.show()
        pass

    def plot_3(self, database):
        newnovo = pd.DataFrame(database,index=None,columns=['Derivada:','X:','Y:'])
        print(newnovo.to_string())
        newnovo.plot(x ='X:', y='Y:',c='Derivada:', kind = 'scatter',colormap="Reds")
        plt.show()
        pass
