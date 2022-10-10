from traceback import print_list
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
            fx = self.function_exercise(xk)
            df = self.dfx(xk, h)
            xk = xk + (alfa * -df)
        return xk

    def function_exercise(self, xk):
        fx = np.exp(-xk) * xk * ((xk ** 2) - xk - 1)
        return fx

    def dfx(self, x, h):
        """ Aproximação da derivada pelo método de diferenças finitas """
        df = (self.function_exercise(x + h) - self.function_exercise(x))/h
        return df


    def task2(self, x0, alfa):
        x = Symbol('x')
        f = exp(-x) * x * ((x ** 2) - x - 1)
        difx = diff(f, x)
        # Apresenta a função derivada
        # print(difx)

        """Aqui já se trata da função derivada"""
        # Converte de uma função simbólica para uma função numérica
        lam_f = lambdify(x, difx)#AKI É O VETOR DE GRADIENTE
        # UTILIZANDO A DERIVADA NA MÃO
        # lam_f = lambdify(x, ((x**3)*exp(x)+2*(x**2)*exp(x)-3*x*exp(x)-exp(x)))
        # Resolve a função para um dado valor de x
        dfdx = lam_f(x0)
        xk = x0 - (alfa * dfdx)  # x_(k+1) <-  xk - alfa * f'(xk)
        return xk, dfdx


    def task3(self, x0, alfa, gmin, kmax):
        xk = x0
        k = 0
        novo=[]
        dfdx=0
        # a função math.fabs é usada para obter o valor absoluto da derivada
        # xk, dfdx = self.task2(xk, gmin)
        # xk=x0
        while k < kmax :#and math.fabs(dfdx) > gmin:
            f = math.exp(-xk) * xk * ((xk ** 2) - xk - 1)
            novo.append([dfdx,xk,f])
            xk, dfdx = self.task2(xk, alfa)
            k += 1
        newnovo = pd.DataFrame(novo,index=None,columns=['Derivada:','X:','Y:']) 
        print(newnovo.to_string())
        newnovo.plot(x ='X:', y='Y:',c='Derivada:', kind = 'scatter',colormap="Reds")
        # newnovo.plot(x ='X:', y='Derivada:', kind = 'line')
        plt.show()
        # fig, ax1 = plt.subplots()
        # ax1.plot(newnovo, y='Y:',x='X:')
        # # ax1.plot(newnovo, x='X:')
        # # fig = px.scatter(newnovo, x='Y:', y='X:', opacity=0.65)#, color="INDICADOR",symbol="INDICADOR")
        # fig.tight_layout()  # otherwise the right y-label is slightly clipped
        # plt.show()
        return print('O valor que minimiza a função f(x) é x=', xk)


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

    def plot_bestfit(self):
        # TODO
        pass
