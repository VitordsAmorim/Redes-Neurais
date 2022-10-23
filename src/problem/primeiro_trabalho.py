import numpy as np
from src.problem.problem_interface import ProblemInterface
from sympy import diff, Symbol
from sympy import lambdify
from sympy import sin, cos, exp
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import math
from mpl_toolkits import mplot3d




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
        novo, flist = [], []

        _, dfdx = self.task2(xk, gmin)

        """ math.fabs is used to get the absolute value of the derivative """
        while k < kmax and math.fabs(dfdx) > gmin:
            f = math.exp(-xk) * xk * ((xk ** 2) - xk - 1)
            novo.append([dfdx,xk,f])
            flist.append(f)
            xk, dfdx = self.task2(xk, alfa)
            k += 1

        self.plot(database=novo)
        self.plot_3(database=novo)
        min_pos = flist.index(min(flist))
        messenger = "Tarefa 3.a \nThe minimum value found for the function f(x) is y= %0.4f from x= %0.4f"
        return print( messenger % (min(flist), novo[min_pos][1])+'\n')


    # (a) Use o método de diferenças finitas para aproximar o gradiente.
    def task4_a(self, Xk, h):
        X = Xk
        dfdx1 = self.dfx_4(X[0], X[1], h, 0)
        dfdx2 = self.dfx_4(X[0], X[1], 0, h)
        return dfdx1, dfdx2

    def dfx_4(self, x1, x2, h1, h2):
        """ Approximation of the derivative by the finite difference method """
        """ In this case, for two variables"""
        df = (self.task4_function(x1 + h1, x2 + h2) - self.task4_function(x1, x2))/(h1 + h2)
        return df

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
        # self.plot_question4()
        return print('task4_c')

    def plot_question4(self):

        x1 = np.linspace(-4, 4, 100)
        x2 = np.linspace(-4, 4, 100)
        x1, x2 = np.meshgrid(x1, x2)
        Z = self.task4_function(x1, x2)

        fig = plt.figure(figsize=(5, 5))
        ax = plt.axes(projection='3d')
        #ax.contour3D(x1, x2, Z, 50, cmap='coolwarm')
        ax.plot_surface(x1, x2, Z, cmap="coolwarm", lw=0.5, rstride=1, cstride=1)

        #ax.set_title('Gráfico 3D', fontsize=18)
        ax.set_xlabel(r'X1', fontsize=10)
        ax.set_ylabel(r'X2', fontsize=10)
        ax.set_zlabel(r'f(X1, X2)', fontsize=10)
        ax.view_init(70, 35)
        plt.show()
        pass

    def plot(self, database):
        """Generate the points from the function"""
        y, xaxis = [], []
        for x in np.arange(-10.0, 40.0, 0.05):
            f = exp(-x) * x * ((x ** 2) - x - 1)
            y.append(f)
            xaxis.append(x)

        plt.subplots()
        plt.plot(xaxis, y, label= r'$f ~ (x) = e^{-x} (x^3- x^2 - x)$')
        plt.xlabel(r'$x$')
        plt.ylabel(r'$f ~ (x)$')

        plt.grid(True, color='gray', linestyle = '--', linewidth = 0.5)
        plt.legend(title='Function')
        plt.text(-0.6, 0.6,  r'$x_0 = 3$    $\alpha = 0.1$    $K_{max} = 100$',
                 bbox={'facecolor': 'white', 'edgecolor': 'k', 'boxstyle': 'round, pad=1'})
        plt.title('Function representation')

        """ limit of x and y axis"""
        plt.xlim(-1, 4)
        plt.ylim(-0.5, 1)

        plt.savefig("Image/first_function.png")
        plt.show()
        plt.close()
        pass

    def plot_3(self, database):
        newnovo = pd.DataFrame(database,index=None,columns=['Derivative','X','$f~(x)$'])
        print(newnovo)
        newnovo.plot(x ='X', y='$f~(x)$',c='Derivative', kind = 'scatter',colormap="Blues")
        plt.title("Gradient descent")
        plt.show()
        pass
