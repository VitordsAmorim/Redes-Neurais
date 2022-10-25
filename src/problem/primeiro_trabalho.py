import numpy as np
from src.problem.problem_interface import ProblemInterface
from sympy import diff, Symbol, lambdify, sin, cos, exp
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import math
from mpl_toolkits import mplot3d
import glob
from PIL import Image


class PrimeiroTrabalho(ProblemInterface):

    def __init__(self, fname):
        pass

    def task1(self, x0, alfa, h):
        xk = x0
        for k in range(1):
            df = self.dfx(xk, h)
            xk = xk - (alfa * df)
        return xk

    def function_exercise(self, xk):
        fx = np.exp(-xk) * xk * ((xk ** 2) - xk - 1)
        return fx

    def dfx(self, x, h):
        """ Approximation of the derivative by the finite difference method
            for one variable """
        df = (self.function_exercise(x + h) - self.function_exercise(x)) / h
        return df

    def task2(self, x0, alfa):
        """ This function implements algorithm 3, presented in the book,
         "Aprendizagem de Máquina", page 90, author: Thomas Rauber.

         Algoritmo 3: Descida de Gradiente, Uma Dimensão
         """
        x = Symbol('x')
        f = exp(-x) * x * ((x ** 2) - x - 1)
        difx = diff(f, x)  # dfx is the derivative of the function

        """ Convert from a symbolic function to a numeric function """
        lam_f = lambdify(x, difx)
        dfdx = lam_f(x0)  # Solve the function for a given value of x
        xk = x0 - (alfa * dfdx)
        return xk, dfdx


    def task3(self, x0, alfa, gmin, kmax):
        """ k    -> represents the iterations
            xk   -> represents the initial value of x
            dfdx -> needed to be calculated before the loop in order for the stopping criterion to be set.
        """
        xk, k, dfdx = x0, 0, 0
        novo = []
        _, dfdx = self.task2(xk, alfa)

        """ math.fabs is used to get the absolute value of the derivative
            kmax, gmin -> they are the hyperparameters(in this case: stopping criterion)
        """
        while k < kmax and math.fabs(dfdx) > gmin:
            f = math.exp(-xk) * xk * ((xk ** 2) - xk - 1)
            novo.append([dfdx, xk, f])

            """task2 -> recebe como parâmetro o valor de x(xk) e alfa(taxa de aprendizagem)
                        e retorna o novo valor para x, e a derivada correspondente """
            xk, dfdx = self.task2(xk, alfa)
            k += 1

        """Plots the function, and the value of the derivative associated with its coordinates."""
        self.plot_3(database=novo)

        resul = np.array(novo)
        resul = np.reshape(resul[:, 2:3],-1)

        """Returns the index of the smallest value found"""
        pos_minimo = int(np.where(resul == min(resul))[0])

        messenger = "Tarefa 3.a \nThe minimum value found for the function f(x) is %0.4f from x= %0.4f"
        print(messenger % (min(resul), novo[pos_minimo][1]) + '\n')
        return

    def gradiente_duas_variaveis(self, initialp, alpha, gmin, kmax, h):
        """ This function implements algorithm 4, presented in the book,
            "Aprendizagem de Máquina", page 93, author: Thomas Rauber.

            "Algoritmo 4: Descida de Gradiente, Mais que uma Dimensão"

            ! It is worth mentioning that the derivative here is
              calculated using the Finite Differences Method to
              calculate the derivative.!
            """

        """ k    -> represents the iterations
            x   -> represents the initial value of x1 and x2
            data_fx1x2 -> list to store the values of f(x1,x2)"""
        k, x = 0, np.array(initialp)
        data_fx1x2 = []

        """Receives the values of X1 and X2 and returns f(X1, X2)"""
        fx1x2 = self.task4_function(x[0], x[1])
        data_fx1x2.append([x[0], x[1], fx1x2])

        """Calculates the gradient vector, that is, the partial
           derivatives with respect to X1 and X2
           grad -> It's the gradient vector """
        grad = self.task4_a(x, h)

        """np.linalg.norm(grad) -> represents the norm of the gradient vector """
        while k < kmax and np.linalg.norm(grad) > gmin:
            grad = self.task4_a(x, h) # calculate the gradient vector
            prod = alpha * grad
            x = x - prod
            fx1x2 = self.task4_function(x[0], x[1])  # returns f(X1, X2)
            data_fx1x2.append([x[0], x[1], fx1x2])   # save the values of f(X1, X2)
            k += 1

        # Data necessary to answer question 4a of the list
        minimo_fx1x2_mdf2 = np.array(data_fx1x2)[len(data_fx1x2)-1:len(data_fx1x2)]

        k, x = 0, np.array(initialp)
        data_fx1x2_literal = []
        
        fx1x2 = self.task4_function(x[0], x[1])
        data_fx1x2_literal.append([x[0], x[1], fx1x2])
        grad = self.task4_b(x)
        while k < kmax and np.linalg.norm(grad) > gmin:
            grad = self.task4_b(x)
            x = x - alpha * grad
            fx1x2 = self.task4_function(x[0], x[1])
            data_fx1x2_literal.append([x[0], x[1], fx1x2])
            k += 1
        # Data necessary to answer question 4b of the list
        minimo_fx1x2_dliteral = np.array(data_fx1x2_literal)[len(data_fx1x2_literal) - 1:len(data_fx1x2_literal)]

        #self.plot_lossfunction(data_fx1x2, data_fx1x2_literal)
        return minimo_fx1x2_mdf2, minimo_fx1x2_dliteral

    def task4_a(self, Xk, h):
        """Calculates the gradient vector by the finite difference method"""
        dfdx1 = self.dfx_4(Xk[0], Xk[1], h, 0)
        dfdx2 = self.dfx_4(Xk[0], Xk[1], 0, h)
        return np.array([dfdx1, dfdx2])


    def dfx_4(self, x1, x2, h1, h2):
        """ Approximation of the derivative by the finite difference method.
        In this case, for two variables """
        return (self.task4_function(x1 + h1, x2 + h2) - self.task4_function(x1, x2)) / (h1 + h2)

    def task4_function(self, x1, x2):
        fx1x2 = (4 - 2.1 * x1 ** 2 + x1 ** 3 / 3) * x1 ** 3 + x1 * x2 + (-4 + 4 * x2 ** 2) * x2 ** 2
        return fx1x2

    def task4_b(self, Xk):
        x1 = Symbol('x1')
        x2 = Symbol('x2')
        f = (4 - 2.1 * x1 ** 2 + x1 ** 3 / 3) * x1 ** 3 + x1 * x2 + (-4 + 4 * x2 ** 2) * x2 ** 2

        """derivada da fx/dx1"""
        difx1 = diff(f, x1)  # dfx1 is the derivative of the function
        lam_f1 = lambdify(x1, difx1)
        dfdx1 = lam_f1(Xk[0])
        func = lambdify(x2, dfdx1)
        dfdw1 = func(Xk[1])

        """derivada da fx/dx2"""
        difx2 = diff(f, x2)  # dfx2 is the derivative of the function
        lam_f2 = lambdify(x2, difx2)
        dfdx2 = lam_f2(Xk[1])
        func = lambdify(x1, dfdx2)
        dfdw2 = func(Xk[0])

        # xk = x0 - (alfa * dfdx)  # x_(k+1) <-  xk - alfa * f'(xk)
        return np.array([dfdw1, dfdw2])

    # (c) Desenhe a trajetória de x k no plano (x 1 , x 2 ),
    # e o valor da função correspondente de f(x1 , x2) no gráfico 3-D.
    def task4_c(self):

        # self.plot_question4()
        self.graf3d()
        return print('task4_c')





    #********************************************
    #
    #   Daqui para baixo estão implementadas
    #   as funções responsáveis por plotar
    #   e exportar os arquivos
    #

    def plot_question4(self):

        x1 = np.linspace(-4, 4, 100)
        x2 = np.linspace(-4, 4, 100)
        x1, x2 = np.meshgrid(x1, x2)
        Z = self.task4_function(x1, x2)

        fig = plt.figure(figsize=(5, 5))
        ax = plt.axes(projection='3d')
        # ax.contour3D(x1, x2, Z, 50, cmap='coolwarm')
        ax.plot_surface(x1, x2, Z, cmap="coolwarm", lw=0.5, rstride=1, cstride=1)

        ax.set_xlabel(r'X1', fontsize=10)
        ax.set_ylabel(r'X2', fontsize=10)
        ax.set_zlabel(r'f(X1, X2)', fontsize=10)
        ax.view_init(70, 35)
        plt.show()
        pass

    def plot_lossfunction(self, database, database_dliteral):

        """ O resultado, resul, apresenta o valor da função que queremos minimizar
        ao realizar o método da descida de gradientes. Logo, a sequência de valores de x1 e x2
        escolhidos, indicam o caminho que percorre a busca pela melhor solução"""

        dados = np.array(database)
        xaxis, yaxis, resul = dados[:, 0:1], dados[:, 1:2], dados[:, 2:3]
        xaxis, yaxis, resul = np.reshape(xaxis, -1), np.reshape(yaxis, -1), np.reshape(resul, -1)
        w1, w2 = xaxis, yaxis

        xaxis, yaxis = np.array(database_dliteral)[:, 0:1], np.array(database_dliteral)[:, 1:2]
        xaxis, yaxis  = np.reshape(xaxis, -1), np.reshape(yaxis, -1)
        l1, l2 = xaxis, yaxis

        """ Define the parameters and how much of the domain of the function
            you want to plot the contour line"""
        fig = plt.figure()  # Create the figure
        min, max = -1.5, 1.5
        xaxis, yaxis = np.arange(min, max, 0.01), np.arange(min, max, 0.01)
        x1, x2 = np.meshgrid(xaxis, yaxis)
        resul = (4 - 2.1 * x1 ** 2 + (x1 ** 3) / 3) * x1 ** 3 + x1 * x2 + (-4 + 4 * x2 ** 2) * x2 ** 2

        """ Added to not have to put this part of the algorithm as a comment often,
            because it's heavy"""
        chave = True
        if chave == True:
            adc = str(0)
            for i in range(0, len(w1), 1):
                tamanho = len(xaxis)

                """ Important to save the images in an orderly way """
                if i == 10:
                    adc = ""

                plt.title('Descida de gradiente')
                plt.xlabel('X1'), plt.ylabel('X2')

                """ xc, yc -> represent the coordinates of the points
                    of a solution. It is done iteratively so that several
                    images are plotted, and then a gif is created """
                xc, yc = float(w1[i:i + 1]), float(w2[i:i + 1])
                l1c, l2c = float(l1[i:i + 1]), float(l2[i:i + 1])

                """ Plot the contour and then the point"""
                plt.contourf(xaxis, yaxis, resul, levels=50, cmap='RdGy')
                plt.plot(xc, yc,   marker="o", markersize=5, markeredgecolor="black",
                         markerfacecolor="green")
                plt.plot(l1c, l2c, marker="d", markersize=5, markeredgecolor="black",
                         markerfacecolor="orange")

                plt.savefig("Image/gif/" + adc + str(i) + ".png")
                plt.clf()
        plt.close(fig)
        return

    def make_gif(self, frame_folder):
        frames = [Image.open(image) for image in sorted(glob.glob(f"{frame_folder}/*.png"))]

        frame_one = frames[0]
        frame_one.save("Image/my_awesome.gif", format="GIF", append_images=frames,
                       save_all=True, duration=400, loop=0)

    def plot_question3(self):
        """Generate the points from the function"""
        y, xaxis = [], []
        for x in np.arange(-4.0, 4.0, 0.01):
            f = exp(-x) * x * ((x ** 2) - x - 1)
            y.append(f), xaxis.append(x)

        plt.plot(xaxis, y, alpha=0.95)
        plt.xlim(-1, 3.5), plt.ylim(-0.5, 1)
        pass

    def plot_3(self, database):

        """Converts the database into a DataFrame, where the value and x1, x2 and
            corresponding derivative are present. Subsequently, the
            self.plot_question3() function is called, where the graph of the
            corresponding function in the interval is plotted. Then, an overlap
            of plots occurs."""

        newnovo = pd.DataFrame(database, index=None, columns=['Derivada', 'X', 'f(x)'])
        newnovo.plot(x='X', y='f(x)', c='Derivada', kind='scatter', cmap="jet", s=50, marker='o',
                     alpha=0.7, label=r'$f ~ (x) = e^{-x} (x^3- x^2 - x)$')

        """Gera os pontos da função para depois acrescentar sobre a curva os pontos, da derivada
        da função """
        self.plot_question3()

        plt.title('Descida de Gradiente')
        plt.xlabel('x'), plt.ylabel('f(x)')

        plt.text(-0.6, 0.6, r'$x_0 = 3$    $\alpha = 0.1$    $K_{max} = 100$',
                 bbox={'facecolor': 'white', 'edgecolor': 'k', 'boxstyle': 'round, pad=1'})

        plt.annotate(
            'Mínimo local:\n x=' + str(round(newnovo['X'][49], 4)) + '\n f(x)=' + str(round(newnovo['f(x)'][49], 4)),
            xy=(newnovo['X'][49], -0.404), xytext=(0.5, 0.2),
            arrowprops=dict(facecolor='gray', shrink=0.05))

        plt.savefig("Image/first_function.png")
        plt.show()
        pass

    def graf3d(self):
        min, max = -1.0, 1.0
        xaxis = np.arange(min, max, 0.01)
        yaxis = np.arange(min, max, 0.01)
        x1, x2 = np.meshgrid(xaxis, yaxis)
        resul = (4 - 2.1 * x1 ** 2 + (x1 ** 3) / 3) * x1 ** 3 + x1 * x2 + (-4 + 4 * x2 ** 2) * x2 ** 2
        figure = plt.figure()
        axis = figure.gca(projection='3d')
        axis.plot_surface(x1, x2, resul, cmap='jet', linewidth=0, antialiased=False)

        plt.title('Gradiente Decedente Multivariado')
        plt.xlabel('X1'), plt.ylabel('X2'), axis.set_zlabel('f(x1,x2)')
        plt.contourf(x1, x2, resul, levels=50, cmap='jet', zdir="z", offset=-3)
        plt.show()
    pass