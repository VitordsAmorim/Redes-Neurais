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
import glob
from PIL import Image



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

        #self.plot()
        self.plot_3(database=novo)
        min_pos = flist.index(min(flist))
        messenger = "Tarefa 3.a \nThe minimum value found for the function f(x) is y= %0.4f from x= %0.4f"
        return print( messenger % (min(flist), novo[min_pos][1])+'\n')


    # (a) Use o método de diferenças finitas para aproximar o gradiente.
    def diferencas_finitas_duas_variaveis(self, initialp, alpha, gmin, kmax, h):
        k = 0
        x = np.array(initialp)
        data_fx1x2,flist = [], []

        fx1x2 = self.task4_function(x[0], x[1])
        data_fx1x2.append([x[0], x[1], fx1x2])
        flist.append(fx1x2)
        grad = self.task4_a(x, h)
        """ math.fabs is used to get the absolute value of the derivative """
        while k < kmax and np.linalg.norm(grad) > gmin:
            grad = self.task4_a(x, h)
            prod = alpha *  grad
            x = x - prod

            fx1x2 = self.task4_function(x[0],x[1])
            data_fx1x2.append([x[0], x[1], fx1x2])
            flist.append(fx1x2)
            print(k)
            k += 1
        self.plot_lossfunction(data_fx1x2)

        return

    def task4_a(self, Xk, h):
        X = Xk
        dfdx1 = self.dfx_4(X[0], X[1], h, 0)
        dfdx2 = self.dfx_4(X[0], X[1], 0, h)
        return np.array([dfdx1, dfdx2])

    def dfx_4(self, x1, x2, h1, h2):
        """ Approximation of the derivative by the finite difference method """
        """ In this case, for two variables"""
        df = (self.task4_function(x1 + h1, x2 + h2) - self.task4_function(x1, x2))/(h1 + h2)
        return df

    def task4_function(self, x1, x2):
        fx1x2 = (4 - 2.1 * x1 ** 2 + x1 ** 3 / 3) * x1 ** 3 + x1 * x2 + (-4 + 4 * x2 ** 2) * x2 ** 2
        return fx1x2

    # (b) Use o gradiente explícito no algoritmo da descida de gradiente
    def task4_b(self, w1, w2):
        x1 = Symbol('x1')
        x2 = Symbol('x2')
        f = (4 - 2.1 * x1 ** 2 + x1 ** 3 / 3) * x1 ** 3 + x1 * x2 + (-4 + 4 * x2 ** 2) * x2 ** 2

        """derivada da fx/dx1"""
        difx1 = diff(f, x1)  # dfx1 is the derivative of the function
        lam_f1 = lambdify(x1, difx1)
        dfdx1 = lam_f1(w1)
        func = lambdify(x2, dfdx1)
        dfdw1 = func(w2)

        """derivada da fx/dx2"""
        difx2 = diff(f, x2)  # dfx2 is the derivative of the function
        lam_f2 = lambdify(x2, difx2)
        dfdx2 = lam_f2(w2)
        func = lambdify(x1, dfdx2)
        dfdw2 = func(w1)

        # xk = x0 - (alfa * dfdx)  # x_(k+1) <-  xk - alfa * f'(xk)
        return dfdw1, dfdw2, difx1, difx2

    # (c) Desenhe a trajetória de x k no plano (x 1 , x 2 ),
    # e o valor da função correspondente de f(x1 , x2) no gráfico 3-D.
    def task4_c(self):

        #self.plot_question4()
        self.graf3d()
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

        ax.set_xlabel(r'X1', fontsize=10)
        ax.set_ylabel(r'X2', fontsize=10)
        ax.set_zlabel(r'f(X1, X2)', fontsize=10)
        ax.view_init(70, 35)
        plt.show()
        pass

    def plot_lossfunction(self, database):

        """ O resultado, resul, apresenta o valor da função que queremos minimizar
        ao realizar o método da descida de gradientes. Logo, a sequência de valores de x1 e x2
        escolhidos, indicam o caminho que percorre a busca pela melhor solução"""
        dados = np.array(database)
        xaxis, yaxis = dados[ : , 0:1], dados[ : , 1:2]
        xaxis, yaxis = np.reshape(xaxis, -1), np.reshape(yaxis, -1)
        resul = dados[:, 2:3]
        resul = np.reshape(resul, -1)
        w1, w2 = xaxis, yaxis

        fig = plt.figure()
        #ax = fig.add_subplot(projection='3d')
        #ax.plot(xaxis, yaxis, resul, color="black")

        min, max = -1.5, 1.5
        xaxis, yaxis = np.arange(min, max, 0.01), np.arange(min, max, 0.01)
        x1, x2 = np.meshgrid(xaxis, yaxis)
        resul = (4 - 2.1 * x1 ** 2 + (x1 ** 3) / 3) * x1 ** 3 + x1 * x2 + (-4 + 4 * x2 ** 2) * x2 ** 2

        #axis = fig.gca(projection='3d')
        #axis.plot_surface(x1, x2, resul, cmap='jet', linewidth=0, antialiased=False)

        #plt.contourf(xaxis, yaxis, resul, levels=50, cmap='RdGy',zdir="z", offset=-3)


        # Agora vem a mágica

        print("")
        adc = str(0)
        for i in range(0, len(w1), 1):
            tamanho = len(xaxis)
            if i == 10:
                adc = ""

            xc = float(w1[i:i+1])
            yc = float(w2[i:i+1])
            plt.title('Descida de gradiente')
            plt.xlabel('X1')
            plt.ylabel('X2')
            plt.contourf(xaxis, yaxis, resul, levels=50, cmap='RdGy', zdir="z", offset=-3)
            plt.plot(xc, yc, marker="d", markersize=4, markeredgecolor="black", markerfacecolor="green")
            plt.savefig("Image/gif/" + adc + str(i) + ".png")
            plt.clf()
            #print(i)

        #ax.set_zlabel('f(x1,x2)')
        #plt.show()
        return

    def make_gif(self, frame_folder):
        frames = [Image.open(image) for image in sorted(glob.glob(f"{frame_folder}/*.png"))]

        frame_one = frames[0]
        frame_one.save("Image/my_awesome.gif", format="GIF", append_images=frames,
                       save_all=True, duration=400, loop=0)

    def plot(self):
        """Generate the points from the function"""
        y, xaxis = [], []

        for x in np.arange(-4.0, 4.0, 0.01):
            f = exp(-x) * x * ((x ** 2) - x - 1)
            y.append(f)
            xaxis.append(x)

        plt.plot(xaxis, y, alpha=0.95)
        plt.xlim(-1, 3.5)
        plt.ylim(-0.5, 1)
        pass

    def plot_3(self, database):

        newnovo = pd.DataFrame(database, index=None, columns=['Derivada', 'X', 'f(x)'])
        newnovo.plot(x='X', y='f(x)', c='Derivada', kind='scatter', cmap="jet", s=50, marker='o',
                     alpha=0.7, label= r'$f ~ (x) = e^{-x} (x^3- x^2 - x)$')

        """Gera os pontos da função para depois acrescentar sobre a curva os pontos, da derivada
        da função """
        self.plot()

        plt.title('Descida de Gradiente')
        plt.xlabel('x')
        plt.ylabel('f(x)')

        """
        Vale a pena por essa informação no relatório:
        
        plt.text(-0.5, 0.6, 'Quanto maior o Valor da Derivada, maior é o passo.\n kmax = 100, porém foi até 49 pois \n o critério de gimin=0.1 ocorreu primeiro',
                 style='italic',
                 bbox={'facecolor': 'white', 'edgecolor': 'k', 'boxstyle': 'round, pad=1'})
        """
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
        resul = (4-2.1*x1**2+(x1**3)/3)*x1**3+x1*x2+(-4+4*x2**2)*x2**2
        figure = plt.figure()
        axis = figure.gca(projection='3d')
        axis.plot_surface(x1, x2, resul, cmap='jet' ,linewidth=0, antialiased=False)

        plt.title('Gradiente Decedente Multivariado')
        plt.xlabel('X1')
        plt.ylabel('X2')
        axis.set_zlabel('f(x1,x2)')
        plt.contourf(x1, x2, resul, levels=50, cmap='jet',zdir="z", offset=-3)
        plt.show()
    pass
