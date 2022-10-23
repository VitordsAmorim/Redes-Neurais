from traceback import print_list
import numpy as np
from src.problem.problem_interface import ProblemInterface
from sympy import diff, Symbol, true
from sympy import lambdify
from sympy import sin, cos, exp
import matplotlib.pyplot as plt
import pandas as pd
from numpy import arange
from numpy import meshgrid
from mpl_toolkits.mplot3d import Axes3D






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
        f = exp(-x) * x * ((x ** 2) - x - 1)
        difx = diff(f, x)
        # Apresenta a função derivada
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
        xk, dfdx = self.task2(xk, gmin)
        xk=x0
        while ((k < kmax) and (math.fabs(dfdx) > gmin)):
            f = math.exp(-xk) * xk * ((xk ** 2) - xk - 1)
            novo.append([dfdx,xk,f])
            xk, dfdx = self.task2(xk, alfa)
            k += 1
        newnovo = pd.DataFrame(novo, index=None, columns=['Derivada:','X:','Y:'])
        print(newnovo.to_string())
        newnovo.plot(x ='X:', y='Y:',c='Derivada:', kind = 'scatter',cmap="jet", s=50,marker= 'o', label='Gradiente', alpha=0.7)
        self.plot()
        ###############################
        plt.suptitle('Exercício 03', fontsize=14, fontweight='bold')
        plt.title('Gradiente Decedente')
        plt.xlabel('Movimentação do Gradiente no eixo X')
        plt.ylabel('Eixo Y')
        plt.text(-0.5, 0.6, 'Quanto maior o Valor da Derivada, maior é o passo.\n kmax = 100, porém foi até 49 pois \n o critério de gimin=0.1 ocorreu primeiro', style='italic',
        bbox={'facecolor': 'white', 'edgecolor': 'k', 'boxstyle': 'round, pad=1'})
        plt.text(-0.5, 0.8, r'$f ~ (x) = e^{-x} (x^3- x^2 - x)$', fontsize=12)
        plt.annotate('ótimo local:\n X='+str(round(newnovo['X:'][49],4))+'\n Y='+str(round(newnovo['Y:'][49],4)), xy=(newnovo['X:'][49], -0.404), xytext=(0.5, 0.2),
            arrowprops=dict(facecolor='black', shrink=0.05))
        ###############################
        plt.show()
        print("========= AOS COLEGAS ==============")
        print("Se colocar o Alpha(txa de aprendizado em 0.9 ele leva 6 passos para achar o melhor,\n mas como sabe qual alpha é o certo?")
        print("====================================")
        return print('O valor que minimiza a função f(x) é x=', newnovo['X:'][49])


    def derivx1(self,x0,alfa):
        x1 = Symbol('x1')
        x2=Symbol('x2')
        #Encontrando o valor de x1 e dx1
        f = (4-2.1*x1**2+x1**(3/3))*x1**3+x1*x0#AKI EU COLOCO O X0 COMO VALOR PARA X2
        dfdx1 = diff(f, x1)
        lam_f = lambdify(x1, dfdx1)#AKI É O VETOR DE GRADIENTE
        dfdx1 = lam_f(x0)
        # x1k = x0 - (alfa * dfdx1)
       #Encontrando o valor de x2 e dx2
        f = x0*x2+(-4+4*x2**2)*x2**2#AKI EU COLOCO O X0 COMO VALOR PARA X1
        dfdx2 = diff(f, x2)
        lam_f = lambdify(x2, dfdx2)#AKI É O VETOR DE GRADIENTE
        dfdx2 = lam_f(x0)
        # x2k = x0 - (alfa * dfdx2)
        grad=([dfdx1],[dfdx2])
        print(grad)
        print(np.dot(grad,-alfa),"aaki")
        # print(x1k,"<<>>",dfdx1,"------",x2k,"<<>>",dfdx2)
        pass


    def derivx2(self):
        x1 = Symbol('x1')
        x2=Symbol('x2')
        f = (4-2.1*x1**2+x1**(3/3))*x1**3+x1*x2+(-4+4*x2**2)*x2**2
        return diff(f, x2)
    def t1(self):
        x1 = Symbol('x1')
        x2=Symbol('x2')
        f = (4-2.1*x1**2+x1**(3/3))*x1**3+x1*x2+(-4+4*x2**2)*x2**2
        return diff(f, x1)
    def t2(self):
        x1 = Symbol('x1')
        x2=Symbol('x2')
        f = (4-2.1*x1**2+x1**(3/3))*x1**3+x1*x2+(-4+4*x2**2)*x2**2
        return diff(f, x2)
    
    def task4_a(self,x0, alfa,gmin, kmax):
        print("vetor do Gradiente de 2 dimensões função do trabalho:(4-2.1*x1**2+x1**3/3)*x1**3+x1*x2+(-4+4*x2**2)*x2**2")
        # print([[self.derivx1()],[self.derivx2()]])
        # ##Início da decida7
        x1k = x0
        x2k=x0
        k = 0
        derivada=[self.t1()],[self.t2()]
        novo=[]
        dfdx=0
        while (k < kmax):
          self.derivx1(1.2,alfa)
      
            # f = (4-2.1*x1k**2+x1k**(3/3))*x1k**3+x1k*x2k+(-4+4*x2k**2)*x2k**2
            # novo.append([dfdx,x1k,x2k,f])
            # x1k, dfdx = self.task2(x1k, alfa)


          k += 1
        # ##########FIM
        self.graf3d()
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
        y = []
        xaxis = []
        for x in np.arange(-4.0, 4.0, 0.01):
            f = exp(-x) * x * ((x ** 2) - x - 1)
            y.append(f)
            xaxis.append(x)
        plt.plot(xaxis, y,alpha=0.5)
        plt.xlim(-1, 3.5)
        plt.ylim(-0.5, 1)
        # plt.show()
        pass


    def graf3d(self):
        min, max = -1.0, 1.0
        xaxis = arange(min, max, 0.01)
        yaxis = arange(min, max, 0.01)
        x1, x2 = meshgrid(xaxis, yaxis)
        resul = (4-2.1*x1**2+x1**(3/3))*x1**3+x1*x2+(-4+4*x2**2)*x2**2
        figure = plt.figure()
        axis = figure.gca(projection='3d')
        axis.plot_surface(x1, x2, resul, cmap='jet' ,linewidth=0, antialiased=False)
        plt.suptitle('Exercício 0', fontsize=14, fontweight='bold')
        plt.title('Gradiente Decedente Multivariado')
        plt.xlabel('Eixo X1')
        plt.ylabel('Eixo x2')
        axis.set_zlabel('Eixo Y')
        plt.contourf(x1, x2, resul, levels=50, cmap='jet',zdir="z", offset=-3)
        plt.show()
    pass

    def plot_bestfit(self):
        # TODO
        pass

