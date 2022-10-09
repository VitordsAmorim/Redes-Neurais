# Call the functions in the Descida_de_gradiante file

def gradient_descent(problem):

    """ Dada a função f (x) = exp(−x) ∗ x ∗ (x ∗ ∗2 − x − 1), o valor inicial x0 = 1, e a taxa
    de aprendizagem alfa= 0.1, calcule o próximo valor x1 pela descida de gradiente, usando
    uma aproximação da derivada pelo método de diferenças finitas (eq. 3.1), com um passo
    h = 0.01, diferenciação numérica. """
    print("Tarefa 1:")
    x1 = problem.task1(x0=1, alfa=0.1, h=0.01)
    print("x0: %.4f, x1: %.4f" % (1, x1))
    print("******************")


    """"    Calcule x1 , usando a derivada f′(x) 
    
    Ajuda: Para calcular a derivada analiticamente, use a função ’diff’ do pacote ’sympy’ do Python.
    A função ’lambdify’ converte de uma função simbólica para uma função numérica."""
    dx1 = problem.task2(x0=1, alfa=0.1)
    print("Tarefa 2:")
    print("x0: %.4f, x1: %.4f" % (1, dx1))
    print("******************")


    """   Usando o método da descida de gradiente, tente aproximar o mínimo de f,
          com o valor inicial x_0=3 e taxa de aprendizagem alpha=0.1.
          
          Defina as duas condições de parada do Algoritmo 3, com gmin = 0.1, k_max = 20

          Qual é o valor de um mínimo obtido?
          
          Usando um gráfico bidimensional, eixo x = x, eixo y = f(x), marque os valores da aproximação
          x_0 --> x_1 --> x_2 --> x_3 --> x_4 --> x_5 ...
          e f(x_0) --> f(x_1) --> f(x_2) --> f(x_3) --> f(x_4) --> f(x_5) ...
          e f'(x_0) --> f'(x_1) --> f'(x_2) --> f'(x_3) --> f'(x_4) --> f'(x_5) ..."""
    problem.task3(x0=3, alfa=0.1, gmin=0.1, kmax=20)


    """ 4. Determine pelo menos dois mínimos da função bidimensional:
    
    f(x1, x2) = (4 - 2.1 * x1**2 + x1**3 / 3 ) * x1**3 + x1 * x2 + (-4 + 4 * x2**2) * x2**2
        
    """
    problem.plot()

    # (a) Use o método de diferenças finitas para aproximar o gradiente.
    problem.task4_a()

    # (b) Use o gradiente explícito no algoritmo da descida de gradiente
    problem.task4_b()

    # (c) Desenhe a trajetória de x k no plano (x 1 , x 2 ),
    # e o valor da função correspondente de f(x1 , x2) no gráfico 3-D.
    problem.task4_c()

    return [print("Fim")]
