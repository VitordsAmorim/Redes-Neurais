from src.problem.primeiro_trabalho import PrimeiroTrabalho
from src.algorithm.gradient_descent import gradient_descent
import time


def build_problem(problem_name):
    if problem_name == "Descida de gradiante":
        return PrimeiroTrabalho("")
    else:
        raise NotImplementedError()

def main():

    # Passagem de parâmetros da atividade
    typeproblem = ["Descida de gradiante"]
    problem = build_problem(typeproblem[0])

    graph = []
    t_inicial = time.time()

    for i in range(0, 1):
        output = gradient_descent(problem)
        graph.append(output)

    # Gera dados para uma possível análise
    t_final = time.time()
    deltat = t_final - t_inicial
    print(deltat)


if __name__ == "__main__":
    main()
