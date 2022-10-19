from src.problem.primeiro_trabalho import PrimeiroTrabalho
from src.algorithm.gradient_descent import gradient_descent
import time


def build_problem(problem_name):
    if problem_name == "Gradient Descent":
        return PrimeiroTrabalho("")
    else:
        raise NotImplementedError()

def main():

    # Parameter passing
    type_problem = ["Gradient Descent"]
    problem = build_problem(type_problem[0])

    graph = []
    t_initial = time.time()

    for i in range(0, 1):
        output = gradient_descent(problem)
        graph.append(output)

    # Generates data for future analysis
    t_final = time.time()
    deltat = t_final - t_initial
    print(deltat)


if __name__ == "__main__":
    main()
