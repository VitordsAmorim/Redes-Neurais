from abc import ABC, abstractmethod

# ProblemInterface is an abstract base class (ABC). The classes that
# inherit from ProblemInterface need to implement the abstract methods
# in order to be instantiable.
class ProblemInterface(ABC):

    @abstractmethod
    def task1(self, x0, alfa, h):
        pass

    @abstractmethod
    def task2(self, x0, alfa):
        pass

    @abstractmethod
    def task3(self, x0, alfa, gmin, kmax):
        pass

    @abstractmethod
    def task4_a(self):
        pass

    @abstractmethod
    def task4_b(self):
        pass

    @abstractmethod
    def task4_c(self, history_x):
        pass

    @abstractmethod
    def plot_question3(self):
        pass
