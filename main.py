import random
import numpy as np
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from tools import Tools



var_tools = Tools('data/p3.xlsx')

print('Trabajos: ' + str(var_tools.jobs))
print('Lotes: ' + str(var_tools.lots))
print('Operations: ' + str(var_tools.operations))
print('Máquinas: ' + str(var_tools.machines))


creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list,  fitness=creator.FitnessMin)


toolbox = base.Toolbox()
toolbox.register("indices", random.sample, range(var_tools.jobs), var_tools.jobs)

# Inicializador individual y de población
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)                  # lista de indivíduos

# Operador inicializador
toolbox.register("evaluate", var_tools.objective_function) # función objetivo
toolbox.register("mate", tools.cxUniformPartialyMatched, indpb=0.05)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=5)

pop = toolbox.population(n=50)                            # inicio emergente
hof = tools.HallOfFame(1)                                 # mejor individuo
stats = tools.Statistics(lambda ind: ind.fitness.values)  # Estadísticas
stats.register("avg", np.mean)
stats.register("std", np.std)
stats.register("min", np.min)
stats.register("max", np.max)

pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.3, mutpb=0.2, ngen=100, stats=stats, halloffame=hof, verbose=True)

var_tools.statistics_graph(log)

# mejor solución
print("Mejor individuo:")
print(hof[0])
print(var_tools.decode(hof[0]))
print(var_tools.decode(hof[0])['finish'].max())

# Mejor resultado de la función objetivo
print("Mejor resultado de la función objetivo:")
print(var_tools.objective_function(hof[0])[0])
var_tools.gantt(hof[0])
