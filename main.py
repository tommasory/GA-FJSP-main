import pandas as pd
import os
import random
import numpy as np
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
import math
import sys
import matplotlib.pyplot as plt

# lea el archivo de Excel que contiene el conjunto de datos 
raw_df = pd.read_excel('data/p1.xlsx')
LOT_NUMBER = 1 # 1, 2 ou 3

# marco de datos que contiene todos los trabajos
jobs_df = raw_df[['lot', 'operation', 'machine', 'proc-time']]
jobs_df = jobs_df.dropna()
jobs_df[['lot', 'operation', 'machine']] = jobs_df[['lot', 'operation', 'machine']].astype(int)
jobs_df.head()

# algunos parámetros de los trabajos a realizar
LOTS = jobs_df['lot'].nunique()
OPERATIONS = jobs_df['operation'].nunique()
MACHINES = jobs_df['machine'].nunique()
JOBS = len(jobs_df.index)

print('Trabajos: ' + str(JOBS))
print('Lotes: ' + str(LOTS))
print('Operations: ' + str(OPERATIONS))
print('Máquinas: ' + str(MACHINES))

# marco de datos con tamaño de lote
lots_size_df = raw_df[['lot.1', 'lotSize_1', 'lotSize_2', 'lotSize_3']]
lots_size_df = lots_size_df.dropna()
lots_size_df = lots_size_df.rename(columns={'lot.1':'lot'})
lots_size_df[['lot', 'lotSize_1', 'lotSize_2', 'lotSize_3']] = lots_size_df[['lot', 'lotSize_1', 'lotSize_2', 'lotSize_3']].astype(int)
lots_size_df.head()

# responsable de arreglar individuo
# 1. elimina operaciones repetidas, dando prioridad al orden de aparición en el cromosoma
# 2. asegura que las operaciones obedezcan los requisitos previos, dando prioridad al orden de aparición en el cromosoma
def fix_individual(individual):

  # crear un marco de datos para representar al individuo original
  individual_df = pd.DataFrame(columns=['lot','operation','machine'])

  # complete el marco de datos del individuo original
  for i in individual:
    lot = jobs_df.loc[i, 'lot']
    operation = jobs_df.loc[i, 'operation']
    machine = jobs_df.loc[i, 'machine']

    # verificar si la operación ya está en el marco de datos del individuo
    is_already = not individual_df[(individual_df['lot'] == lot) & (individual_df['operation'] == operation)].empty
    if(is_already):
      continue

    individual_df.loc[i, ['lot', 'operation', 'machine']] = lot, operation, machine
  individual_df = (individual_df.reset_index()).drop('index', axis=1)
  
  # crear un marco de datos para representar al individuo reparado
  fixed_df = pd.DataFrame(columns=['lot','operation','machine'])

  # arreglar al individuo
  for i in individual_df.index:

    if(not (i in individual_df.index)):
      continue

    lot = individual_df.loc[i, 'lot']
    operation = individual_df.loc[i, 'operation']
    machine = individual_df.loc[i, 'machine']

    # verificar si esta operación por lotes ya está en el marco de datos
    is_already = not fixed_df[(fixed_df['lot'] == lot) & (fixed_df['operation'] == operation) & (fixed_df['machine'] == machine)].empty
    if(is_already):
      continue

    # verificar si esta operación se puede realizar
    prev_lot_op = fixed_df.loc[fixed_df['lot'] == lot]['operation'].max()

    if(math.isnan(prev_lot_op)):
      prev_lot_op = 0

    # no se puede realizar la operación, se deben realizar los requisitos previos antes
    if(operation - prev_lot_op != 1):

      # buscar requisitos previos
      lot_req_op = (individual_df.loc[(individual_df['lot'] == lot) & 
                                      (individual_df['operation'] < operation)]).sort_values(by='operation', ascending=True)
          
      indexes = lot_req_op.index.values
    
      # eliminar los requisitos previos del marco de datos del individuo original
      individual_df = individual_df.drop(indexes)

      # agregar requisitos previos al marco de datos fijo
      fixed_df = (fixed_df.append(lot_req_op, ignore_index=True))

    # agregar la operación al marco de datos del individuo fijo
    fixed_lst = [[lot, operation, machine]]
    df = pd.DataFrame(fixed_lst, columns = ['lot', 'operation', 'machine'])
    fixed_df = fixed_df.append(df, ignore_index=True)

    # eliminar la operación de marco de datos individual
    individual_df = individual_df.drop(i)
    
  # llenar el individuo construido de acuerdo con el marco de datos
  fixed_individual = []
  for i in fixed_df.index:    
    lot = fixed_df.loc[i, 'lot']
    operation = fixed_df.loc[i, 'operation']
    machine = fixed_df.loc[i, 'machine']
    
    index = jobs_df.loc[(jobs_df['lot'] == lot) & 
                        (jobs_df['operation'] == operation) & 
                        (jobs_df['machine'] == machine)].index.values[0]

    fixed_individual.append(index)
  
  return fixed_individual

#individual
########################################
# N indexes that are mapped to jobs_df #
########################################

def decode(individual):

  # establecer el orden de los índices según el orden de las operaciones
  individual_fixed = fix_individual(individual)

  # crear marco de datos para indicar la aplicación de operaciones en máquinas
  schedule_df = pd.DataFrame(columns=['lot','operation','machine','start','finish'])

  # llenar todo el marco de datos con la información del individuo
  for i in individual_fixed:

    lot = jobs_df.loc[i, 'lot']
    operation = jobs_df.loc[i, 'operation']
    machine = jobs_df.loc[i, 'machine']

    proc_time = jobs_df.loc[(jobs_df['lot'] == lot) & 
                            (jobs_df['operation'] == operation) & 
                            (jobs_df['machine'] == machine)].reset_index().loc[0, 'proc-time']

    lot_size = lots_size_df.loc[(lots_size_df['lot'] == lot)].reset_index().loc[0, 'lotSize_{}'.format(LOT_NUMBER)]

    # comprobar el final de la última operación de la máquina
    last_finish_machine = schedule_df.loc[schedule_df['machine'] == machine]['finish'].max()
    if(math.isnan(last_finish_machine)):
      last_finish_machine = 0

    # verificar cuándo fue el final de la última operación por lotes
    last_finish_prev_op = schedule_df.loc[schedule_df['lot'] == lot]['finish'].max()
    if(math.isnan(last_finish_prev_op)):
      last_finish_prev_op = 0

    # la operación comenzará después de que la máquina esté libre y finalice la operación anterior del lote
    start = max(last_finish_machine, last_finish_prev_op)

    schedule_lst = [[lot, operation, machine, start, start+proc_time*lot_size]]
    df = pd.DataFrame(schedule_lst, columns = ['lot', 'operation', 'machine', 'start', 'finish'])
    schedule_df = schedule_df.append(df, ignore_index=True)
    
  return schedule_df

def gantt(individual):
  schedule_df = decode(individual)
  makespan = objective_function(individual)[0]

  # Declarar una figura "gnt"
  fig, gnt = plt.subplots(figsize=(20,5)) 
    
  # Configuración de los límites del eje Y
  step = 20 
  y_lim = MACHINES*step
  gnt.set_ylim(0, y_lim)

  # Configuración de los límites del eje X 
  gnt.set_xlim(0, makespan) 
    
  # Establecer etiquetas para el eje x y el eje y
  gnt.set_xlabel('Time') 
  gnt.set_ylabel('Machine') 
    
  y_ticks = [y for y in range(int(step/2), y_lim, step)] 
  gnt.set_yticks(y_ticks) 

  # Etiquetado de ticks del eje y
  y_labels =  ['M'+str(y+1) for y in range(MACHINES)]
  gnt.set_yticklabels(y_labels) 

  # gnt.grid(True) 

  color_map = {1: 'tab:blue', 
              2: 'tab:orange', 
              3: 'tab:green', 
              4: 'tab:red', 
              5: 'tab:purple',
              6: 'tab:brown', 
              7: 'tab:pink', 
              8: 'tab:gray', 
              9: 'tab:olive', 
              10: 'tab:cyan',
              11: 'b', 
              12: 'g'}

  for op in schedule_df.values:
    lot = op[0]
    operation = op[1]
    machine = op[2]
    start = op[3]
    finish = op[4]

    color = color_map[lot]
    y = y_ticks[machine-1]-int(step/2)

    gnt.broken_barh([(start, finish-start)], (y, step), facecolors =(color), edgecolors=('white'))
    label='O'+str(lot)+','+str(operation)
    gnt.text(x=start + (finish-start)/2, 
                      y=y+int(step/2),
                      s=label, 
                      ha='center', 
                      va='center',
                      color='white',
                    )
  plt.show()

# Función objetivo igual a la última finalización en la programación
def objective_function(individual):
  schedule_df = decode(individual)
  makespan = schedule_df['finish'].max()
  return (makespan),

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list,  fitness=creator.FitnessMin)


toolbox = base.Toolbox()
toolbox.register("indices", random.sample, range(JOBS), JOBS)

# Inicializador individual y de población
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)                  # lista de indivíduos

# Operador inicializador
toolbox.register("evaluate", objective_function) # función objetiva
toolbox.register("mate", tools.cxUniformPartialyMatched, indpb=0.05)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=5)

pop = toolbox.population(n=50)                            # inicio emergente
hof = tools.HallOfFame(1)                                 # mejor chico
stats = tools.Statistics(lambda ind: ind.fitness.values)  # Estadísticas
stats.register("avg", np.mean)
stats.register("std", np.std)
stats.register("min", np.min)
stats.register("max", np.max)

pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.3, mutpb=0.2, ngen=100, stats=stats, halloffame=hof, verbose=True)

gen, min, avg = log.select('gen', 'min', 'avg')
plt.plot(gen, min)
plt.plot(gen, avg)
plt.xlabel('generation')
plt.legend(['minimum makespan', 'average makespan'])
plt.show()

# mejor solución
print("Melhor Indivíduo:")
print(hof[0])
print(decode(hof[0]))
print(decode(hof[0])['finish'].max())

# Mejor resultado de la función objetivo
print("Melhor Resultado da Função Objetivo:")
print(objective_function(hof[0])[0])
gantt(hof[0])
