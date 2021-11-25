from deap import base, creator, gp, tools, algorithms
import operator, math, random, numpy
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data2.csv")

X1 = df["x1"].to_list()
X2 = df["x2"].to_list()
X3 = df["x3"].to_list()
Y = df["y"].to_list()



def protectedDiv(left, right):
    try:
        return left/right
    except ZeroDivisionError:
        return 1

pset = gp.PrimitiveSet("main", 3)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(protectedDiv, 2)
pset.addPrimitive(operator.neg, 1)
# pset.addPrimitive(math.cos, 1)
# pset.addPrimitive(math.sin, 1)
pset.addEphemeralConstant("c", lambda: random.uniform(-10000, 10000))

pset.renameArguments(ARG0 = 'x1')
pset.renameArguments(ARG1 = 'x2')
pset.renameArguments(ARG2 = 'x3')

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset = pset, min_ = 1, max_ = 2)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset = pset)

def evalSymbReg(individual, points):
    func = toolbox.compile(expr = individual)
    sqerrors = ((func(x1, x2, x3) - y)**2 for x1, x2, x3, y in zip(X1, X2, X3, points))
    return math.fsum(sqerrors)/ len(points),

toolbox.register("evaluate", evalSymbReg, points = Y)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset = pset)

toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value = 3))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value = 3))

stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
stats_size = tools.Statistics(len)
mstats = tools.MultiStatistics(fitness = stats_fit, size = stats_size)
mstats.register("avg", numpy.mean)
mstats.register("std", numpy.std)
mstats.register("min", numpy.min)
mstats.register("max", numpy.max)

pop = toolbox.population(n= 400)
hof = tools.HallOfFame(1)

pop, log = algorithms.eaSimple(pop, toolbox, 0.6, 0.1, 100, stats = mstats, halloffame = hof, verbose=True)

print("Best_fintess_value:",  hof[0].fitness.values, "\nRegression_equation_for_the_best:", str(gp.PrimitiveTree(hof[0])))

func = toolbox.compile(expr = hof[0])

import inspect
print(inspect.getargspec(func))

# x = list(range(1, 255))
# y = [func(i) for i in x]


# plt.plot(x, y)
# plt.scatter(X, Y)
# plt.show()