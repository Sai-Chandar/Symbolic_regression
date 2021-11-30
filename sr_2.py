from deap import base, creator, gp, tools, algorithms
import operator, math, random, numpy
from numpy.core.numeric import Inf
import pandas as pd
import matplotlib.pyplot as plt
import inspect

df = pd.read_csv("data2.csv")

X1 = df["x1"].to_list()
X2 = df["x2"].to_list()
X3 = df["x3"].to_list()
Y = df["y"].to_list()



def best_hof(hof_list):
    best = Inf
    best_index = 0
    for i in range(len(hof_list)):
        if hof_list[i].fitness.values[0] <= best:
            best = hof_list[i].fitness.values[0]
            best_index = i
    return hof_list[best_index]


def protectedDiv(left, right):
    try:
        return left/right
    except ZeroDivisionError:
        return 1

#pset1 -> [x, add, sub, mul, protectedDiv, neg, EphemeralConstant]
#pset2 -> [x, mul, EphemeralConstant]
#pset3 -> [x, protectedDiv, EphemeralConstant]


pset = gp.PrimitiveSet("main", 3)
# pset.addPrimitive(operator.add, 2)
# pset.addPrimitive(operator.sub, 2)
# pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(protectedDiv, 2)
# pset.addPrimitive(operator.neg, 1)
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
    # print(inspect.getargspec(func))
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


def main(run: int):
    pop = toolbox.population(n= POP)
    hof = tools.HallOfFame(1)

    pop, log = algorithms.eaSimple(pop, toolbox, CXPB, MUTPB, NGEN, stats = mstats, halloffame = hof, verbose=False)

    # print("Best_fintess_value:",  hof[0].fitness.values, "\nRegression_equation_for_the_best:", str(gp.PrimitiveTree(hof[0])))

    #code for average convergence data
    col_name = "fit_run_{0}".format(run)
    data = log.chapters["fitness"].select("gen", "min")
    df[col_name] = data[1]

    #keeping track of best sol for each run
    hof_list.append(hof[0])
    local_hof.append(hof[0])




NGEN = 100


hof_list = []

# [300, 400, 500]
# [0.4, 0.6, 0.8]
# [0.1, 0.3, 0.5]

for pop in [300, 400, 500]:
    local_hof = []
    for cx in [0.4, 0.6, 0.8]:
        for mt in [0.1, 0.3, 0.5]:
            CXPB = cx
            MUTPB = mt
            POP = pop

            gens = list(range(0, NGEN+1))
            df = pd.DataFrame(gens, columns = ["gens"])

            for i in range(30):
                main(run = i)

            df.to_csv("./solutions/problem2/cov_data/cov_data_pset3_{0}_{1}_{2}.csv".format(CXPB, MUTPB, POP))

            print("Done with {0}, {1}, {2} case.".format(mt, cx, pop))
    
    pop_best = best_hof(local_hof)
    d = {'beat_fitness': [pop_best.fitness.values[0]], 'reg_equation': [str(gp.PrimitiveTree(pop_best))] }
    pop_sol = pd.DataFrame(data= d)
    pop_sol.to_csv("./solutions/problem2/pset3_pop_sol_{0}.csv".format(pop))
    


# print(hof_list)

best_sol = best_hof(hof_list)
print("best sol:", best_sol.fitness.values)
d = {'beat_fitness': [best_sol.fitness.values[0]], 'reg_equation': [str(gp.PrimitiveTree(best_sol))] }
sol = pd.DataFrame(data= d)
sol.to_csv("./solutions/problem2/sol_pset3.csv")



# import inspect
# print(inspect.getargspec(func))

# x = list(range(1, 255))
# y = [func(i) for i in x]


# plt.plot(x, y)
# plt.scatter(X, Y)
# plt.show()