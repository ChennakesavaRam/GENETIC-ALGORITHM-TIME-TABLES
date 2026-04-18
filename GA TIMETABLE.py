import random
from deap import base, creator, tools, algorithms

# -----------------------------
# DATA FOR TIMETABLE
# -----------------------------
courses = [
    ("Math", "Dr.Smith"),
    ("Physics", "Dr.John"),
    ("Chemistry", "Dr.Alice"),
    ("Biology", "Dr.Brown"),
    ("CS", "Dr.David")
]

rooms = ["R1", "R2", "R3"]
timeslots = ["Mon1", "Mon2", "Tue1", "Tue2", "Wed1"]

NUM_CLASSES = len(courses)

# -----------------------------
# FITNESS FUNCTION SETUP
# -----------------------------
if "FitnessMax" not in creator.__dict__:
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))

if "Individual" not in creator.__dict__:
    creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

toolbox.register("timeslot", random.randint, 0, len(timeslots)-1)
toolbox.register("room", random.randint, 0, len(rooms)-1)

def create_class():
    return (toolbox.timeslot(), toolbox.room())

toolbox.register("individual",
                 tools.initRepeat,
                 creator.Individual,
                 create_class,
                 n=NUM_CLASSES)

toolbox.register("population",
                 tools.initRepeat,
                 list,
                 toolbox.individual)

# -----------------------------
# CUSTOM MUTATION AND EVALUATION
# -----------------------------
def custom_mutation(individual):
    """Randomly reassigns timeslot and room for a gene with 0.2 probability."""
    for i in range(len(individual)):
        if random.random() < 0.2:
            individual[i] = (random.randint(0, len(timeslots)-1),
                             random.randint(0, len(rooms)-1))
    return individual,

def custom_evaluate(individual):
    """Calculates penalties for room and teacher conflicts."""
    penalty = 0
    for i in range(len(individual)):
        t1, r1 = individual[i]
        teacher1 = courses[i][1]

        for j in range(i+1, len(individual)):
            t2, r2 = individual[j]
            teacher2 = courses[j][1]

            # Conflict Logic
            if t1 == t2 and r1 == r2:
                penalty += 10
            if t1 == t2 and teacher1 == teacher2:
                penalty += 10

    return (100 - penalty,)

# -----------------------------
# GENETIC OPERATORS REGISTRATION
# -----------------------------
toolbox.register("evaluate", custom_evaluate)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", custom_mutation)
toolbox.register("select", tools.selTournament, tournsize=3)

# -----------------------------
# MAIN PROGRAM
# -----------------------------
def main():
    population = toolbox.population(n=100)
    NGEN, CXPB, MUTPB = 50, 0.7, 0.2

    print("Starting Evolution...")

    # Initial Evaluation
    fitnesses = list(map(toolbox.evaluate, population))
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit

    for gen in range(NGEN):
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))

        # Crossover
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        # Mutation
        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate invalid individuals
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        population[:] = offspring

    # Results
    best_ind = tools.selBest(population, 1)[0]
    print(f"\nBest Fitness: {best_ind.fitness.values[0]}")
    print("\nDecoded Timetable:")
    for i, (ts_idx, r_idx) in enumerate(best_ind):
        print(f"  {courses[i][0]} ({courses[i][1]}): {timeslots[ts_idx]} in {rooms[r_idx]}")

if __name__ == "__main__":
    main()