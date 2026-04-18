import random
from deap import base, creator, tools, algorithms

# ---------------------------------------------------------
# 1. DYNAMIC INPUT COLLECTION
# ---------------------------------------------------------
def get_user_input():
    print("--- Timetable Configuration ---")
    
    # Courses and Teachers
    num_courses = int(input("Enter number of courses: "))
    courses = []
    for i in range(num_courses):
        name = input(f"  Name of course {i+1}: ")
        teacher = input(f"  Teacher for {name}: ")
        courses.append((name, teacher))
    
    # Rooms
    num_rooms = int(input("\nEnter number of rooms: "))
    rooms = [input(f"  Enter Room Name {i+1}: ") for i in range(num_rooms)]
    
    # Real-Time Timeslots
    print("\n--- Timeslot Configuration ---")
    start_hour = int(input("Enter starting hour (e.g., 9 for 9 AM): "))
    num_slots = int(input("How many consecutive hours of classes?: "))
    
    timeslots = []
    for i in range(num_slots):
        current_hour = start_hour + i
        period = "AM" if current_hour < 12 else "PM"
        display_hour = current_hour if current_hour <= 12 else current_hour - 12
        if display_hour == 0: display_hour = 12
        timeslots.append(f"{display_hour}:00 {period}")
    
    return courses, rooms, timeslots

# Execution starts here to define global variables for the GA
courses, rooms, timeslots = get_user_input()
NUM_CLASSES = len(courses)

# ---------------------------------------------------------
# 2. GENETIC ALGORITHM SETUP (Global Scope)
# ---------------------------------------------------------
if "FitnessMax" not in creator.__dict__:
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))

if "Individual" not in creator.__dict__:
    creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

# Registering attributes
toolbox.register("timeslot_idx", random.randint, 0, len(timeslots)-1)
toolbox.register("room_idx", random.randint, 0, len(rooms)-1)

def create_class():
    return (toolbox.timeslot_idx(), toolbox.room_idx())

toolbox.register("individual", tools.initRepeat, creator.Individual, create_class, n=NUM_CLASSES)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# ---------------------------------------------------------
# 3. EVALUATION AND MUTATION LOGIC
# ---------------------------------------------------------
def custom_evaluate(individual):
    """Checks for room and teacher conflicts at the same time."""
    penalty = 0
    for i in range(len(individual)):
        t1, r1 = individual[i]
        teacher1 = courses[i][1]

        for j in range(i+1, len(individual)):
            t2, r2 = individual[j]
            teacher2 = courses[j][1]

            # Conflict: Same time and same room
            if t1 == t2 and r1 == r2:
                penalty += 10
            # Conflict: Same time and same teacher
            if t1 == t2 and teacher1 == teacher2:
                penalty += 10

    return (100 - penalty,)

def custom_mutation(individual):
    """Randomly reassigns time or room for a class."""
    for i in range(len(individual)):
        if random.random() < 0.2:
            individual[i] = (random.randint(0, len(timeslots)-1),
                             random.randint(0, len(rooms)-1))
    return individual,

toolbox.register("evaluate", custom_evaluate)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", custom_mutation)
toolbox.register("select", tools.selTournament, tournsize=3)

# ---------------------------------------------------------
# 4. MAIN PROGRAM EXCECUTION
# ---------------------------------------------------------
def main():
    # Now 'toolbox' is accessible because it's in the global scope
    population = toolbox.population(n=100)
    NGEN, CXPB, MUTPB = 50, 0.7, 0.2

    print("\nEvolution in progress...")

    # Using eaSimple for a clean evolutionary loop
    algorithms.eaSimple(population, toolbox, cxpb=CXPB, mutpb=MUTPB, ngen=NGEN, verbose=False)

    # Pull the best result
    best_ind = tools.selBest(population, 1)[0]
    
    print("\n" + "="*50)
    print("                FINAL TIMETABLE")
    print("="*50)
    
    # Logic to sort classes by time for a better UX
    scheduled_classes = []
    for i, (ts_idx, r_idx) in enumerate(best_ind):
        scheduled_classes.append((ts_idx, i, r_idx))
    
    scheduled_classes.sort() # Sorts by timeslot index

    for ts_idx, c_idx, r_idx in scheduled_classes:
        time_label = timeslots[ts_idx]
        course_name = courses[c_idx][0]
        teacher_name = courses[c_idx][1]
        room_name = rooms[r_idx]
        
        print(f" {time_label.ljust(8)} | {course_name.ljust(15)} | {teacher_name.ljust(15)} | Room: {room_name}")
    
    print("="*50)
    print(f"Solution Fitness Score: {best_ind.fitness.values[0]}/100")

if __name__ == "__main__":
    main()