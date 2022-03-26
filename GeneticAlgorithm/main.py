import random
import statistics

from data_loading import *
from Models.Machine import Machine
from collections import namedtuple
import numpy as np, pandas as pd, matplotlib.pyplot as plt

instances = get_instances()
Specimen = namedtuple('Specimen', 'machines fitness')


def generate_clean_machines(instance):
    machines = []
    for i in range(instance.machinesNumb):
        machines.append(Machine(i, -1, -1))
    return machines


def generate_clean_grid(instance):
    fields = []
    for x in range(instance.dimX):
        for y in range(instance.dimY):
            fields.append((x, y))
    return fields


def create_random_specimen(instance):
    machines = generate_clean_machines(instance)
    grid = generate_clean_grid(instance)
    for i in range(instance.machinesNumb):
        random_field = random.choice(grid)
        machines[i].x = random_field[0]
        machines[i].y = random_field[1]
        grid.remove(random_field)
    return Specimen(machines, check_fitness(instance, machines))


def create_random_population(instance, population_size):
    population = []
    for i in range(population_size):
        population.append(create_random_specimen(instance))
    return population


def get_machine(specimen, number):
    return [x for x in specimen if x.number == number][0]


def check_fitness(instance, specimens_machines):
    paths = get_paths_list(instance)
    costs_sum = 0
    for i in range(len(paths)):
        source = get_machine(specimens_machines, paths[i].source)
        dest = get_machine(specimens_machines, paths[i].dest)
        distance = abs(source.x - dest.x) + abs(source.y - dest.y)
        product = paths[i].cost * paths[i].amount * distance
        costs_sum += product
    return costs_sum


def tournament_selection(population, tournament_size):
    tournament = random.choices(population, k=tournament_size)
    best_specimen = population[0]
    for spec in tournament:
        if spec.fitness < best_specimen.fitness:
            best_specimen = spec
    return best_specimen


def roulette_wheel_selection(population):
    population_fitness = sum([specimen.fitness for specimen in population])
    probabilities = [specimen.fitness / population_fitness for specimen in population]
    inverse_probabilities = [(1 / pow(x, 4)) for x in probabilities]
    inverse_probabilities = [x / sum(inverse_probabilities) for x in inverse_probabilities]
    chosen_index = np.random.choice(len(population), p=inverse_probabilities)
    return population[chosen_index]


def breed(instance, parent1, parent2, breed_rate):
    if random.random() < breed_rate:
        child_p1, child_p2, used_machines = [], [], []
        grid = generate_clean_grid(instance)
        cut_point = int(random.random() * len(parent1.machines))

        for i in range(2, cut_point):
            child_p1.append(parent1.machines[i])
            used_machines.append(parent1.machines[i].number)
            grid.remove((parent1.machines[i].x, parent1.machines[i].y))

        for m in parent2.machines:
            if m.number not in used_machines:
                if (m.x, m.y) in grid:
                    child_p2.append(m)
                    grid.remove((m.x, m.y))
                    used_machines.append(m.number)

        for m in parent2.machines:
            if m.number not in used_machines:
                place = int(random.random() * len(grid))
                child_p2.append(Machine(m.number, grid[place][0], grid[place][1]))
                grid.remove(grid[place])

        child = child_p1 + child_p2
        return Specimen(child, check_fitness(instance, child))
    else:
        return Specimen(parent1.machines, check_fitness(instance, parent1.machines))


def is_ok(spec):
    numbers = []
    for m in spec.machines:
        numbers.append(m.number)
    if len(numbers) > len(set(numbers)):
        return False
    return True


def mutate(instance, specimen, mutation_rate):
    if random.random() < mutation_rate:
        child, used_machines = [], []
        grid = generate_clean_grid(instance)

        for i in range(len(specimen.machines) - 3):
            child.append(specimen.machines[i])
            used_machines.append(specimen.machines[i].number)
            grid.remove((specimen.machines[i].x, specimen.machines[i].y))

        for m in specimen.machines:
            if m.number not in used_machines:
                place = int(random.random() * len(grid))
                child.append(Machine(m.number, grid[place][0], grid[place][1]))
                grid.remove(grid[place])

        return Specimen(child, check_fitness(instance, child))
    return specimen


def crete_next_generation_tournament(instance, population, tournament_size, mutation_rate, breed_rate):
    next_generation = []
    for i in range(len(population)):
        parent1 = tournament_selection(population, tournament_size)
        parent2 = tournament_selection(population, tournament_size)
        child = breed(instance, parent1, parent2, breed_rate)
        child = mutate(instance, child, mutation_rate)
        next_generation.append(child)
    return next_generation


def crete_next_generation_roulette(instance, population, mutation_rate, breed_rate):
    next_generation = []
    for i in range(len(population)):
        parent1 = roulette_wheel_selection(population)
        parent2 = roulette_wheel_selection(population)
        child = breed(instance, parent1, parent2, breed_rate)
        child = mutate(instance, child, mutation_rate)
        next_generation.append(child)
    return next_generation


def genetic_algorithm_tournament(instance, population_size, tournament_size, mutation_rate, breed_rate, generations):
    print('Instance - ' + instance.name)
    print('Tournament Selection')
    population = create_random_population(instance, population_size)
    avg_fitness, best_fitness, worst_fitness = [], [], []
    avg_fitness.append(sum(map(lambda x: x.fitness, population)) / len(population))
    best_specimen = min(population, key=lambda t: t[1])
    best_fitness.append(best_specimen.fitness)
    worst_specimen = max(population, key=lambda t: t[1])
    worst_fitness.append(worst_specimen.fitness)

    for i in range(0, generations):
        population = crete_next_generation_tournament(instance, population, tournament_size, mutation_rate, breed_rate)
        avg_fitness.append(sum(map(lambda x: x.fitness, population)) / len(population))
        best_specimen = min(population, key=lambda t: t[1])
        best_fitness.append(best_specimen.fitness)
        worst_specimen = max(population, key=lambda t: t[1])
        worst_fitness.append(worst_specimen.fitness)

    print('Best: ' + str(best_fitness[generations]))
    print('Average: ' + str(avg_fitness[generations]))
    print('Worst: ' + str(worst_fitness[generations]))

    plt.plot(avg_fitness, label='Average')
    plt.plot(best_fitness, label='Best')
    plt.plot(worst_fitness, label='Worst')
    plt.ylabel('Fitness')
    plt.xlabel('Generation')
    plt.legend()
    plt.show()
    return avg_fitness[generations], best_fitness[generations], worst_fitness[generations]


def genetic_algorithm_roulette(instance, population_size, mutation_rate, breed_rate, generations):
    print('Instance - ' + instance.name)
    print('Roulette Wheel Selection')
    population = create_random_population(instance, population_size)
    avg_fitness, best_fitness, worst_fitness = [], [], []
    avg_fitness.append(sum(map(lambda x: x.fitness, population)) / len(population))
    best_specimen = min(population, key=lambda t: t[1])
    best_fitness.append(best_specimen.fitness)
    worst_specimen = max(population, key=lambda t: t[1])
    worst_fitness.append(worst_specimen.fitness)

    for i in range(0, generations):
        population = crete_next_generation_roulette(instance, population, mutation_rate, breed_rate)
        avg_fitness.append(sum(map(lambda x: x.fitness, population)) / len(population))
        best_specimen = min(population, key=lambda t: t[1])
        best_fitness.append(best_specimen.fitness)
        worst_specimen = max(population, key=lambda t: t[1])
        worst_fitness.append(worst_specimen.fitness)

    print('Best: ' + str(best_fitness[generations]))
    print('Average: ' + str(avg_fitness[generations]))
    print('Worst: ' + str(worst_fitness[generations]))

    plt.plot(avg_fitness, label='Average')
    plt.plot(best_fitness, label='Best')
    plt.plot(worst_fitness, label='Worst')
    plt.ylabel('Fitness')
    plt.xlabel('Generation')
    plt.legend()
    plt.show()
    return avg_fitness[generations], best_fitness[generations], worst_fitness[generations]


def Average(lst):
    return sum(lst) / len(lst)


def avg_tournament(instance, population_size, tournament_size, mutation_rate, breed_rate, generations):
    print('Instance - ' + instance.name)
    print('Tournament Selection')
    avg_results = []
    best_results = []
    worst_results = []
    for i in range(10):
        res = genetic_algorithm_tournament(instance, population_size, tournament_size, mutation_rate, breed_rate,
                                           generations)
        avg_results.append(res[0])
        best_results.append(res[1])
        worst_results.append(res[2])
    std = statistics.stdev(avg_results)
    print('std: ' + str(std))
    print('best: ' + str(Average(best_results)))
    print('avg: ' + str(Average(avg_results)))
    print('worst: ' + str(Average(worst_results)))


def avg_roulette(instance, population_size, mutation_rate, breed_rate, generations):
    print('Instance - ' + instance.name)
    print('Roulette Selection')
    avg_results = []
    best_results = []
    worst_results = []
    for i in range(10):
        res = genetic_algorithm_roulette(instance, population_size, mutation_rate, breed_rate, generations)
        avg_results.append(res[0])
        best_results.append(res[1])
        worst_results.append(res[2])
    std = statistics.stdev(avg_results)
    print('std: ' + str(std))
    print('best: ' + str(Average(best_results)))
    print('avg: ' + str(Average(avg_results)))
    print('worst: ' + str(Average(worst_results)))


def avg_random(instance, population_size):
    print('Instance - ' + instance.name)
    print('Random population')
    avg_results = []
    best_results = []
    worst_results = []
    for i in range(100):
        res = create_random_population(instance, population_size)
        population = res
        avg_fitness, best_fitness, worst_fitness = [], [], []
        avg_fitness.append(sum(map(lambda x: x.fitness, population)) / len(population))
        best_specimen = min(population, key=lambda t: t[1])
        best_fitness.append(best_specimen.fitness)
        worst_specimen = max(population, key=lambda t: t[1])
        worst_fitness.append(worst_specimen.fitness)
        avg_results.append(avg_fitness[0])
        best_results.append(best_fitness[0])
        worst_results.append(worst_fitness[0])
    std = statistics.stdev(avg_results)
    print('std: ' + str(std))
    print('best: ' + str(Average(best_results)))
    print('avg: ' + str(Average(avg_results)))
    print('worst: ' + str(Average(worst_results)))


if __name__ == '__main__':
    genetic_algorithm_tournament(instances[2], 200, 3, 0.1, 0.9, 300)
    avg_tournament(instances[0], 500, 3, 0.1, 0.8, 100)
    print('\n')
    genetic_algorithm_roulette(instances[2], 200, 0.1, 0.9, 300)
    avg_roulette(instances[2], 100, 0.1, 0.8, 300)
    avg_random(instances[1], 1000)
