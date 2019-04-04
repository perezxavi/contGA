import numpy 
import matplotlib.pyplot

def ind_fitness(equation_inputs, ind):
	
	return numpy.sum(numpy.power(ind*equation_inputs, 2))

def pop_fitness(equation_inputs, pop):
	# Calculating the fitness value of each solution in the current population.
	fitness= numpy.ndarray(len(pop))
	i=0
	for ind in pop:
		fitness[i] = ind_fitness(equation_inputs, ind)	
		i=i+1
	return fitness

def select_parents(pop, fitness, num_parents):
	# Selecting the best individuals in the current generation as parents for producing the offspring of the next generation.
	parents = numpy.empty((num_parents, pop.shape[1]))
	
	for parent_num in range(num_parents):
		idx_player_1 = numpy.random.randint(0, pop.shape[0])
		idx_player_2 = numpy.random.randint(0, pop.shape[0])
	
		idx_winner =  idx_player_1
		if fitness[idx_player_1] > fitness[idx_player_2]:
			idx_winner =  idx_player_2

		parents[parent_num, :] = pop[idx_winner, :]
	return parents

def select_elite(pop, fitness, num_ind):
	# Selecting the best individuals in the current generation as parents for producing the offspring of the next generation.
	elite = numpy.empty((num_ind, pop.shape[1]))
	
	aux_fitness=fitness[:]
	
	for ind in range(num_ind):

		best_fitness_idx = numpy.where(aux_fitness == numpy.min(aux_fitness))
		best_fitness_idx = best_fitness_idx[0][0]
		aux_fitness[best_fitness_idx] = numpy.Inf
		
		elite[ind, :] = pop[best_fitness_idx, :]
	   
	return elite





def crossover(parents, offspring_size):
	offspring = numpy.empty(offspring_size)
	# The point at which crossover takes place between two parents. Usually, it is at the center.
	
	for k in range(0,offspring_size[0],2):
		crossover_point = numpy.random.randint(low=1, high=offspring_size[0])
		#print(crossover_point)
		# Index of the first parent to mate.
		parent1_idx = k
		# Index of the second parent to mate.
		parent2_idx = parent1_idx+1
		# The new offspring will have its first half of its genes taken from the first parent.
		offspring[k, :crossover_point] = parents[parent1_idx, 0:crossover_point]
		# The new offspring will have its second half of its genes taken from the second parent.
		offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]

		# The new offspring will have its first half of its genes taken from the first parent.
		offspring[k+1, :crossover_point] = parents[parent2_idx, 0:crossover_point]
		# The new offspring will have its second half of its genes taken from the second parent.
		offspring[k+1, crossover_point:] = parents[parent1_idx, crossover_point:]

	
	return offspring

def mutation(offspring_crossover):
	# Mutation changes a single gene in each offspring randomly.
	global num_weights
	for idx in range(offspring_crossover.shape[0]):
		for w in range(num_weights):
			if numpy.random.uniform(0, 1.0) > 0.95:
				# The random value to be added to the gene.
				random_value = numpy.random.uniform(-10.0, 10.0)
				offspring_crossover[idx, w] = offspring_crossover[idx, w] + random_value
	return offspring_crossover



def continuous_genetic_algorithm(equation_inputs, sol_per_pop, num_parents, num_generations

	# Defining the population size.
	pop_size = (sol_per_pop,num_weights) 
	#Creating the initial population.
	new_population = numpy.random.uniform(low=-1000.0, high=1000.0, size=pop_size)

	best_outputs = []

	for generation in range(num_generations):
	
		# Measuring the fitness of each chromosome in the population.
		fitness = pop_fitness(equation_inputs, new_population)

		best_outputs.append(numpy.min(fitness))
		# The best result in the current iteration.
		print("Best result : ", numpy.min(fitness))
	
		elite = select_elite(new_population, fitness, sol_per_pop-num_parents)

		# Selecting the best parents in the population.
		parents = select_parents(new_population, fitness, num_parents)
	 
		offspring_crossover = crossover(parents, offspring_size=(num_parents, num_weights))
		offspring_mutation = mutation(offspring_crossover)

		new_population[0:elite.shape[0], :] = elite
		new_population[elite.shape[0]:, :] = offspring_mutation
	

	fitness = pop_fitness(equation_inputs, new_population)
	best_match_idx = numpy.where(fitness == numpy.min(fitness))

	return fitness, new_population[best_match_idx, :]


