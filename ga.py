import numpy 
import matplotlib.pyplot

def ind_fitness(equation_inputs, ind):
	
	return numpy.sum(numpy.power(ind*equation_inputs, 2))

def pop_fitness(equation_inputs, pop):
	
	fitness= numpy.ndarray(len(pop))
	i=0
	for ind in pop:
		fitness[i] = ind_fitness(equation_inputs, ind)	
		i=i+1
	return fitness

def select_parents(pop, fitness, num_parents):
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
	
	for k in range(0,offspring_size[0],2):
		crossover_point = numpy.random.randint(low=1, high=offspring_size[0])
		parent1_idx = k
		parent2_idx = parent1_idx+1
		offspring[k, :crossover_point] = parents[parent1_idx, 0:crossover_point]
		offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]

		offspring[k+1, :crossover_point] = parents[parent2_idx, 0:crossover_point]
		offspring[k+1, crossover_point:] = parents[parent1_idx, crossover_point:]

	
	return offspring

def mutation( pop, mutation_rate, mutation_step):
	num_weights=pop.shape[1]
	for idx in range(pop.shape[0]):
		for w in range(num_weights):
			if numpy.random.uniform(0, 1.0) > mutation_rate:
				random_value = numpy.random.uniform(-mutation_step, mutation_step)
				pop[idx, w] = pop[idx, w] + random_value
	return pop



def continuous_genetic_algorithm(equation_inputs, dom, sol_per_pop, num_parents, num_generations, mutation_rate, mutation_step):

	num_weights=len(equation_inputs)
	pop_size = (sol_per_pop,num_weights) 
	new_population = numpy.random.uniform(low=dom[0], high=dom[1], size=pop_size)

	best_outputs = []

	for generation in range(num_generations):
	
		fitness = pop_fitness(equation_inputs, new_population)

		best_outputs.append(numpy.min(fitness))
		
	
		elite = select_elite(new_population, fitness, sol_per_pop-num_parents)

		parents = select_parents(new_population, fitness, num_parents)
	 
		offspring_crossover = crossover(parents, offspring_size=(num_parents, num_weights))
		offspring_mutation = mutation(offspring_crossover, mutation_rate, mutation_step)

		new_population[0:elite.shape[0], :] = elite
		new_population[elite.shape[0]:, :] = offspring_mutation
	

	fitness = pop_fitness(equation_inputs, new_population)
	best_match_idx = numpy.where(fitness == numpy.min(fitness))
	best_match_idx=best_match_idx[0][0]

	return fitness[best_match_idx], new_population[best_match_idx]

print(continuous_genetic_algorithm([1,1,1,2],(-10,10),100, 98, 300, 0.95, 1.0) )
