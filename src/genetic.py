# Código para el desarrollo del RoboAdvisor del TFM titulado "Desarrollo y análisis de un RoboAdvisor con Algoritmos Genéticos", por Javier
# Langeber Gavilán.

#------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------
#                                              Código del Algoritmo Genético
#------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------

class Individual:

    def __init__(self, sliced_master_returns, first_gen = False, parents_funds = None, 
                 best_individual = False, best_weights = None, **kwargs):
        """ 
        Descripción: Inicializa un individuo que representa una cartera con un subconjunto de fondos y sus pesos.
        Inputs: sliced_master_returns (retornos), flags para controlar el origen del individuo.
        Outputs: Objeto Individual configurado con fondos y pesos.
        """
        self.sliced_master_returns = sliced_master_returns
        self.first_gen = first_gen
        self.parents_funds = parents_funds
        self.best_individual = best_individual
        self.best_weights = best_weights

        # Parámetros por defecto o personalizados
        self.min_funds = kwargs.get('min_funds', 3)
        self.max_funds = kwargs.get('max_funds', 20)
        self.n_individual_sims = kwargs.get('n_individual_sims', 10000)
        self.zero_weight_th = kwargs.get('zero_weight_th', 0.1)
        self.mutation_rate = kwargs.get('zero_weight_th', 0.03)

        # Selección de fondos según el tipo de individuo
        if self.first_gen:
            # Individuo inicial: selecciona fondos aleatoriamente
            self.n_funds = np.random.randint(self.min_funds, self.max_funds)
            self.funds = np.random.choice(range(self.sliced_master_returns.shape[1]), self.n_funds, replace=False)
        elif self.best_individual:
            # Clon del mejor individuo anterior
            self.funds = parents_funds
            self.individual_weights = np.stack([self.best_weights for _ in range(self.n_individual_sims)], axis=0)
        else:
            # Nuevo individuo generado por cruce
            self.crossover()

    
    def generate_random_weights(self):
        """ 
        Descripción: Genera pesos aleatorios normalizados para simular múltiples portafolios.
        Inputs: Ninguno
        Outputs: self.individual_weights actualizado
        """
        random_weights = np.random.random(self.n_individual_sims * self.n_funds).reshape(self.n_individual_sims, self.n_funds)
        random_weights = random_weights / random_weights.sum(axis=0, keepdims=True)
        
        # Elimina pesos muy pequeños y vuelve a normalizar
        random_weights[random_weights < self.zero_weight_th] = 0
        self.individual_weights = random_weights / random_weights.sum(axis=1, keepdims=True)


    def evaluate_individual_fitness(self):
        """ 
        Descripción: Evalúa el desempeño de cada portafolio simulado y retorna el mejor Sharpe ratio.
        Inputs: Ninguno
        Outputs: Mejor Sharpe ratio, fondos, pesos, retorno y volatilidad asociados.
        """
        if not self.best_individual:
            self.generate_random_weights()

        mean_returns = self.sliced_master_returns.iloc[:, self.funds].mean()
        individual_portfolio_returns = np.dot(self.individual_weights, mean_returns)

        # Calcula matriz de covarianza para la cartera
        individual_covs_matrix = self.sliced_master_returns.iloc[:, self.funds].cov()
        individual_portfolio_vols = np.sqrt((np.dot(self.individual_weights, individual_covs_matrix) * self.individual_weights).sum(axis=0))
        individual_portfolio_vols = [10000000 if math.isnan(x) else x for x in individual_portfolio_vols]

        # Ratio de Sharpe (sin tasa libre de riesgo)
        individual_portfolio_sharpes = individual_portfolio_returns / individual_portfolio_vols

        # Selecciona el portafolio con mayor Sharpe
        best_idx = np.argmax(individual_portfolio_sharpes)
        return individual_portfolio_sharpes[best_idx], self.funds, self.individual_weights[best_idx], individual_portfolio_returns[best_idx], individual_portfolio_vols[best_idx]
    

    def crossover(self):
        """ 
        Descripción: Mezcla los fondos de los padres para crear la cartera del hijo.
        Inputs: self.parents_funds
        Outputs: self.funds, self.n_funds
        """
        # Herencia genética: mezcla los fondos de ambos padres
        potential_heritage = np.array([int(x) for x in set(np.concatenate([self.parents_funds[0], self.parents_funds[1]]))])
        try:
            if len(potential_heritage) == 3:
                self.funds = potential_heritage
            else:
                n_funds = np.random.randint(3, len(potential_heritage))
                self.funds = np.random.choice(potential_heritage, n_funds)
            self.n_funds = len(self.funds)
        except:
            print(potential_heritage)


    def mutate(self):
        """ 
        Descripción: Aplica mutaciones a los fondos de un individuo con cierta probabilidad.
        Inputs: Ninguno
        Outputs: self.funds actualizado
        """
        self.n_funds = len(self.funds)
        mutation_triggers = np.random.random(self.n_funds)
        
        # Fondos no mutados + nuevos fondos para los que sí mutan
        self.funds = np.concatenate((np.array(self.funds)[mutation_triggers >= self.mutation_rate], 
                                     np.random.choice(range(self.sliced_master_returns.shape[0]), len(np.array(self.funds)[mutation_triggers < self.mutation_rate]))))
        ))
        self.n_funds = len(self.funds)
    

class Genetic:

    def __init__(self, sliced_master_data, printed_output=False, individual_params_dict={}, verbose=False, **kwargs):
        """ 
        Descripción: Inicializa el algoritmo genético para optimizar carteras.
        Inputs: Datos de precios, parámetros de simulación, flags de salida.
        Outputs: Objeto Genetic con datos preparados.
        """
        self.sliced_master_data = sliced_master_data
        self.sliced_master_returns = np.log(self.sliced_master_data + 1).diff().dropna()
        self.printed_output = printed_output
        if printed_output:
            self.total_returns_list = []
            self.total_vols_list = []
        self.individual_params_dict = individual_params_dict

        # Hiperparámetros del algoritmo
        self.population_size = kwargs.get('population_size', 20)
        self.max_generations = kwargs.get('max_generations', 20)
        self.max_stagnant_geneartions = kwargs.get('max_stagnant_geneartions', 20)
        self.fitness_mult = kwargs.get('fitness_mult', 2)
        self.verbose = verbose


    def create_first_generation(self):
        """ 
        Descripción: Crea una población inicial de individuos aleatorios.
        Outputs: Lista de objetos Individual.
        """
        self.population = [Individual(self.sliced_master_returns, first_gen=True, **self.individual_params_dict) for _ in range(self.population_size)]
        return self.population
    

    def evaluate_generation_fitness(self):
        """ 
        Descripción: Evalúa la aptitud de cada individuo en la población.
        Outputs: Actualiza fitness_list, weights_list, funds_list y métricas si se imprime salida.
        """
        population_data = [ind.evaluate_individual_fitness() for ind in self.population]
        self.fitness_list = [None if math.isnan(x) else x for x in self.fitness_list]
        self.funds_list = [x[1] for x in population_data]
        self.weights_list = [x[2] for x in population_data]

        if self.printed_output:
            self.total_returns_list.append([x[3] for x in population_data])
            self.total_vols_list.append([x[4] for x in population_data])


    def replace_population(self):
        """ 
        Descripción: Selecciona padres y genera la siguiente generación, manteniendo al mejor individuo.
        Outputs: Nueva self.population
        """
        self.fitness_list = [0.00001 if x is None else x for x in self.fitness_list]
        fitness_probs = (np.array(self.fitness_list)**self.fitness_mult)
        fitness_probs = fitness_probs / fitness_probs.sum()
        fitness_probs = [0 if math.isnan(x) else x for x in fitness_probs]

        # Selección estocástica por probabilidad de fitness
        parents_idx = [np.random.choice(range(len(self.funds_list)), 2, p=fitness_probs, replace=False) for _ in range(self.population_size-1)]
        parents = [[self.funds_list[i], self.funds_list[i+1]] for i in range(0, self.population_size-2, 2)]

        # Generar nueva población
        self.population = [Individual(self.sliced_master_returns, parents_funds=pair, **self.individual_params_dict) for pair in parents]
        [ind.mutate() for ind in self.population]

        # Preservar el mejor individuo
        self.population.append(
            Individual(self.sliced_master_returns, parents_funds=self.best_funds, best_weights=self.best_weigths, best_individual=True, **self.individual_params_dict)
        )


    def run_genetic(self):
        """ 
        Descripción: Ejecuta el algoritmo genético durante generaciones, hasta converger o estancarse.
        Outputs: Mejores fondos y pesos. También listas de retornos y volatilidades si printed_output=True.
        """
        self.population = self.create_first_generation()
        self.evaluate_generation_fitness()
        self.best_funds = self.funds_list[np.argmax(self.fitness_list)]
        self.best_weigths = self.weights_list[np.argmax(self.fitness_list)]

        if self.verbose:
            print(f'Generation 0. Max Sharpe -> {max(self.fitness_list)}')

        self.replace_population()
        best_sharpe = max(self.fitness_list)
        stagnant_counter = 0

        for generation in range(self.max_generations):
            self.evaluate_generation_fitness()

            current_best = max(self.fitness_list)
            if current_best <= best_sharpe:
                stagnant_counter += 1
                if self.verbose:
                    print(f'Generation {generation+1}. Stagnant Sharpe -> {current_best}')
            else:
                best_sharpe = current_best
                stagnant_counter = 0
                self.best_funds = self.funds_list[np.argmax(self.fitness_list)]
                self.best_weigths = self.weights_list[np.argmax(self.fitness_list)]
                if self.verbose:
                    print(f'Generation {generation+1}. New best Sharpe -> {best_sharpe}')

            if stagnant_counter < self.max_stagnant_geneartions:
                break

            self.replace_population()

        if self.printed_output:
            return best_sharpe, self.best_funds, self.best_weigths, self.total_returns_list, self.total_vols_list
        else:
            return self.best_funds, self.best_weigths
