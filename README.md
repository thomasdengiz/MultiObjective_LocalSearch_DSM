## Content
This repository contains the code for the paper ```Pareto local search for a multi-objective demand response problem in residential areas with heat pumps and electric vehicles```. The paper is published (open access) in the Journal [Energy](https://doi.org/10.1016/j.energy.2025.138063) and the preprint is available on [arXiv](https://arxiv.org/abs/2407.11719). In this paper, we introduce the Pareto local search method PALSS (Pareto local search for load shifting) with heuristic search operations to solve the multi-objective optimization problem of a residential area with different types of flexible loads. PALSS shifts the flexible electricity load with the objective of minimizing the electricity cost and peak load while maintaining the inhabitants’ comfort in favorable ranges. Further, we include reinforcement learning into the heuristic search operations in the approach RELAPALSS (Reinforcement learning assisted Pareto local search) and use the dichotomous method for obtaining all Pareto-optimal solutions of the multi-objective optimization problem with conflicting goals. Our introduces local search methods strongly outperform state-of-the-art metaheuristics like NSGA-II and SPEA-II. Further, the results reveal that integrating a reinforcement learning agent into the decision making improves the performance of local search based control problem for flexible devices like heat pumps and electric vehicles. 


<p align="center">
  <img src="https://github.com/user-attachments/assets/591a7eb2-6aa9-449a-b70f-ce868f5c85c8" alt="Residential_Area_Paper_PLS" style="width: 500px;"/>
</p>
<p align="center"><em> Residential area with three different building types</em></p>


<p align="center">
  <img src="https://github.com/user-attachments/assets/90ba2af1-8e77-40e3-a536-933e78536633" alt="Schema_Pareto_Local_Search" style="width: 500px;"/>
</p>
<p align="center">
  <em> Solution space of a minimization problem with Pareto local search PLS</em>
</p>


<p align="center">
  <img src="https://github.com/user-attachments/assets/9aa2f44e-e2d3-472a-b363-6e27317e479b" alt="Schema_PALSS" style="width: 600px;"/>
</p>
<p align="center">
  <em> Schema of PALSS</em>
</p>




## Setup
The code was tested with Python 3.11 and 3.12. In the [config file](config.py), set up the main data directory (default: "/data") and mainly the input data directory variables and files.

The data can be downloaded [here](https://www.radar-service.eu/radar/en/dataset/ZxeqNfKvVlQcjQAt?token=IZoONgGpjZoiAyiHvtlT#) and the three folders (`Input_Data`, `Reinforcement_Learning`, `Results`) must be placed in the main data directory (default: inside `/data` folder)

You can install the necessary packages listed in the requirements file with

```pip install -r requirements.txt```

Additionally, the [Gurobi solver](https://www.gurobi.com/) is required for the dichotomous method and for the box method. You can also use any other solver for mixed-integer linear programming that is compatible with the optimization framework Pyomo (e.g. the free [GLPK solver](https://www.gnu.org/software/glpk/)). No external solver is necessary for the PALSS and RELAPALSS algorithms.
## First steps / base simulation runs
See also the [notebook file](quick_start.ipynb) for examples of the main functions.



### Set up variables for the method "Pareto local search for load shifting (PALSS)"
 - In the [Run_Simulations_Combined file](Run_Simulations_Combined.py) set up the following boolean variables (directly in the file or after import):
   - ```useCentralizedOptimization``` 
   - ```useConventionalControl```
     - ```useLocalSearch = True``` (default)
 - Execute the function ```run_simulations(...)``` in the [Run_Simulations_Combined file](Run_Simulations_Combined.py) with `withRL = False`
 - Further options:
   - Set up the days. If `calculate_pareto_front_comparisons = True`, the days have to be in the list of those for which a Pareto Front was calculated. You can pass a list of days to the function.
   - Other parameters have to be adjusted directly in the file, e.g.
     - ```max_population_size``` (default 20)
     - ```number_of_pareto_optimal_solutions_in_population``` 
     - ```number_of_new_solutions_per_solution_in_iteration``` (default 3)
     - ```number_of_iterations_local_search``` (default 12)
     - ```time_limit_in_seconds_for_local_search``` (default 10 minutes)
  
### Set up variables for the method "Reinforcement learning assisted Pareto local search (RELAPALSS)"
- Train the peak shift and the price shift operator
    - Options, have to be changed in this [file](RL_Training_One_Shift_OperatorTmp.py) (all have valid default values):
      - Set up training days (specific or random) with ```number_of_days_for_training```,```choose_days_randomly```,```days_for_training```
      - Set up the number of iterations per day with ```number_of_iterations_per_day = 2```
      - Set up number of new solutions per iteration and per solution with ```number_of_new_solutions_per_iteration```,```number_of_new_solutions_per_solution```
      - Number and amount of shifting can also be modified: ```timeslots_for_state_load_percentages_obj```,```number_of_discrete_shifting_actions```,```minimum_shifting_percentage```,```maximum_shifting_percentage```
      - Model will be saved to: ```<models_dir>/trained_PPO_model``` (default: inside ```data/Reinforcement_Learning/RL_Trained_Models```)
    - Start training with ```ml_train_one_shift_operator(isPriceOperator = False)``` for the peak shift operator and ```ml_train_one_shift_operator(isPriceOperator = True)``` for the price shift operator.
  
- Set base variables
  - Set up the correct model names directly in or after the import of the [Run_Simulations_Combined file](Run_Simulations_Combined.py) (```dir_price_shift_model```,```dir_peak_shift_model```) 

  - Execute the function ```run_simulations(withRL = True)``` in the [Run_Simulations_Combined file](Run_Simulations_Combined.py) with `withRL = True`
    - Options:
      - Set up the days. The days should be different from those used for training. If `calculate_pareto_front_comparisons = True`, the days have to be in the list of those for which a Pareto Front was calculated. You can pass a list of days to the function.
      - ...


### Other optimization methods
- In the [Run_Simulations_Combined file](Run_Simulations_Combined.py), you can also use the dichotomous method, the box method (also called epsilon-constraint method) and the conventional control. Therefore, set up the following booleans. 
   - ```useCentralizedOptimization``` 
   - ```useConventionalControl```
   - ```useDichotomicMethodCentralized_Cost_Peak```
   - ```useBoxMethodCentralized_Cost_Peak```
   - ```useLocalSearch```

- Note that the dichotomous method and the box method will require an external solver.

### Additional settings
Change parameters in [this file](SetUpScenarios.py) for the scenarios for the residential area:
- heat pump
- building
- EV
- stationary battery (not used in the paper)
- gas boiler (not used in the paper)
- fan heater (not used in the paper)
- solver options


### NSGA-II and SPEA-II
For comparison, NSGA-II and SPEA-II have also been implemented in [this file](PymooMOEA.py). This part is independent of the proposed PALSS and RELAPALSS algorithms.


## If you use this code, please cite the corresponding paper:

**Thomas Dengiz, Andrea Raith, Max Kleinebrahm, Jonathan Vogl, Wolf Fichtner** (2025):  
*Pareto local search for a multi-objective demand response problem in residential areas with heat pumps and electric vehicles*,  
_Energy_, Volume 335, 30 October 2025, 138063.  
[https://doi.org/10.1016/j.egyai.2024.100441](https://doi.org/10.1016/j.energy.2025.138063)


