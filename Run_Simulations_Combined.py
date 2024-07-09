"""
Main script for the simulations of the Pareto-Local-Search approach, the dichotomous method
, the box method (another exact method for multiobjective optization; also called the epsilon-constraint method) and the conventional control

Main result folder is stored on file (path is specified by the variable "folderPath_WholeSimulation")
"""
from torch.testing._internal.distributed.distributed_utils import with_dist

import SetUpScenarios
import numpy as np
import os
from datetime import datetime
import time


import config

#Set up

# Specify the used control methods
useCentralizedOptimization = True #only w/o RL
useConventionalControl = True #only w/o RL
useDichotomicMethodCentralized_Cost_Peak = False
useDichotomicMethodCentralized_Cost_Comfort = False
dichotomicMethodTermination_NumberOfIterations = 100# if withRL else 70
dichotomicMethod_toleranceLambdaNewSolution = 0.001
dichotomicMethodprintListOfActivePoints = False

#Specifiy Mutiobjective optimization method
useBoxMethodCentralized_Cost_Peak = False
useBoxMethodCentralized_Cost_Comfort = False
boxMethodTermination_AverageDifference = 0
boxMethodTermination_NumberOfSolutions = 70

global useLocalSearch
useLocalSearch = True
calculate_pareto_front_comparisons = True

create_result_load_profiles_multi_opt = True #only w/o RL
scaleResultsWeightedSum = False
font_size_title_Pareto_Plot = 14


#Local Search Parameters
max_population_size = 20
number_of_pareto_optimal_solutions_in_population = int(max_population_size * 0.6)
number_of_new_solutions_per_solution_in_iteration =3
number_of_iterations_local_search = 12
time_limit_in_seconds_for_local_search = 10*60 #2*10if withRL else 10 * 60

share_of_cost_min_iterations = 0.6
share_of_peak_min_iterations = 0.3
share_of_comfort_max_iterations = 0.1

threshold_discomfort_local_search = 0.3

#Objectives and scenarios defined by the dictionary "optParameters"
optParameters = {
    'optimizationGoal_minimizePeakLoad': False,
    'optimizationGoal_minimizeCosts': True,
    'optimizationGoal_minimizeGas': False,
    'optimizationGoal_minimizeThermalDiscomfort': False,
    'optimizationGoal_minimizeSurplusEnergy': False,
    'optimization_1Objective': True,
    'optimization_2Objective': False,
    'optimization_3Objectives': False,
    'optimization_4Objectives': False,
    'objective_minimizePeakLoad_normalizationValue': 1,
    'objective_minimizeCosts_normalizationValue': 1,
    'objective_minimizeThermalDiscomfort_normalizationValue': 1,
    'objective_minimizeSurplusEnergy_normalizationValue': 1,
    'objective_minimizePeakLoad_weight': 0.5,
    'objective_minimizeCosts_weight': 0.5,
    'objective_minimizeThermalDiscomfort_weight': 0.5,
    'objective_minimizeSurplusEnergy_weight': 0.5,
    'epsilon_objective_minimizeCosts_Active': False, # specifies if the epsilon constraint method (for the objective cost) is used
    'epsilon_objective_minimizePeakLoad_Active': False,
    'epsilon_objective_minimizeGasConsumption_Active': False ,
    'epsilon_objective_minimizeThermalDiscomfort_Active':False,
    'epsilon_objective_minimizeCosts_TargetValue': 9999999,  # specifies the target value for epsilon constraint method (for the objective cost). If the current solution is below that value, the method terminates
    'epsilon_objective_minimizeMaximumLoad_TargetValue': 9999999,
    'epsilon_objective_minimizeGasConsumption_TargetValue': 9999999,
    'epsilon_objective_minimizeThermalDiscomfort_TargetValue': 9999999
}

# RL - trained models
global dir_price_shift_model,dir_peak_shift_model
dir_price_shift_model = None
dir_peak_shift_model = None




includeObjectivesInReturnStatementCentralized = False #Default
useCorrectionsAtTheEndOfDay = False



# Choose internal Controller
run_simulateDays_NoAddtionalController_ANNSchedule = False
run_simulateDays_WithAddtionalController_Schedule = True
useInternalControllerToOverruleActions_simulateDays_WithAddtionalController_Schedule = True

useInternalControllerForRL = True



#Maximal starting times for the heating devices
considerMaximumNumberOfStartsHP_Combined = True
considerMaxiumNumberOfStartsHP_Individual = False
considerMaxiumNumberOfStartsHP_MFH_Individual = True
considerMaxiumNumberOfStartsGas_Individual = True
isHPAlwaysSwitchedOn = False
maximumNumberOfStarts_Combined = 4
maximumNumberOfStarts_Individual = 2

#Parameters for the internal controller
minimalRunTimeHeatPump = 0
timeslotsForCorrectingActionsBeforeTheAndOfTheDay = 0
numberOfTimeSlotHeatingWithMinModulationDegreeWhenStartingToHeat =0

#Adjust parameters to the time resolution
if SetUpScenarios.timeResolution_InMinutes ==1:
    minimalRunTimeHeatPump = 30
    timeslotsForCorrectingActionsBeforeTheAndOfTheDay = 0
    numberOfTimeSlotHeatingWithMinModulationDegreeWhenStartingToHeat = 5
if SetUpScenarios.timeResolution_InMinutes ==5:
    minimalRunTimeHeatPump = 6
    timeslotsForCorrectingActionsBeforeTheAndOfTheDay = 0
    numberOfTimeSlotHeatingWithMinModulationDegreeWhenStartingToHeat = 4
if SetUpScenarios.timeResolution_InMinutes ==10:
    minimalRunTimeHeatPump = 3
    timeslotsForCorrectingActionsBeforeTheAndOfTheDay = 0
    numberOfTimeSlotHeatingWithMinModulationDegreeWhenStartingToHeat = 4
if SetUpScenarios.timeResolution_InMinutes ==15:
    minimalRunTimeHeatPump = 2
    timeslotsForCorrectingActionsBeforeTheAndOfTheDay = 0
    numberOfTimeSlotHeatingWithMinModulationDegreeWhenStartingToHeat = 2
if SetUpScenarios.timeResolution_InMinutes ==30:
    minimalRunTimeHeatPump = 1
    timeslotsForCorrectingActionsBeforeTheAndOfTheDay = 0
    numberOfTimeSlotHeatingWithMinModulationDegreeWhenStartingToHeat = 1
if SetUpScenarios.timeResolution_InMinutes ==60:
    minimalRunTimeHeatPump = 1
    timeslotsForCorrectingActionsBeforeTheAndOfTheDay = 0
    numberOfTimeSlotHeatingWithMinModulationDegreeWhenStartingToHeat = 1
additionalNumberOfAllowedStarts = 7
additionalNumberOfAllowedStarts_BeforeConsideringMinimalRuntime = 3

if useCorrectionsAtTheEndOfDay ==False:
    timeslotsForCorrectingActionsBeforeTheAndOfTheDay = -1

minimalModulationDegreeOfTheMaximumPowerInCaseOfANecessaryCorrection = 0.7 # Should be between 0.5 and 1. Only quantifies the minimal degree. If no new peak is created by this action in the simulation , the value will be higher

daysOfTheYearForSimulation_Testing =[1]# [ 5] if withRL else [1]

#Run simulations
def run_simulations(withRL, days_for_simulation = None):
    global useLocalSearch,dir_peak_shift_model, dir_price_shift_model
    import Building_Optimization_Problem
    import ICSimulation
    import pandas as pd
    import random
    import pickle

    #Meaure the time needed
    import time
    start_time = time.time()

    if days_for_simulation is None: #default
        days_for_simulation = [9,11] #  [9, 11, 15, 23, 39, 45, 55, 72, 80, 292, 303, 314, 319, 328, 332, 346,350, 361]
    df_results_multiopt_local_search = pd.DataFrame(columns=["Day", "GD_Conventional", "GD PF_Approx", "HV PF_Approx", "HV PF_Full", "HV_Ratio"])
    currentDatetimeString = datetime.today().strftime('%d_%m_%Y_Time_%H_%M_%S')
    simulationName = "RL2_RL3" if withRL else "Base"

    #Run the simulations for the different days
    for currentDay_iteration in days_for_simulation:
        currentDay = currentDay_iteration
        print(f"Current Day: {currentDay}")
        # define the directory to be created for the result files
        folderName_WholeSimulation = currentDatetimeString + "_" + simulationName + "_Min"+ str(round(time_limit_in_seconds_for_local_search/60)) +"_BTCombined_" + str(SetUpScenarios.numberOfBuildings_Total)
        folderPath_resultFile_multiOpt = os.path.join(config.DIR_INSTANCE_BASE, folderName_WholeSimulation)
        folderPath_WholeSimulation = os.path.join(config.DIR_INSTANCE_BASE,folderName_WholeSimulation + "/Day" + str(currentDay))
        pathForCreatingTheResultData_Dichotromic = folderPath_WholeSimulation + "/Dichotromic"
        pathForCreatingTheResultData_LocalSearch = folderPath_WholeSimulation + "/LocalSearch"
        pathForCreatingTheResultData_Box = folderPath_WholeSimulation + "/BoxMethod"
        pathForCreatingTheResultData_Decentralized = folderPath_WholeSimulation + "/Decentralized"
        pathForCreatingTheResultData_SupervisedML = folderPath_WholeSimulation + "/ML"
        pathForCreatingTheResultData_RL = folderPath_WholeSimulation + "/RL"
        pathForCreatingTheResultData_Conventional = folderPath_WholeSimulation + "/Conventional"

        try:
            os.makedirs(folderPath_WholeSimulation)
            if not withRL:
                if useConventionalControl == True:
                    os.makedirs(pathForCreatingTheResultData_Conventional)
                if useDichotomicMethodCentralized_Cost_Peak == True:
                    os.makedirs(pathForCreatingTheResultData_Dichotromic)
                if useBoxMethodCentralized_Cost_Peak == True:
                    os.makedirs(pathForCreatingTheResultData_Box)
            if useLocalSearch == True:
                os.makedirs(pathForCreatingTheResultData_LocalSearch)

        except OSError:
            print ("Creation of the directory %s failed" % folderPath_WholeSimulation)
        else:
            print ("Successfully created the directory %s" % folderPath_WholeSimulation)


        #Exact methods centralized (testing)
        if useCentralizedOptimization == True or withRL:
            print("\n--------Centralized Optimization-------\n")
            includeObjectivesInReturnStatementCentralized = False

            indexOfBuildingsOverall_BT1 = [i for i in range (1, SetUpScenarios.numberOfBuildings_BT1 + 1)]
            indexOfBuildingsOverall_BT2 = [i for i in range (1, SetUpScenarios.numberOfBuildings_BT2 + 1)]
            indexOfBuildingsOverall_BT3 = [i for i in range (1, SetUpScenarios.numberOfBuildings_BT3 + 1)]
            indexOfBuildingsOverall_BT4 = [i for i in range (1, SetUpScenarios.numberOfBuildings_BT4 + 1 )]
            indexOfBuildingsOverall_BT5 = [i for i in range (1, SetUpScenarios.numberOfBuildings_BT5 + 1 )]
            indexOfBuildingsOverall_BT6 = [i for i in range (1, SetUpScenarios.numberOfBuildings_BT6 + 1 )]
            indexOfBuildingsOverall_BT7 = [i for i in range (1, SetUpScenarios.numberOfBuildings_BT7 + 1 )]

            #Exectue (Pareto) Local Search
            if useLocalSearch == True:
                # Measure wall-clock time and CPU time
                start_time = time.time()
                start_cpu = time.process_time()

                results_list_archive= []
                results_list_population = []
                results_list_per_iteration = []


                indexOfBuildingsOverall_BT1 = [i + 1 for i in range(0, SetUpScenarios.numberOfBuildings_BT1)]
                indexOfBuildingsOverall_BT2 = [i + 1 for i in range(0, SetUpScenarios.numberOfBuildings_BT2)]
                indexOfBuildingsOverall_BT3 = [i + 1 for i in range(0, SetUpScenarios.numberOfBuildings_BT3)]
                indexOfBuildingsOverall_BT4 = [i + 1 for i in range(0, SetUpScenarios.numberOfBuildings_BT4)]
                indexOfBuildingsOverall_BT5 = [i + 1 for i in range(0, SetUpScenarios.numberOfBuildings_BT5)]
                indexOfBuildingsOverall_BT6 = [i + 1 for i in range(0, SetUpScenarios.numberOfBuildings_BT6)]
                indexOfBuildingsOverall_BT7 = [i + 1 for i in range(0, SetUpScenarios.numberOfBuildings_BT7)]

                df_results_archive = pd.DataFrame(columns=["id of the run", "Costs", "Peak Load", "Thermal Discomfort"])
                df_results_population = pd.DataFrame(columns=["id of the run", "Costs", "Peak Load", "Thermal Discomfort"])
                df_results_per_iteration = pd.DataFrame(columns=["id of the run", "Costs", "Peak Load", "Thermal Discomfort"])
                id_of_the_run = 1


                # run simulation environment with conventional control
                simulationObjective_surplusEnergy_kWh_combined, simulationObjective_maximumLoad_kW_combined, simulationObjective_thermalDiscomfort_combined, simulationObjective_gasConsumptionkWh_combined, simulationObjective_costs_Euro_combined, simulationObjective_combinedScore_combined, simulationResult_electricalLoad_combined, price_array, simulationInput_BT1_availabilityPattern, combined_array_thermal_discomfort, outputVector_BT1_heatGenerationCoefficientSpaceHeating, outputVector_BT1_heatGenerationCoefficientDHW, outputVector_BT1_chargingPowerEV, outputVector_BT2_heatGenerationCoefficientSpaceHeating, outputVector_BT2_heatGenerationCoefficientDHW, outputVector_BT3_chargingPowerEV, outputVector_BT4_heatGenerationCoefficientSpaceHeating, outputVector_BT5_chargingPowerBAT, outputVector_BT5_disChargingPowerBAT, outputVector_BT6_heatGenerationCoefficient_GasBoiler, outputVector_BT6_heatGenerationCoefficient_ElectricalHeatingElement, outputVector_BT6_heatTransferCoefficient_StorageToRoom, outputVector_BT7_heatGenerationCoefficient_GasBoiler, outputVector_BT7_electricalPowerFanHeater, combined_array_thermal_discomfort, simulationResult_thermalDiscomfort_BT1, simulationResult_thermalDiscomfort_BT2, simulationResult_thermalDiscomfort_BT3, simulationResult_thermalDiscomfort_BT4, simulationResult_thermalDiscomfort_BT5, simulationResult_thermalDiscomfort_BT6, simulationResult_thermalDiscomfort_BT7 = ICSimulation.simulateDays_ConventionalControl(indexOfBuildingsOverall_BT1, indexOfBuildingsOverall_BT2, indexOfBuildingsOverall_BT3, indexOfBuildingsOverall_BT4, indexOfBuildingsOverall_BT5, indexOfBuildingsOverall_BT6, indexOfBuildingsOverall_BT7, currentDay, pathForCreatingTheResultData_Conventional, useLocalSearch)
                df_results_per_iteration.loc[id_of_the_run - 1] = [id_of_the_run, simulationObjective_costs_Euro_combined, simulationObjective_maximumLoad_kW_combined, simulationObjective_thermalDiscomfort_combined]

                help_value_normalization_maxiumLoad_conventional = simulationObjective_maximumLoad_kW_combined
                help_value_normalization_cost_conventional = simulationObjective_costs_Euro_combined


                #run simulation with light controller
                preCorrectSchedules_AvoidingFrequentStarts = False
                overruleActions = False
                pathForCreatingTheResultData = pathForCreatingTheResultData_LocalSearch + "/Profiles/Solution_ID_Base"
                returned_objects  = ICSimulation.simulateDays_WithLightController_Schedule(indexOfBuildingsOverall_BT1, indexOfBuildingsOverall_BT2, indexOfBuildingsOverall_BT3, indexOfBuildingsOverall_BT4, indexOfBuildingsOverall_BT5, indexOfBuildingsOverall_BT6, indexOfBuildingsOverall_BT7, currentDay, outputVector_BT1_heatGenerationCoefficientSpaceHeating, outputVector_BT1_heatGenerationCoefficientDHW, outputVector_BT1_chargingPowerEV, outputVector_BT2_heatGenerationCoefficientSpaceHeating, outputVector_BT2_heatGenerationCoefficientDHW, outputVector_BT3_chargingPowerEV, outputVector_BT4_heatGenerationCoefficientSpaceHeating, outputVector_BT5_chargingPowerBAT, outputVector_BT5_disChargingPowerBAT, outputVector_BT6_heatGenerationCoefficient_GasBoiler, outputVector_BT6_heatGenerationCoefficient_ElectricalHeatingElement, outputVector_BT6_heatTransferCoefficient_StorageToRoom, outputVector_BT7_heatGenerationCoefficient_GasBoiler, outputVector_BT7_electricalPowerFanHeater, pathForCreatingTheResultData, preCorrectSchedules_AvoidingFrequentStarts, optParameters, useLocalSearch)

                #Set initial solutoin for the population
                results_dict = {"simulationObjective_surplusEnergy_kWh_combined": returned_objects[0], "simulationObjective_maximumLoad_kW_combined": returned_objects[1], "simulationObjective_thermalDiscomfort_combined": returned_objects[2], "simulationObjective_gasConsumptionkWh_combined": returned_objects[3], "simulationObjective_costs_Euro_combined": returned_objects[4], "simulationObjective_combinedScore_combined": returned_objects[5], "simulationResult_electricalLoad_combined": returned_objects[6], "price_array": returned_objects[7], "simulationInput_BT1_availabilityPattern": returned_objects[8], "combined_array_thermal_discomfort": returned_objects[9], "outputVector_BT1_heatGenerationCoefficientSpaceHeating": returned_objects[10], "outputVector_BT1_heatGenerationCoefficientDHW": returned_objects[11], "outputVector_BT1_chargingPowerEV": returned_objects[12], "outputVector_BT2_heatGenerationCoefficientSpaceHeating": returned_objects[13], "outputVector_BT2_heatGenerationCoefficientDHW": returned_objects[14], "outputVector_BT3_chargingPowerEV": returned_objects[15], "outputVector_BT4_heatGenerationCoefficientSpaceHeating": returned_objects[16], "outputVector_BT5_chargingPowerBAT": returned_objects[17], "outputVector_BT5_disChargingPowerBAT": returned_objects[18], "outputVector_BT6_heatGenerationCoefficient_GasBoiler": returned_objects[19], "outputVector_BT6_heatGenerationCoefficient_ElectricalHeatingElement": returned_objects[20], "outputVector_BT6_heatTransferCoefficient_StorageToRoom": returned_objects[21], "outputVector_BT7_heatGenerationCoefficient_GasBoiler": returned_objects[22], "outputVector_BT7_electricalPowerFanHeater": returned_objects[23], "combined_array_thermal_discomfort": returned_objects[24], "thermal_discomfort_space_heating_BT1": returned_objects[25], "thermal_discomfort_dhw_BT1": returned_objects[26], "thermal_discomfort_space_heating_BT2": returned_objects[27], "thermal_discomfort_dhw_BT2": returned_objects[28], "thermal_discomfort_space_heating_BT4": returned_objects[29], "id of the run": id_of_the_run}
                results_list_population.append(results_dict)
                df_results_population.loc[id_of_the_run - 1] = [id_of_the_run, results_dict['simulationObjective_costs_Euro_combined'],  results_dict['simulationObjective_maximumLoad_kW_combined'], results_dict['simulationObjective_thermalDiscomfort_combined']]



                #Do the local search iteration

                id_of_the_run = 1
                termination_flag_active = False
                id_of_the_iterations = 1
                for index_iteration in range (0, number_of_iterations_local_search):
                    if termination_flag_active == True:
                        break

                    print("")
                    print("Local Search Iteration: " + str(index_iteration + 1))
                    df_results_per_iteration = df_results_per_iteration.drop(df_results_per_iteration.index)
                    results_list_per_iteration = []
                    #merged_list_population_current_iteration = []
                    if index_iteration ==0:
                        number_of_new_solutions_per_solution = 9
                    else:
                        number_of_new_solutions_per_solution = number_of_new_solutions_per_solution_in_iteration

                    #Store the current population
                    file_path = os.path.join(config.DIR_RL_INPUT, "list_population_NB" + str(SetUpScenarios.numberOfBuildings_Total) + "_Day" + str(currentDay) + "_It" + str(index_iteration) + ".pkl")
                    with open(file_path, "wb+") as file:
                        pickle.dump(results_list_population, file)

                    for index_solution_in_population in range (0, len (results_list_population)):
                        #Check termination condition
                        if termination_flag_active == True:
                            break
                        print(f"solution_in_population: {index_solution_in_population}", end='')

                        # set the solution vector (original solution) for the next solution
                        outputVector_BT1_heatGenerationCoefficientSpaceHeating_original = results_list_population[index_solution_in_population]['outputVector_BT1_heatGenerationCoefficientSpaceHeating'].copy()
                        outputVector_BT1_heatGenerationCoefficientDHW_original = results_list_population[index_solution_in_population]['outputVector_BT1_heatGenerationCoefficientDHW'].copy()
                        outputVector_BT1_chargingPowerEV_original = results_list_population[index_solution_in_population]['outputVector_BT1_chargingPowerEV'].copy()
                        outputVector_BT2_heatGenerationCoefficientSpaceHeating_original = results_list_population[index_solution_in_population]['outputVector_BT2_heatGenerationCoefficientSpaceHeating'].copy()
                        outputVector_BT2_heatGenerationCoefficientDHW_original = results_list_population[index_solution_in_population]['outputVector_BT2_heatGenerationCoefficientDHW'].copy()
                        outputVector_BT3_chargingPowerEV_original = results_list_population[index_solution_in_population]['outputVector_BT3_chargingPowerEV'].copy()
                        outputVector_BT4_heatGenerationCoefficientSpaceHeating_original = results_list_population[index_solution_in_population]['outputVector_BT4_heatGenerationCoefficientSpaceHeating'].copy()
                        outputVector_BT5_chargingPowerBAT_original = results_list_population[index_solution_in_population]['outputVector_BT5_chargingPowerBAT'].copy()
                        outputVector_BT5_disChargingPowerBAT_original = results_list_population[index_solution_in_population]['outputVector_BT5_disChargingPowerBAT'].copy()
                        outputVector_BT6_heatGenerationCoefficient_GasBoiler_original = results_list_population[index_solution_in_population]['outputVector_BT6_heatGenerationCoefficient_GasBoiler'].copy()
                        outputVector_BT6_heatGenerationCoefficient_ElectricalHeatingElement_original = results_list_population[index_solution_in_population]['outputVector_BT6_heatGenerationCoefficient_ElectricalHeatingElement'].copy()
                        outputVector_BT6_heatTransferCoefficient_StorageToRoom_original = results_list_population[index_solution_in_population]['outputVector_BT6_heatTransferCoefficient_StorageToRoom'].copy()
                        outputVector_BT7_heatGenerationCoefficient_GasBoiler_original = results_list_population[index_solution_in_population]['outputVector_BT7_heatGenerationCoefficient_GasBoiler'].copy()
                        outputVector_BT7_electricalPowerFanHeater_original = results_list_population[index_solution_in_population]['outputVector_BT7_electricalPowerFanHeater'].copy()

                        for index_solution_local_seach in range (0, number_of_new_solutions_per_solution):
                            #print(f"index_solution_local_seach: {index_solution_local_seach}")

                            #Set the vectors of decision variables to the original one of the current solution in the population
                            outputVector_BT1_heatGenerationCoefficientSpaceHeating = outputVector_BT1_heatGenerationCoefficientSpaceHeating_original.copy()
                            outputVector_BT1_heatGenerationCoefficientDHW = outputVector_BT1_heatGenerationCoefficientDHW_original.copy()
                            outputVector_BT1_chargingPowerEV = outputVector_BT1_chargingPowerEV_original.copy()
                            outputVector_BT2_heatGenerationCoefficientSpaceHeating = outputVector_BT2_heatGenerationCoefficientSpaceHeating_original.copy()
                            outputVector_BT2_heatGenerationCoefficientDHW = outputVector_BT2_heatGenerationCoefficientDHW_original.copy()
                            outputVector_BT3_chargingPowerEV = outputVector_BT3_chargingPowerEV_original.copy()
                            outputVector_BT4_heatGenerationCoefficientSpaceHeating = outputVector_BT4_heatGenerationCoefficientSpaceHeating_original.copy()
                            outputVector_BT5_chargingPowerBAT = outputVector_BT5_chargingPowerBAT_original.copy()
                            outputVector_BT5_disChargingPowerBAT = outputVector_BT5_disChargingPowerBAT_original.copy()
                            outputVector_BT6_heatGenerationCoefficient_GasBoiler = outputVector_BT6_heatGenerationCoefficient_GasBoiler_original.copy()
                            outputVector_BT6_heatGenerationCoefficient_ElectricalHeatingElement = outputVector_BT6_heatGenerationCoefficient_ElectricalHeatingElement_original.copy()
                            outputVector_BT6_heatTransferCoefficient_StorageToRoom = outputVector_BT6_heatTransferCoefficient_StorageToRoom_original.copy()
                            outputVector_BT7_heatGenerationCoefficient_GasBoiler = outputVector_BT7_heatGenerationCoefficient_GasBoiler_original.copy()
                            outputVector_BT7_electricalPowerFanHeater_ = outputVector_BT7_electricalPowerFanHeater_original.copy()

                            #Test Temp: print variable of the base solution before the modifications to file
                            returned_objects = ICSimulation.simulateDays_WithLightController_Schedule(indexOfBuildingsOverall_BT1, indexOfBuildingsOverall_BT2, indexOfBuildingsOverall_BT3, indexOfBuildingsOverall_BT4, indexOfBuildingsOverall_BT5, indexOfBuildingsOverall_BT6, indexOfBuildingsOverall_BT7, currentDay, outputVector_BT1_heatGenerationCoefficientSpaceHeating, outputVector_BT1_heatGenerationCoefficientDHW, outputVector_BT1_chargingPowerEV, outputVector_BT2_heatGenerationCoefficientSpaceHeating, outputVector_BT2_heatGenerationCoefficientDHW, outputVector_BT3_chargingPowerEV, outputVector_BT4_heatGenerationCoefficientSpaceHeating, outputVector_BT5_chargingPowerBAT, outputVector_BT5_disChargingPowerBAT, outputVector_BT6_heatGenerationCoefficient_GasBoiler, outputVector_BT6_heatGenerationCoefficient_ElectricalHeatingElement, outputVector_BT6_heatTransferCoefficient_StorageToRoom, outputVector_BT7_heatGenerationCoefficient_GasBoiler, outputVector_BT7_electricalPowerFanHeater, pathForCreatingTheResultData, preCorrectSchedules_AvoidingFrequentStarts, optParameters, useLocalSearch)
                            if index_solution_local_seach ==0:
                                print(f", Costs: {returned_objects[4]}, Peak: {returned_objects[1]}, DC: {returned_objects[2]}")
                            electrical_load_profile_current_solution = returned_objects[6]
                            availability_pattern_EV_BT1 = returned_objects [8]
                            thermal_discomfort_space_heating_BT1 = returned_objects[25]
                            thermal_discomfort_dhw_BT1 = returned_objects[26]
                            thermal_discomfort_space_heating_BT2= returned_objects[27]
                            thermal_discomfort_dhw_BT2= returned_objects[28]
                            thermal_discomfort_space_heating_BT4= returned_objects[29]

                            id_of_the_run+=1


                            #Choose the search direction (optimization goal)
                            random_float_search_direction = random.uniform(0, 1)
                            if random_float_search_direction < share_of_cost_min_iterations:
                                optimize_costs_local_search = True
                                optimize_peak_local_search = False
                                optimize_comfort_local_search = False

                            if random_float_search_direction >= share_of_cost_min_iterations and random_float_search_direction < share_of_cost_min_iterations + share_of_peak_min_iterations:
                                optimize_costs_local_search = False
                                optimize_peak_local_search = True
                                optimize_comfort_local_search = False
                            if random_float_search_direction >=  share_of_cost_min_iterations + share_of_peak_min_iterations and random_float_search_direction < share_of_cost_min_iterations + share_of_peak_min_iterations + share_of_comfort_max_iterations:
                                optimize_costs_local_search = False
                                optimize_peak_local_search = False
                                optimize_comfort_local_search = True




                        #Local Search: Optimize for costs (to be tested and loop necessary for multiple solutions)
                            #Local Search: Optimize for costs (to be tested and loop necessary for multiple solutions)
                            if optimize_costs_local_search == True:
                                # Find the timeslots with the highgest prices
                                k = int(60 / SetUpScenarios.timeResolution_InMinutes)  # number of highest price timeslots to select

                                # create arrays with highes and lowest prices for 1/1 day
                                k = int(60 /SetUpScenarios.timeResolution_InMinutes) #Price stages per part of timeslots: One price for every hour (day-ahead market) leads to the 60. In case of intraday market data it would be 15/timeResolutoin
                                sorted_indices_ascending = np.argsort(-price_array)
                                sorted_indices_descending = np.argsort(price_array)

                                highest_k_prices_array_full_day_1 = np.array([sorted_indices_ascending[:k],sorted_indices_ascending[k:2 * k],sorted_indices_ascending[2 * k:3 * k], sorted_indices_ascending[3 * k:4 * k],sorted_indices_ascending[4 * k:5 * k],sorted_indices_ascending[5 * k:6 * k],sorted_indices_ascending[6 * k:7 * k],sorted_indices_ascending[7 * k:8 * k],sorted_indices_ascending[8 * k:9 * k],sorted_indices_ascending[9 * k:10 * k],sorted_indices_ascending[10 * k:11 * k],sorted_indices_ascending[11 * k:12 * k]])
                                lowest_k_prices_array_full_day_1 = np.array([sorted_indices_descending[:k],sorted_indices_descending[k:2 * k],sorted_indices_descending[2 * k:3 * k], sorted_indices_descending[3 * k:4 * k],sorted_indices_descending[4 * k:5 * k],sorted_indices_descending[5 * k:6 * k],sorted_indices_descending[6 * k:7 * k],sorted_indices_descending[7 * k:8 * k],sorted_indices_descending[8 * k:9 * k],sorted_indices_descending[9 * k:10 * k],sorted_indices_descending[10 * k:11 * k],sorted_indices_descending[11 * k:12 * k]])

                                #Create array with percentrages of loads (in relation to the total load) for the highest and lowest price timeslots
                                sum_of_loads = np.sum(electrical_load_profile_current_solution)
                                reshaped_array_highest_prices = highest_k_prices_array_full_day_1.reshape((24, 1))
                                reshaped_array_lowest_prices = lowest_k_prices_array_full_day_1.reshape((24, 1))
                                percentage_array_loads_per_timeslot_highest_prices = np.zeros(len(reshaped_array_highest_prices))
                                for i in range (0, len(reshaped_array_highest_prices)):
                                    percentage_array_loads_per_timeslot_highest_prices = np.round((electrical_load_profile_current_solution[reshaped_array_highest_prices] / sum_of_loads) * 100, 1)
                                percentage_array_loads_per_timeslot_lowest_prices = np.zeros(len(reshaped_array_lowest_prices))
                                for i in range (0, len(reshaped_array_lowest_prices)):
                                    percentage_array_loads_per_timeslot_lowest_prices = np.round((electrical_load_profile_current_solution[reshaped_array_lowest_prices] / sum_of_loads) * 100, 1)

                                if not withRL:

                                    #create the arrays with the highes and lowest prices for 1/2 day
                                    sorted_indices_half_day_part_1_ascending = np.argsort(-price_array[0: int(len(price_array)*0.5)])
                                    sorted_indices_half_day_part_2_ascending  = np.argsort(-price_array[int(len(price_array)*0.5):])
                                    sorted_indices_half_day_part_2_ascending += int(len(price_array)*0.5)
                                    sorted_indices_half_day_part_1_descending = np.argsort(price_array[0: int(len(price_array)*0.5)])
                                    sorted_indices_half_day_part_2_descending  = np.argsort(price_array[int(len(price_array)*0.5):])
                                    sorted_indices_half_day_part_2_descending += int(len(price_array) * 0.5)

                                    highest_k_prices_array_half_day_part_1 = np.array([sorted_indices_half_day_part_1_ascending[:k],sorted_indices_half_day_part_1_ascending[k:2 * k],sorted_indices_half_day_part_1_ascending[2 * k:3 * k], sorted_indices_half_day_part_1_ascending[3 * k:4 * k],sorted_indices_half_day_part_1_ascending[4 * k:5 * k],sorted_indices_half_day_part_1_ascending[5 * k:6 * k]])
                                    highest_k_prices_array_half_day_part_2 = np.array([sorted_indices_half_day_part_2_ascending[:k],sorted_indices_half_day_part_2_ascending[k:2 * k],sorted_indices_half_day_part_2_ascending[2 * k:3 * k], sorted_indices_half_day_part_2_ascending[3 * k:4 * k],sorted_indices_half_day_part_2_ascending[4 * k:5 * k],sorted_indices_half_day_part_2_ascending[5 * k:6 * k]])
                                    highest_k_prices_half_day_combined_array = np.stack([ highest_k_prices_array_half_day_part_1, highest_k_prices_array_half_day_part_2], axis=0)
                                    lowest_k_prices_array_half_day_part_1 = np.array([sorted_indices_half_day_part_1_descending[:k],sorted_indices_half_day_part_1_descending[k:2 * k],sorted_indices_half_day_part_1_descending[2 * k:3 * k], sorted_indices_half_day_part_1_descending[3 * k:4 * k],sorted_indices_half_day_part_1_descending[4 * k:5 * k],sorted_indices_half_day_part_2_descending[5 * k:6 * k]])
                                    lowest_k_prices_array_half_day_part_2 = np.array([sorted_indices_half_day_part_2_descending[:k],sorted_indices_half_day_part_2_descending[k:2 * k],sorted_indices_half_day_part_2_descending[2 * k:3 * k], sorted_indices_half_day_part_2_descending[3 * k:4 * k],sorted_indices_half_day_part_2_descending[4 * k:5 * k],sorted_indices_half_day_part_2_descending[5 * k:6 * k]])
                                    lowest_k_prices_half_day_combined_array = np.stack([lowest_k_prices_array_half_day_part_1, lowest_k_prices_array_half_day_part_2], axis=0)

                                    # create the arrays with the highes and lowest prices for 1/3 day
                                    sorted_indices_onethird_day_part_1_ascending = np.argsort(-price_array[0: int(len(price_array)*0.33)])
                                    sorted_indices_onethird_day_part_2_ascending  = np.argsort(-price_array[int(len(price_array)*0.33):int(len(price_array)*0.66)])
                                    sorted_indices_onethird_day_part_2_ascending += int(len(price_array)*0.33)
                                    sorted_indices_onethird_day_part_3_ascending = np.argsort(-price_array[int(len(price_array) * 0.66):])
                                    sorted_indices_onethird_day_part_3_ascending += int(len(price_array) * 0.66)

                                    sorted_indices_onethird_day_part_1_descending = np.argsort(price_array[0: int(len(price_array)*0.33)])
                                    sorted_indices_onethird_day_part_2_descending  = np.argsort(price_array[int(len(price_array)*0.33):int(len(price_array)*0.66)])
                                    sorted_indices_onethird_day_part_2_descending += int(len(price_array)*0.33)
                                    sorted_indices_onethird_day_part_3_descending = np.argsort(price_array[int(len(price_array) * 0.66):])
                                    sorted_indices_onethird_day_part_3_descending += int(len(price_array) * 0.66)

                                    highest_k_prices_array_onethird_day_part_1 = np.array([sorted_indices_onethird_day_part_1_ascending [:k],sorted_indices_onethird_day_part_1_ascending [k:2 * k],sorted_indices_onethird_day_part_1_ascending [2 * k:3 * k], sorted_indices_onethird_day_part_1_ascending [3 * k:4 * k]])
                                    highest_k_prices_array_onethird_day_part_2 = np.array([sorted_indices_onethird_day_part_2_ascending [:k],sorted_indices_onethird_day_part_2_ascending [k:2 * k],sorted_indices_onethird_day_part_2_ascending [2 * k:3 * k], sorted_indices_onethird_day_part_2_ascending [3 * k:4 * k]])
                                    highest_k_prices_array_onethird_day_part_3 = np.array([sorted_indices_onethird_day_part_3_ascending [:k],sorted_indices_onethird_day_part_3_ascending [k:2 * k],sorted_indices_onethird_day_part_3_ascending [2 * k:3 * k], sorted_indices_onethird_day_part_3_ascending [3 * k:4 * k]])
                                    highest_k_prices_onethird_day_combined_array =np.stack([highest_k_prices_array_onethird_day_part_1, highest_k_prices_array_onethird_day_part_2, highest_k_prices_array_onethird_day_part_3], axis=0)

                                    lowest_k_prices_array_onethird_day_part_1 = np.array([sorted_indices_onethird_day_part_1_descending[:k],sorted_indices_onethird_day_part_1_descending[k:2 * k],sorted_indices_onethird_day_part_1_descending[2 * k:3 * k], sorted_indices_onethird_day_part_1_descending[3 * k:4 * k]])
                                    lowest_k_prices_array_onethird_day_part_2 = np.array([sorted_indices_onethird_day_part_2_descending[:k],sorted_indices_onethird_day_part_2_descending[k:2 * k],sorted_indices_onethird_day_part_2_descending[2 * k:3 * k], sorted_indices_onethird_day_part_2_descending[3 * k:4 * k]])
                                    lowest_k_prices_array_onethird_day_part_3 = np.array([sorted_indices_onethird_day_part_3_descending[:k],sorted_indices_onethird_day_part_3_descending[k:2 * k],sorted_indices_onethird_day_part_3_descending[2 * k:3 * k], sorted_indices_onethird_day_part_3_descending[3 * k:4 * k]])
                                    lowest_k_prices_onethird_day_combined_array =np.stack([lowest_k_prices_array_onethird_day_part_1, lowest_k_prices_array_onethird_day_part_2, lowest_k_prices_array_onethird_day_part_3], axis=0)


                                    #Create the shifting percentages within one interval (3. Parameter of the local seach algorithm)
                                    percentage_shifted_loads = np.zeros(12)
                                    threshold_random_number_dhw_heating = 85 # Unit %

                                    random_index_shifting_timeslot_1 = random.randint(0, 11)
                                    random_index_shifting_timeslot_2 = random.randint(0, 11)

                                    timeslots = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
                                    weights = [50-index_iteration*2, 40-index_iteration*2, 15+index_iteration*2, 10+index_iteration*2, 7+index_iteration, 6+index_iteration*0.5, 5+index_iteration*0.5, 4+index_iteration*0.5, 3+index_iteration*0.25, 2+index_iteration*0.25, 1+index_iteration*0.25, 0+index_iteration*0.5]

                                    random_index_shifting_timeslot_1 = random.choices(timeslots, weights=weights, k=1)[0]
                                    random_index_shifting_timeslot_2 = random.choices(timeslots, weights=weights, k=1)[0]

                                    percentage_shifted_loads [random_index_shifting_timeslot_1] = random.uniform(20-index_iteration, 40-index_iteration*2)
                                    percentage_shifted_loads[random_index_shifting_timeslot_2] = random.uniform(20-index_iteration, 40-index_iteration*2)

                                    # Print Help Info
                                    '''
                                    print(f"Info: random_index_shifting_timeslot_1: {random_index_shifting_timeslot_1}")
                                    print(f"Info: random_index_shifting_timeslot_2: {random_index_shifting_timeslot_2}")
                                    print(f"Info: percentage_shifted_loads_1: {round(percentage_shifted_loads [random_index_shifting_timeslot_1],1)}")
                                    print(f"Info: percentage_shifted_loads_2: {round(percentage_shifted_loads[random_index_shifting_timeslot_2],1)}")
                                    '''

                                    random_number_param_1 = random.random()

                                    help_threshould_1 = 1.6
                                    help_thrshould_2_addition = 0.3
                                    if random_number_param_1 < help_threshould_1:
                                        param_1_number_of_partitions_for_shifting = 1
                                    if random_number_param_1 >= help_threshould_1 and random_number_param_1 < help_threshould_1 + help_thrshould_2_addition:
                                        param_1_number_of_partitions_for_shifting = 2
                                    if random_number_param_1 >= help_threshould_1 + help_thrshould_2_addition:
                                        param_1_number_of_partitions_for_shifting = 3

                                    if param_1_number_of_partitions_for_shifting == 1:
                                        param_2_number_of_intervales_within_partition = 12
                                    if param_1_number_of_partitions_for_shifting == 2:
                                        param_2_number_of_intervales_within_partition = 6
                                    if param_1_number_of_partitions_for_shifting == 3:
                                        param_2_number_of_intervales_within_partition = 4

                                    selected_indices_param_2_number_of_intervales_within_partition = np.where(percentage_shifted_loads > 0.1)[0].tolist()

                                    for helpIteration in range (0,2):
                                        #Shift profiles cosindering the full day partition
                                        if param_1_number_of_partitions_for_shifting == 1:

                                            for current_price_inverval in selected_indices_param_2_number_of_intervales_within_partition:
                                                # Change the profiles of BT1
                                                for index_BT1 in indexOfBuildingsOverall_BT1:
                                                    help_balance_time_slot = helpIteration
                                                    #Shift Space heating power from highes price timeslot to lowest: Full day
                                                    help_output_mod_degree_old = outputVector_BT1_heatGenerationCoefficientSpaceHeating [index_BT1 - 1] [highest_k_prices_array_full_day_1[current_price_inverval] [help_balance_time_slot]]
                                                    outputVector_BT1_heatGenerationCoefficientSpaceHeating [index_BT1 - 1] [highest_k_prices_array_full_day_1[current_price_inverval] [help_balance_time_slot]] = outputVector_BT1_heatGenerationCoefficientSpaceHeating [index_BT1 - 1] [highest_k_prices_array_full_day_1[current_price_inverval] [help_balance_time_slot]]   - percentage_shifted_loads[current_price_inverval]/100 -0.0
                                                    help_output_mod_degree_new = outputVector_BT1_heatGenerationCoefficientSpaceHeating [index_BT1 - 1] [highest_k_prices_array_full_day_1[current_price_inverval] [help_balance_time_slot]]
                                                    if help_output_mod_degree_new <0:
                                                        help_output_mod_degree_new = 0
                                                    changed_mod_degree = help_output_mod_degree_old - help_output_mod_degree_new
                                                    outputVector_BT1_heatGenerationCoefficientSpaceHeating [index_BT1 - 1] [lowest_k_prices_array_full_day_1[current_price_inverval] [help_balance_time_slot]] = outputVector_BT1_heatGenerationCoefficientSpaceHeating [index_BT1 - 1]  [lowest_k_prices_array_full_day_1[current_price_inverval] [help_balance_time_slot]]  +  changed_mod_degree


                                                    if outputVector_BT1_heatGenerationCoefficientSpaceHeating [index_BT1 - 1] [lowest_k_prices_array_full_day_1[current_price_inverval] [help_balance_time_slot]]  < SetUpScenarios.minimalModulationdDegree_HP /100:
                                                        outputVector_BT1_heatGenerationCoefficientSpaceHeating[index_BT1 - 1][lowest_k_prices_array_full_day_1[current_price_inverval] [help_balance_time_slot]] = SetUpScenarios.minimalModulationdDegree_HP /100
                                                    if outputVector_BT1_heatGenerationCoefficientSpaceHeating [index_BT1 - 1] [highest_k_prices_array_full_day_1[current_price_inverval] [help_balance_time_slot]]  < SetUpScenarios.minimalModulationdDegree_HP /100:
                                                        outputVector_BT1_heatGenerationCoefficientSpaceHeating[index_BT1 - 1][highest_k_prices_array_full_day_1[current_price_inverval] [help_balance_time_slot]] = 0
                                                    #Shift DHW heating power from highes price timeslot to lowest

                                                    if outputVector_BT1_heatGenerationCoefficientDHW [index_BT1 - 1]  [highest_k_prices_array_full_day_1[current_price_inverval] [help_balance_time_slot]] > 0.1:
                                                        help_output_mod_degree_old =  outputVector_BT1_heatGenerationCoefficientDHW [index_BT1 - 1]  [highest_k_prices_array_full_day_1[current_price_inverval] [help_balance_time_slot]]
                                                        outputVector_BT1_heatGenerationCoefficientDHW [index_BT1 - 1]  [highest_k_prices_array_full_day_1[current_price_inverval] [help_balance_time_slot]] = outputVector_BT1_heatGenerationCoefficientDHW [index_BT1 - 1] [highest_k_prices_array_full_day_1[current_price_inverval] [help_balance_time_slot]]   - percentage_shifted_loads[current_price_inverval]/100 -0.0
                                                        help_output_mod_degree_new =  outputVector_BT1_heatGenerationCoefficientDHW [index_BT1 - 1]  [highest_k_prices_array_full_day_1[current_price_inverval] [help_balance_time_slot]]
                                                        if help_output_mod_degree_new <0:
                                                            help_output_mod_degree_new = 0
                                                        changed_mod_degree = help_output_mod_degree_old - help_output_mod_degree_new
                                                        outputVector_BT1_heatGenerationCoefficientDHW [index_BT1 - 1] [lowest_k_prices_array_full_day_1[current_price_inverval] [help_balance_time_slot]] = outputVector_BT1_heatGenerationCoefficientDHW [index_BT1 - 1]  [lowest_k_prices_array_full_day_1[current_price_inverval] [help_balance_time_slot]]   + changed_mod_degree

                                                        if outputVector_BT1_heatGenerationCoefficientDHW [index_BT1 - 1] [lowest_k_prices_array_full_day_1[current_price_inverval] [help_balance_time_slot]] < SetUpScenarios.minimalModulationdDegree_HP /100:
                                                            outputVector_BT1_heatGenerationCoefficientDHW[index_BT1 - 1][lowest_k_prices_array_full_day_1[current_price_inverval] [help_balance_time_slot]]= SetUpScenarios.minimalModulationdDegree_HP /100
                                                        if outputVector_BT1_heatGenerationCoefficientDHW [index_BT1 - 1] [highest_k_prices_array_full_day_1[current_price_inverval] [help_balance_time_slot]] < SetUpScenarios.minimalModulationdDegree_HP /100:
                                                            outputVector_BT1_heatGenerationCoefficientDHW[index_BT1 - 1][highest_k_prices_array_full_day_1[current_price_inverval] [help_balance_time_slot]]= 0


                                                    # Shift charging power EV from highes price timeslot to lowest
                                                    help_output_EVpower_old = outputVector_BT1_chargingPowerEV [index_BT1 - 1] [highest_k_prices_array_full_day_1[current_price_inverval] [help_balance_time_slot]]
                                                    outputVector_BT1_chargingPowerEV [index_BT1 - 1] [highest_k_prices_array_full_day_1[current_price_inverval] [help_balance_time_slot]] = outputVector_BT1_chargingPowerEV [index_BT1 - 1] [highest_k_prices_array_full_day_1[current_price_inverval] [help_balance_time_slot]]   - (percentage_shifted_loads[current_price_inverval]*2/100) *SetUpScenarios.chargingPowerMaximal_EV
                                                    if outputVector_BT1_chargingPowerEV [index_BT1 - 1] [highest_k_prices_array_full_day_1[current_price_inverval] [help_balance_time_slot]]<0:
                                                        outputVector_BT1_chargingPowerEV [index_BT1 - 1] [highest_k_prices_array_full_day_1[current_price_inverval] [help_balance_time_slot]] = 0
                                                    help_output_EVpower_new = outputVector_BT1_chargingPowerEV [index_BT1 - 1] [highest_k_prices_array_full_day_1[current_price_inverval] [help_balance_time_slot]]
                                                    changed_EVpower = help_output_EVpower_old - help_output_EVpower_new

                                                    #Check if the EV is available at the desired timeslot for increasing the load
                                                    current_price_inverval_temp = current_price_inverval
                                                    while availability_pattern_EV_BT1 [index_BT1 - 1] [lowest_k_prices_array_full_day_1[current_price_inverval_temp] [help_balance_time_slot]] ==0 and current_price_inverval_temp<len(lowest_k_prices_array_full_day_1) - 1:
                                                        current_price_inverval_temp += 1
                                                    outputVector_BT1_chargingPowerEV [index_BT1 - 1] [lowest_k_prices_array_full_day_1[current_price_inverval_temp] [help_balance_time_slot]] = outputVector_BT1_chargingPowerEV [index_BT1 - 1]  [lowest_k_prices_array_full_day_1[current_price_inverval] [help_balance_time_slot]]   + changed_EVpower


                                                # Change the profiles of BT2
                                                for index_BT2 in indexOfBuildingsOverall_BT2:
                                                    help_balance_time_slot = helpIteration
                                                    #Shift Space heating power from highes price timeslot to lowest: Full day
                                                    help_output_mod_degree_old = outputVector_BT2_heatGenerationCoefficientSpaceHeating [index_BT2 - 1] [highest_k_prices_array_full_day_1[current_price_inverval] [help_balance_time_slot]]
                                                    outputVector_BT2_heatGenerationCoefficientSpaceHeating [index_BT2 - 1] [highest_k_prices_array_full_day_1[current_price_inverval] [help_balance_time_slot]] = outputVector_BT2_heatGenerationCoefficientSpaceHeating [index_BT2 - 1] [highest_k_prices_array_full_day_1[current_price_inverval] [help_balance_time_slot]]   - percentage_shifted_loads[current_price_inverval]/100 -0.0
                                                    help_output_mod_degree_new = outputVector_BT2_heatGenerationCoefficientSpaceHeating [index_BT2 - 1] [highest_k_prices_array_full_day_1[current_price_inverval] [help_balance_time_slot]]
                                                    if help_output_mod_degree_new <0:
                                                        help_output_mod_degree_new = 0
                                                    changed_mod_degree = help_output_mod_degree_old - help_output_mod_degree_new
                                                    outputVector_BT2_heatGenerationCoefficientSpaceHeating [index_BT2 - 1] [lowest_k_prices_array_full_day_1[current_price_inverval] [help_balance_time_slot]] = outputVector_BT2_heatGenerationCoefficientSpaceHeating [index_BT2 - 1]  [lowest_k_prices_array_full_day_1[current_price_inverval] [help_balance_time_slot]]  +  changed_mod_degree


                                                    if outputVector_BT2_heatGenerationCoefficientSpaceHeating [index_BT2 - 1] [lowest_k_prices_array_full_day_1[current_price_inverval] [help_balance_time_slot]]  < SetUpScenarios.minimalModulationdDegree_HP /100:
                                                        outputVector_BT2_heatGenerationCoefficientSpaceHeating[index_BT2 - 1][lowest_k_prices_array_full_day_1[current_price_inverval] [help_balance_time_slot]] = SetUpScenarios.minimalModulationdDegree_HP /100
                                                    if outputVector_BT2_heatGenerationCoefficientSpaceHeating [index_BT2 - 1] [highest_k_prices_array_full_day_1[current_price_inverval] [help_balance_time_slot]]  < SetUpScenarios.minimalModulationdDegree_HP /100:
                                                        outputVector_BT2_heatGenerationCoefficientSpaceHeating[index_BT2 - 1][highest_k_prices_array_full_day_1[current_price_inverval] [help_balance_time_slot]] = 0
                                                    #Shift DHW heating power from highes price timeslot to lowest

                                                    if outputVector_BT2_heatGenerationCoefficientDHW [index_BT2 - 1]  [highest_k_prices_array_full_day_1[current_price_inverval] [help_balance_time_slot]] > 0.1:
                                                        help_output_mod_degree_old =  outputVector_BT2_heatGenerationCoefficientDHW [index_BT2 - 1]  [highest_k_prices_array_full_day_1[current_price_inverval] [help_balance_time_slot]]
                                                        outputVector_BT2_heatGenerationCoefficientDHW [index_BT2 - 1]  [highest_k_prices_array_full_day_1[current_price_inverval] [help_balance_time_slot]] = outputVector_BT2_heatGenerationCoefficientDHW [index_BT2 - 1] [highest_k_prices_array_full_day_1[current_price_inverval] [help_balance_time_slot]]   - percentage_shifted_loads[current_price_inverval]/100 -0.0
                                                        help_output_mod_degree_new =  outputVector_BT2_heatGenerationCoefficientDHW [index_BT2 - 1]  [highest_k_prices_array_full_day_1[current_price_inverval] [help_balance_time_slot]]
                                                        if help_output_mod_degree_new <0:
                                                            help_output_mod_degree_new = 0
                                                        changed_mod_degree = help_output_mod_degree_old - help_output_mod_degree_new
                                                        outputVector_BT2_heatGenerationCoefficientDHW [index_BT2 - 1] [lowest_k_prices_array_full_day_1[current_price_inverval] [help_balance_time_slot]] = outputVector_BT2_heatGenerationCoefficientDHW [index_BT2 - 1]  [lowest_k_prices_array_full_day_1[current_price_inverval] [help_balance_time_slot]]   + changed_mod_degree

                                                        if outputVector_BT2_heatGenerationCoefficientDHW [index_BT2 - 1] [lowest_k_prices_array_full_day_1[current_price_inverval] [help_balance_time_slot]] < SetUpScenarios.minimalModulationdDegree_HP /100:
                                                            outputVector_BT2_heatGenerationCoefficientDHW[index_BT2 - 1][lowest_k_prices_array_full_day_1[current_price_inverval] [help_balance_time_slot]]= SetUpScenarios.minimalModulationdDegree_HP /100
                                                        if outputVector_BT2_heatGenerationCoefficientDHW [index_BT2 - 1] [highest_k_prices_array_full_day_1[current_price_inverval] [help_balance_time_slot]] < SetUpScenarios.minimalModulationdDegree_HP /100:
                                                            outputVector_BT2_heatGenerationCoefficientDHW[index_BT2 - 1][highest_k_prices_array_full_day_1[current_price_inverval] [help_balance_time_slot]]= 0

                                                # Change the profiles of BT4
                                                for index_BT4 in indexOfBuildingsOverall_BT4:

                                                    help_balance_time_slot = random.randint(0, k-1)

                                                    #Shift Space heating power from highes price timeslot to lowest: Full day
                                                    help_temp_highest_price = highest_k_prices_array_full_day_1[current_price_inverval] [help_balance_time_slot]
                                                    help_temp_lowest_price = lowest_k_prices_array_full_day_1[current_price_inverval] [help_balance_time_slot]

                                                    help_output_mod_degree_old = outputVector_BT4_heatGenerationCoefficientSpaceHeating [index_BT4 - 1] [highest_k_prices_array_full_day_1[current_price_inverval] [help_balance_time_slot]]
                                                    outputVector_BT4_heatGenerationCoefficientSpaceHeating [index_BT4 - 1] [highest_k_prices_array_full_day_1[current_price_inverval] [help_balance_time_slot]] = outputVector_BT4_heatGenerationCoefficientSpaceHeating [index_BT4 - 1] [highest_k_prices_array_full_day_1[current_price_inverval] [help_balance_time_slot]]   - percentage_shifted_loads[current_price_inverval]/100 -0.0
                                                    help_output_mod_degree_new = outputVector_BT4_heatGenerationCoefficientSpaceHeating [index_BT4 - 1] [highest_k_prices_array_full_day_1[current_price_inverval] [help_balance_time_slot]]

                                                    if outputVector_BT4_heatGenerationCoefficientSpaceHeating [index_BT4 - 1] [lowest_k_prices_array_full_day_1[current_price_inverval] [help_balance_time_slot]]  < SetUpScenarios.minimalModulationdDegree_HP /100:
                                                        outputVector_BT4_heatGenerationCoefficientSpaceHeating[index_BT4 - 1][lowest_k_prices_array_full_day_1[current_price_inverval] [help_balance_time_slot]] = SetUpScenarios.minimalModulationdDegree_HP /100
                                                    if outputVector_BT4_heatGenerationCoefficientSpaceHeating [index_BT4 - 1] [highest_k_prices_array_full_day_1[current_price_inverval] [help_balance_time_slot]]  < SetUpScenarios.minimalModulationdDegree_HP /100:
                                                        outputVector_BT4_heatGenerationCoefficientSpaceHeating[index_BT4 - 1][highest_k_prices_array_full_day_1[current_price_inverval] [help_balance_time_slot]] = 0

                                                    changed_mod_degree = help_output_mod_degree_old - help_output_mod_degree_new
                                                    outputVector_BT4_heatGenerationCoefficientSpaceHeating [index_BT4 - 1] [lowest_k_prices_array_full_day_1[current_price_inverval] [help_balance_time_slot]] = outputVector_BT4_heatGenerationCoefficientSpaceHeating [index_BT4 - 1]  [lowest_k_prices_array_full_day_1[current_price_inverval] [help_balance_time_slot]]  +  changed_mod_degree


                                        #Shift profiles cosindering the half day partition
                                        if param_1_number_of_partitions_for_shifting == 2:
                                            for current_partition_of_the_day in range (0, 2):
                                                for current_price_inverval in range(0, param_2_number_of_intervales_within_partition):
                                                    # Change the profiles of BT1
                                                    for index_BT1 in indexOfBuildingsOverall_BT1:
                                                        help_balance_time_slot = random.randint(0, k-1)
                                                        #Shift Space heating power from highes price timeslot to lowest: Full day
                                                        help_output_mod_degree_old =outputVector_BT1_heatGenerationCoefficientSpaceHeating [index_BT1 - 1] [highest_k_prices_half_day_combined_array [current_partition_of_the_day][current_price_inverval] [help_balance_time_slot]]
                                                        outputVector_BT1_heatGenerationCoefficientSpaceHeating [index_BT1 - 1] [highest_k_prices_half_day_combined_array [current_partition_of_the_day][current_price_inverval] [help_balance_time_slot]] = outputVector_BT1_heatGenerationCoefficientSpaceHeating [index_BT1 - 1] [highest_k_prices_array_full_day_1[current_price_inverval] [help_balance_time_slot]]   - percentage_shifted_loads[current_price_inverval]/100 -0.0
                                                        help_output_mod_degree_new = outputVector_BT1_heatGenerationCoefficientSpaceHeating [index_BT1 - 1] [highest_k_prices_half_day_combined_array [current_partition_of_the_day][current_price_inverval] [help_balance_time_slot]]
                                                        changed_mod_degree = help_output_mod_degree_old - help_output_mod_degree_new
                                                        outputVector_BT1_heatGenerationCoefficientSpaceHeating [index_BT1 - 1] [lowest_k_prices_half_day_combined_array [current_partition_of_the_day][current_price_inverval] [help_balance_time_slot]] = outputVector_BT1_heatGenerationCoefficientSpaceHeating [index_BT1 - 1]  [lowest_k_prices_array_full_day_1[current_price_inverval] [help_balance_time_slot]]   + changed_mod_degree

                                                        if outputVector_BT1_heatGenerationCoefficientSpaceHeating [index_BT1 - 1] [lowest_k_prices_half_day_combined_array [current_partition_of_the_day][current_price_inverval] [help_balance_time_slot]]  < SetUpScenarios.minimalModulationdDegree_HP /100:
                                                            outputVector_BT1_heatGenerationCoefficientSpaceHeating[index_BT1 - 1][lowest_k_prices_half_day_combined_array [current_partition_of_the_day][current_price_inverval] [help_balance_time_slot]] = SetUpScenarios.minimalModulationdDegree_HP /100
                                                        if outputVector_BT1_heatGenerationCoefficientSpaceHeating [index_BT1 - 1] [highest_k_prices_half_day_combined_array [current_partition_of_the_day][current_price_inverval] [help_balance_time_slot]] < SetUpScenarios.minimalModulationdDegree_HP/100:
                                                            outputVector_BT1_heatGenerationCoefficientSpaceHeating [index_BT1 - 1] [highest_k_prices_half_day_combined_array [current_partition_of_the_day][current_price_inverval] [help_balance_time_slot]] =0

                                                        #Shift DHW heating power from highes price timeslot to lowest
                                                        random_number_dhw_heating = random.random()
                                                        if random_number_dhw_heating > threshold_random_number_dhw_heating / 100:
                                                            help_output_mod_degree_old = outputVector_BT1_heatGenerationCoefficientDHW [index_BT1 - 1]  [highest_k_prices_half_day_combined_array [current_partition_of_the_day][current_price_inverval] [help_balance_time_slot]]
                                                            outputVector_BT1_heatGenerationCoefficientDHW [index_BT1 - 1]  [highest_k_prices_half_day_combined_array [current_partition_of_the_day][current_price_inverval] [help_balance_time_slot]] = outputVector_BT1_heatGenerationCoefficientDHW [index_BT1 - 1] [highest_k_prices_array_full_day_1[current_price_inverval] [help_balance_time_slot]]   - percentage_shifted_loads[current_price_inverval]/100 -0.0
                                                            help_output_mod_degree_new  = outputVector_BT1_heatGenerationCoefficientDHW [index_BT1 - 1]  [highest_k_prices_half_day_combined_array [current_partition_of_the_day][current_price_inverval] [help_balance_time_slot]]
                                                            changed_mod_degree = help_output_mod_degree_old - help_output_mod_degree_new
                                                            outputVector_BT1_heatGenerationCoefficientDHW [index_BT1 - 1] [lowest_k_prices_half_day_combined_array [current_partition_of_the_day][current_price_inverval] [help_balance_time_slot]] = outputVector_BT1_heatGenerationCoefficientDHW [index_BT1 - 1]  [lowest_k_prices_array_full_day_1[current_price_inverval] [help_balance_time_slot]]   + changed_mod_degree

                                                            if outputVector_BT1_heatGenerationCoefficientDHW [index_BT1 - 1] [lowest_k_prices_half_day_combined_array [current_partition_of_the_day][current_price_inverval] [help_balance_time_slot]] < SetUpScenarios.minimalModulationdDegree_HP /100:
                                                                outputVector_BT1_heatGenerationCoefficientDHW[index_BT1 - 1][lowest_k_prices_half_day_combined_array [current_partition_of_the_day][current_price_inverval] [help_balance_time_slot]]= SetUpScenarios.minimalModulationdDegree_HP /100
                                                            if outputVector_BT1_heatGenerationCoefficientDHW [index_BT1 - 1] [highest_k_prices_half_day_combined_array [current_partition_of_the_day][current_price_inverval] [help_balance_time_slot]] < SetUpScenarios.minimalModulationdDegree_HP/100:
                                                                outputVector_BT1_heatGenerationCoefficientDHW [index_BT1 - 1] [highest_k_prices_half_day_combined_array [current_partition_of_the_day][current_price_inverval] [help_balance_time_slot]] =0


                                                        # Shift charging power EV from highes price timeslot to lowest
                                                        help_output_EVpower_old = outputVector_BT1_chargingPowerEV [index_BT1 - 1] [highest_k_prices_half_day_combined_array [current_partition_of_the_day][current_price_inverval] [help_balance_time_slot]]
                                                        outputVector_BT1_chargingPowerEV [index_BT1 - 1] [highest_k_prices_half_day_combined_array [current_partition_of_the_day][current_price_inverval] [help_balance_time_slot]] = outputVector_BT1_chargingPowerEV [index_BT1 - 1] [highest_k_prices_array_full_day_1[current_price_inverval] [help_balance_time_slot]]   * ((100-percentage_shifted_loads[current_price_inverval])/100)
                                                        help_output_EVpower_new = outputVector_BT1_chargingPowerEV [index_BT1 - 1] [highest_k_prices_half_day_combined_array [current_partition_of_the_day][current_price_inverval] [help_balance_time_slot]]
                                                        changed_EVpower = help_output_EVpower_old - help_output_EVpower_new
                                                        outputVector_BT1_chargingPowerEV [index_BT1 - 1] [lowest_k_prices_half_day_combined_array [current_partition_of_the_day][current_price_inverval] [help_balance_time_slot]] = outputVector_BT1_chargingPowerEV [index_BT1 - 1]  [lowest_k_prices_array_full_day_1[current_price_inverval] [help_balance_time_slot]]   + changed_EVpower

                                                    # Change the profiles of BT2
                                                    for index_BT2 in indexOfBuildingsOverall_BT2:
                                                        help_balance_time_slot = random.randint(0, k-1)
                                                        #Shift Space heating power from highes price timeslot to lowest: Full day
                                                        help_output_mod_degree_old =outputVector_BT2_heatGenerationCoefficientSpaceHeating [index_BT2 - 1] [highest_k_prices_half_day_combined_array [current_partition_of_the_day][current_price_inverval] [help_balance_time_slot]]
                                                        outputVector_BT2_heatGenerationCoefficientSpaceHeating [index_BT2 - 1] [highest_k_prices_half_day_combined_array [current_partition_of_the_day][current_price_inverval] [help_balance_time_slot]] = outputVector_BT2_heatGenerationCoefficientSpaceHeating [index_BT2 - 1] [highest_k_prices_array_full_day_1[current_price_inverval] [help_balance_time_slot]]   - percentage_shifted_loads[current_price_inverval]/100 -0.0
                                                        help_output_mod_degree_new = outputVector_BT2_heatGenerationCoefficientSpaceHeating [index_BT2 - 1] [highest_k_prices_half_day_combined_array [current_partition_of_the_day][current_price_inverval] [help_balance_time_slot]]
                                                        changed_mod_degree = help_output_mod_degree_old - help_output_mod_degree_new
                                                        outputVector_BT2_heatGenerationCoefficientSpaceHeating [index_BT2 - 1] [lowest_k_prices_half_day_combined_array [current_partition_of_the_day][current_price_inverval] [help_balance_time_slot]] = outputVector_BT2_heatGenerationCoefficientSpaceHeating [index_BT2 - 1]  [lowest_k_prices_array_full_day_1[current_price_inverval] [help_balance_time_slot]]   + changed_mod_degree

                                                        if outputVector_BT2_heatGenerationCoefficientSpaceHeating [index_BT2 - 1] [lowest_k_prices_half_day_combined_array [current_partition_of_the_day][current_price_inverval] [help_balance_time_slot]]  < SetUpScenarios.minimalModulationdDegree_HP /100:
                                                            outputVector_BT2_heatGenerationCoefficientSpaceHeating[index_BT2 - 1][lowest_k_prices_half_day_combined_array [current_partition_of_the_day][current_price_inverval] [help_balance_time_slot]] = SetUpScenarios.minimalModulationdDegree_HP /100
                                                        if outputVector_BT2_heatGenerationCoefficientSpaceHeating [index_BT2 - 1] [highest_k_prices_half_day_combined_array [current_partition_of_the_day][current_price_inverval] [help_balance_time_slot]] < SetUpScenarios.minimalModulationdDegree_HP/100:
                                                            outputVector_BT2_heatGenerationCoefficientSpaceHeating [index_BT2 - 1] [highest_k_prices_half_day_combined_array [current_partition_of_the_day][current_price_inverval] [help_balance_time_slot]] =0


                                                        #Shift DHW heating power from highes price timeslot to lowest
                                                        random_number_dhw_heating = random.random()
                                                        if random_number_dhw_heating > threshold_random_number_dhw_heating / 100:
                                                            help_output_mod_degree_old = outputVector_BT2_heatGenerationCoefficientDHW [index_BT2 - 1]  [highest_k_prices_half_day_combined_array [current_partition_of_the_day][current_price_inverval] [help_balance_time_slot]]
                                                            outputVector_BT2_heatGenerationCoefficientDHW [index_BT2 - 1]  [highest_k_prices_half_day_combined_array [current_partition_of_the_day][current_price_inverval] [help_balance_time_slot]] = outputVector_BT2_heatGenerationCoefficientDHW [index_BT2 - 1] [highest_k_prices_array_full_day_1[current_price_inverval] [help_balance_time_slot]]   - percentage_shifted_loads[current_price_inverval]/100 -0.0
                                                            help_output_mod_degree_new  = outputVector_BT2_heatGenerationCoefficientDHW [index_BT2 - 1]  [highest_k_prices_half_day_combined_array [current_partition_of_the_day][current_price_inverval] [help_balance_time_slot]]
                                                            changed_mod_degree = help_output_mod_degree_old - help_output_mod_degree_new
                                                            outputVector_BT2_heatGenerationCoefficientDHW [index_BT2 - 1] [lowest_k_prices_half_day_combined_array [current_partition_of_the_day][current_price_inverval] [help_balance_time_slot]] = outputVector_BT2_heatGenerationCoefficientDHW [index_BT2 - 1]  [lowest_k_prices_array_full_day_1[current_price_inverval] [help_balance_time_slot]]   + changed_mod_degree


                                                            if outputVector_BT2_heatGenerationCoefficientDHW [index_BT2 - 1] [lowest_k_prices_half_day_combined_array [current_partition_of_the_day][current_price_inverval] [help_balance_time_slot]] < SetUpScenarios.minimalModulationdDegree_HP /100:
                                                                outputVector_BT2_heatGenerationCoefficientDHW[index_BT2 - 1][lowest_k_prices_half_day_combined_array [current_partition_of_the_day][current_price_inverval] [help_balance_time_slot]]= SetUpScenarios.minimalModulationdDegree_HP /100
                                                            if outputVector_BT2_heatGenerationCoefficientDHW [index_BT2 - 1] [highest_k_prices_half_day_combined_array [current_partition_of_the_day][current_price_inverval] [help_balance_time_slot]] < SetUpScenarios.minimalModulationdDegree_HP/100:
                                                                outputVector_BT2_heatGenerationCoefficientDHW [index_BT2 - 1] [highest_k_prices_half_day_combined_array [current_partition_of_the_day][current_price_inverval] [help_balance_time_slot]] =0

                                                    # Change the profiles of BT4
                                                    for index_BT4 in indexOfBuildingsOverall_BT4:
                                                        help_balance_time_slot = random.randint(0, k-1)
                                                        #Shift Space heating power from highes price timeslot to lowest: Full day
                                                        help_temp_highest_price = highest_k_prices_half_day_combined_array [current_partition_of_the_day][current_price_inverval] [help_balance_time_slot]
                                                        help_temp_lowest_price = lowest_k_prices_half_day_combined_array [current_partition_of_the_day][current_price_inverval] [help_balance_time_slot]

                                                        help_output_mod_degree_old =outputVector_BT4_heatGenerationCoefficientSpaceHeating [index_BT4 - 1] [highest_k_prices_half_day_combined_array [current_partition_of_the_day][current_price_inverval] [help_balance_time_slot]]
                                                        outputVector_BT4_heatGenerationCoefficientSpaceHeating [index_BT4 - 1] [highest_k_prices_half_day_combined_array [current_partition_of_the_day][current_price_inverval] [help_balance_time_slot]] = outputVector_BT4_heatGenerationCoefficientSpaceHeating [index_BT4 - 1] [highest_k_prices_array_full_day_1[current_price_inverval] [help_balance_time_slot]]   - percentage_shifted_loads[current_price_inverval]/100 -0.0
                                                        help_output_mod_degree_new = outputVector_BT4_heatGenerationCoefficientSpaceHeating [index_BT4 - 1] [highest_k_prices_half_day_combined_array [current_partition_of_the_day][current_price_inverval] [help_balance_time_slot]]
                                                        changed_mod_degree = help_output_mod_degree_old - help_output_mod_degree_new
                                                        outputVector_BT4_heatGenerationCoefficientSpaceHeating [index_BT4 - 1] [lowest_k_prices_half_day_combined_array [current_partition_of_the_day][current_price_inverval] [help_balance_time_slot]] = outputVector_BT4_heatGenerationCoefficientSpaceHeating [index_BT4 - 1]  [lowest_k_prices_array_full_day_1[current_price_inverval] [help_balance_time_slot]]   + changed_mod_degree

                                                        if outputVector_BT4_heatGenerationCoefficientSpaceHeating [index_BT4 - 1] [lowest_k_prices_half_day_combined_array [current_partition_of_the_day][current_price_inverval] [help_balance_time_slot]]  < SetUpScenarios.minimalModulationdDegree_HP /100:
                                                            outputVector_BT4_heatGenerationCoefficientSpaceHeating[index_BT4 - 1][lowest_k_prices_half_day_combined_array [current_partition_of_the_day][current_price_inverval] [help_balance_time_slot]] = SetUpScenarios.minimalModulationdDegree_HP /100
                                                        if outputVector_BT4_heatGenerationCoefficientSpaceHeating [index_BT4 - 1] [highest_k_prices_half_day_combined_array [current_partition_of_the_day][current_price_inverval] [help_balance_time_slot]] < SetUpScenarios.minimalModulationdDegree_HP/100:
                                                            outputVector_BT4_heatGenerationCoefficientSpaceHeating [index_BT4 - 1] [highest_k_prices_half_day_combined_array [current_partition_of_the_day][current_price_inverval] [help_balance_time_slot]] =0

                                        #Shift profiles cosindering the third day partition
                                        if param_1_number_of_partitions_for_shifting == 3:
                                            for current_partition_of_the_day in range (0, 3):
                                                for current_price_inverval in range(0, param_2_number_of_intervales_within_partition):
                                                    # Change the profiles of BT1
                                                    for index_BT1 in indexOfBuildingsOverall_BT1:
                                                        help_balance_time_slot = random.randint(0, k-1)
                                                        #Shift Space heating power from highes price timeslot to lowest: Full day
                                                        help_output_mod_degree_old = outputVector_BT1_heatGenerationCoefficientSpaceHeating [index_BT1 - 1] [highest_k_prices_onethird_day_combined_array [current_partition_of_the_day][current_price_inverval] [help_balance_time_slot]]
                                                        outputVector_BT1_heatGenerationCoefficientSpaceHeating [index_BT1 - 1] [highest_k_prices_onethird_day_combined_array [current_partition_of_the_day][current_price_inverval] [help_balance_time_slot]] = outputVector_BT1_heatGenerationCoefficientSpaceHeating [index_BT1 - 1] [highest_k_prices_array_full_day_1[current_price_inverval] [help_balance_time_slot]]   - percentage_shifted_loads[current_price_inverval]/100 -0.0
                                                        help_output_mod_degree_new = outputVector_BT1_heatGenerationCoefficientSpaceHeating [index_BT1 - 1] [highest_k_prices_onethird_day_combined_array [current_partition_of_the_day][current_price_inverval] [help_balance_time_slot]]
                                                        changed_mod_degree = help_output_mod_degree_old - help_output_mod_degree_new
                                                        outputVector_BT1_heatGenerationCoefficientSpaceHeating [index_BT1 - 1] [lowest_k_prices_onethird_day_combined_array [current_partition_of_the_day][current_price_inverval] [help_balance_time_slot]] = outputVector_BT1_heatGenerationCoefficientSpaceHeating [index_BT1 - 1]  [lowest_k_prices_array_full_day_1[current_price_inverval] [help_balance_time_slot]]   + changed_mod_degree

                                                        if outputVector_BT1_heatGenerationCoefficientSpaceHeating [index_BT1 - 1] [lowest_k_prices_onethird_day_combined_array [current_partition_of_the_day][current_price_inverval] [help_balance_time_slot]]  < SetUpScenarios.minimalModulationdDegree_HP /100:
                                                            outputVector_BT1_heatGenerationCoefficientSpaceHeating[index_BT1 - 1][lowest_k_prices_onethird_day_combined_array [current_partition_of_the_day][current_price_inverval] [help_balance_time_slot]] = SetUpScenarios.minimalModulationdDegree_HP /100
                                                        if outputVector_BT1_heatGenerationCoefficientSpaceHeating [index_BT1 - 1] [highest_k_prices_onethird_day_combined_array [current_partition_of_the_day][current_price_inverval] [help_balance_time_slot]]  < SetUpScenarios.minimalModulationdDegree_HP /100:
                                                            outputVector_BT1_heatGenerationCoefficientSpaceHeating[index_BT1 - 1][highest_k_prices_onethird_day_combined_array [current_partition_of_the_day][current_price_inverval] [help_balance_time_slot]] = 0

                                                        #Shift DHW heating power from highes price timeslot to lowest
                                                        random_number_dhw_heating = random.random()
                                                        if random_number_dhw_heating > threshold_random_number_dhw_heating / 100:
                                                            help_output_mod_degree_old = outputVector_BT1_heatGenerationCoefficientDHW [index_BT1 - 1]  [highest_k_prices_onethird_day_combined_array [current_partition_of_the_day][current_price_inverval] [help_balance_time_slot]]
                                                            outputVector_BT1_heatGenerationCoefficientDHW [index_BT1 - 1]  [highest_k_prices_onethird_day_combined_array [current_partition_of_the_day][current_price_inverval] [help_balance_time_slot]] = outputVector_BT1_heatGenerationCoefficientDHW [index_BT1 - 1] [highest_k_prices_array_full_day_1[current_price_inverval] [help_balance_time_slot]]  - percentage_shifted_loads[current_price_inverval]/100 -0.0
                                                            help_output_mod_degree_new = outputVector_BT1_heatGenerationCoefficientDHW [index_BT1 - 1]  [highest_k_prices_onethird_day_combined_array [current_partition_of_the_day][current_price_inverval] [help_balance_time_slot]]
                                                            changed_mod_degree = help_output_mod_degree_old - help_output_mod_degree_new
                                                            outputVector_BT1_heatGenerationCoefficientDHW [index_BT1 - 1] [lowest_k_prices_onethird_day_combined_array [current_partition_of_the_day][current_price_inverval] [help_balance_time_slot]] = outputVector_BT1_heatGenerationCoefficientDHW [index_BT1 - 1]  [lowest_k_prices_array_full_day_1[current_price_inverval] [help_balance_time_slot]]   + changed_mod_degree

                                                            if outputVector_BT1_heatGenerationCoefficientDHW [index_BT1 - 1] [lowest_k_prices_onethird_day_combined_array [current_partition_of_the_day][current_price_inverval] [help_balance_time_slot]] < SetUpScenarios.minimalModulationdDegree_HP /100:
                                                                outputVector_BT1_heatGenerationCoefficientDHW[index_BT1 - 1][lowest_k_prices_onethird_day_combined_array [current_partition_of_the_day][current_price_inverval] [help_balance_time_slot]]= SetUpScenarios.minimalModulationdDegree_HP /100
                                                            if outputVector_BT1_heatGenerationCoefficientDHW [index_BT1 - 1] [highest_k_prices_onethird_day_combined_array [current_partition_of_the_day][current_price_inverval] [help_balance_time_slot]] < SetUpScenarios.minimalModulationdDegree_HP /100:
                                                                outputVector_BT1_heatGenerationCoefficientDHW[index_BT1 - 1][highest_k_prices_onethird_day_combined_array [current_partition_of_the_day][current_price_inverval] [help_balance_time_slot]]= SetUpScenarios.minimalModulationdDegree_HP /100


                                                        # Shift charging power EV from highes price timeslot to lowest
                                                        help_output_EVpower_old = outputVector_BT1_chargingPowerEV [index_BT1 - 1] [highest_k_prices_onethird_day_combined_array [current_partition_of_the_day][current_price_inverval] [help_balance_time_slot]]
                                                        outputVector_BT1_chargingPowerEV [index_BT1 - 1] [highest_k_prices_onethird_day_combined_array [current_partition_of_the_day][current_price_inverval] [help_balance_time_slot]] = outputVector_BT1_chargingPowerEV [index_BT1 - 1] [highest_k_prices_array_full_day_1[current_price_inverval] [help_balance_time_slot]]   * ((100-percentage_shifted_loads[current_price_inverval])/100)
                                                        help_output_EVpower_new = outputVector_BT1_chargingPowerEV [index_BT1 - 1] [highest_k_prices_onethird_day_combined_array [current_partition_of_the_day][current_price_inverval] [help_balance_time_slot]]
                                                        changed_EVpower = help_output_EVpower_old - help_output_mod_degree_new
                                                        outputVector_BT1_chargingPowerEV [index_BT1 - 1] [lowest_k_prices_onethird_day_combined_array [current_partition_of_the_day][current_price_inverval] [help_balance_time_slot]] = outputVector_BT1_chargingPowerEV [index_BT1 - 1]  [lowest_k_prices_array_full_day_1[current_price_inverval] [help_balance_time_slot]]   + changed_EVpower
                                                        help_balance_time_slot += 1
                                                        if help_balance_time_slot >=k:
                                                            help_balance_time_slot = 0

                                                    # Change the profiles of BT2
                                                    for index_BT2 in indexOfBuildingsOverall_BT2:
                                                        help_balance_time_slot = random.randint(0, k-1)
                                                        #Shift Space heating power from highes price timeslot to lowest: Full day
                                                        help_output_mod_degree_old = outputVector_BT2_heatGenerationCoefficientSpaceHeating [index_BT2 - 1] [highest_k_prices_onethird_day_combined_array [current_partition_of_the_day][current_price_inverval] [help_balance_time_slot]]
                                                        outputVector_BT2_heatGenerationCoefficientSpaceHeating [index_BT2 - 1] [highest_k_prices_onethird_day_combined_array [current_partition_of_the_day][current_price_inverval] [help_balance_time_slot]] = outputVector_BT2_heatGenerationCoefficientSpaceHeating [index_BT2 - 1] [highest_k_prices_array_full_day_1[current_price_inverval] [help_balance_time_slot]]   - percentage_shifted_loads[current_price_inverval]/100 -0.0
                                                        help_output_mod_degree_new = outputVector_BT2_heatGenerationCoefficientSpaceHeating [index_BT2 - 1] [highest_k_prices_onethird_day_combined_array [current_partition_of_the_day][current_price_inverval] [help_balance_time_slot]]
                                                        changed_mod_degree = help_output_mod_degree_old - help_output_mod_degree_new
                                                        outputVector_BT2_heatGenerationCoefficientSpaceHeating [index_BT2 - 1] [lowest_k_prices_onethird_day_combined_array [current_partition_of_the_day][current_price_inverval] [help_balance_time_slot]] = outputVector_BT2_heatGenerationCoefficientSpaceHeating [index_BT2 - 1]  [lowest_k_prices_array_full_day_1[current_price_inverval] [help_balance_time_slot]]   + changed_mod_degree


                                                        if outputVector_BT2_heatGenerationCoefficientSpaceHeating [index_BT2 - 1] [lowest_k_prices_onethird_day_combined_array [current_partition_of_the_day][current_price_inverval] [help_balance_time_slot]]  < SetUpScenarios.minimalModulationdDegree_HP /100:
                                                            outputVector_BT2_heatGenerationCoefficientSpaceHeating[index_BT2 - 1][lowest_k_prices_onethird_day_combined_array [current_partition_of_the_day][current_price_inverval] [help_balance_time_slot]] = SetUpScenarios.minimalModulationdDegree_HP /100
                                                        if outputVector_BT2_heatGenerationCoefficientSpaceHeating [index_BT2 - 1] [highest_k_prices_onethird_day_combined_array [current_partition_of_the_day][current_price_inverval] [help_balance_time_slot]]  < SetUpScenarios.minimalModulationdDegree_HP /100:
                                                            outputVector_BT2_heatGenerationCoefficientSpaceHeating[index_BT2 - 1][highest_k_prices_onethird_day_combined_array [current_partition_of_the_day][current_price_inverval] [help_balance_time_slot]] = SetUpScenarios.minimalModulationdDegree_HP /100

                                                        #Shift DHW heating power from highes price timeslot to lowest
                                                        random_number_dhw_heating = random.random()
                                                        if random_number_dhw_heating > threshold_random_number_dhw_heating / 100:
                                                            help_output_mod_degree_old = outputVector_BT2_heatGenerationCoefficientDHW [index_BT2 - 1]  [highest_k_prices_onethird_day_combined_array [current_partition_of_the_day][current_price_inverval] [help_balance_time_slot]]
                                                            outputVector_BT2_heatGenerationCoefficientDHW [index_BT2 - 1]  [highest_k_prices_onethird_day_combined_array [current_partition_of_the_day][current_price_inverval] [help_balance_time_slot]] = outputVector_BT2_heatGenerationCoefficientDHW [index_BT2 - 1] [highest_k_prices_array_full_day_1[current_price_inverval] [help_balance_time_slot]]  - percentage_shifted_loads[current_price_inverval]/100 -0.0
                                                            help_output_mod_degree_new = outputVector_BT2_heatGenerationCoefficientDHW [index_BT2 - 1]  [highest_k_prices_onethird_day_combined_array [current_partition_of_the_day][current_price_inverval] [help_balance_time_slot]]
                                                            changed_mod_degree = help_output_mod_degree_old - help_output_mod_degree_new
                                                            outputVector_BT2_heatGenerationCoefficientDHW [index_BT2 - 1] [lowest_k_prices_onethird_day_combined_array [current_partition_of_the_day][current_price_inverval] [help_balance_time_slot]] = outputVector_BT2_heatGenerationCoefficientDHW [index_BT2 - 1]  [lowest_k_prices_array_full_day_1[current_price_inverval] [help_balance_time_slot]]   + changed_mod_degree

                                                            if outputVector_BT2_heatGenerationCoefficientDHW [index_BT2 - 1] [lowest_k_prices_onethird_day_combined_array [current_partition_of_the_day][current_price_inverval] [help_balance_time_slot]] < SetUpScenarios.minimalModulationdDegree_HP /100:
                                                                outputVector_BT2_heatGenerationCoefficientDHW[index_BT2 - 1][lowest_k_prices_onethird_day_combined_array [current_partition_of_the_day][current_price_inverval] [help_balance_time_slot]]= SetUpScenarios.minimalModulationdDegree_HP /100
                                                            if outputVector_BT2_heatGenerationCoefficientDHW [index_BT2 - 1] [highest_k_prices_onethird_day_combined_array [current_partition_of_the_day][current_price_inverval] [help_balance_time_slot]] < SetUpScenarios.minimalModulationdDegree_HP /100:
                                                                outputVector_BT2_heatGenerationCoefficientDHW[index_BT2 - 1][highest_k_prices_onethird_day_combined_array [current_partition_of_the_day][current_price_inverval] [help_balance_time_slot]]= SetUpScenarios.minimalModulationdDegree_HP /100


                                                    # Change the profiles of BT4
                                                    for index_BT4 in indexOfBuildingsOverall_BT4:
                                                        help_balance_time_slot = random.randint(0, k-1)
                                                        #Shift Space heating power from highes price timeslot to lowest: Full day
                                                        help_output_mod_degree_old = outputVector_BT4_heatGenerationCoefficientSpaceHeating [index_BT4 - 1]  [highest_k_prices_onethird_day_combined_array [current_partition_of_the_day][current_price_inverval] [help_balance_time_slot]]
                                                        outputVector_BT4_heatGenerationCoefficientSpaceHeating [index_BT4 - 1]  [highest_k_prices_onethird_day_combined_array [current_partition_of_the_day][current_price_inverval] [help_balance_time_slot]] = outputVector_BT4_heatGenerationCoefficientSpaceHeating [index_BT4 - 1] [highest_k_prices_array_full_day_1[current_price_inverval] [help_balance_time_slot]]   - percentage_shifted_loads[current_price_inverval]/100 -0.0
                                                        help_output_mod_degree_new = outputVector_BT4_heatGenerationCoefficientSpaceHeating [index_BT4 - 1]  [highest_k_prices_onethird_day_combined_array [current_partition_of_the_day][current_price_inverval] [help_balance_time_slot]]
                                                        changed_mod_degree = help_output_mod_degree_old - help_output_mod_degree_new
                                                        outputVector_BT4_heatGenerationCoefficientSpaceHeating [index_BT4 - 1] [lowest_k_prices_onethird_day_combined_array [current_partition_of_the_day][current_price_inverval] [help_balance_time_slot]] = outputVector_BT4_heatGenerationCoefficientSpaceHeating [index_BT4 - 1]  [lowest_k_prices_array_full_day_1[current_price_inverval] [help_balance_time_slot]]   + changed_mod_degree

                                                        if outputVector_BT4_heatGenerationCoefficientSpaceHeating [index_BT4 - 1] [lowest_k_prices_onethird_day_combined_array [current_partition_of_the_day][current_price_inverval] [help_balance_time_slot]]  < SetUpScenarios.minimalModulationdDegree_HP /100:
                                                            outputVector_BT4_heatGenerationCoefficientSpaceHeating[index_BT4 - 1][lowest_k_prices_onethird_day_combined_array [current_partition_of_the_day][current_price_inverval] [help_balance_time_slot]] = SetUpScenarios.minimalModulationdDegree_HP /100
                                                        if outputVector_BT4_heatGenerationCoefficientSpaceHeating [index_BT4 - 1] [highest_k_prices_onethird_day_combined_array [current_partition_of_the_day][current_price_inverval] [help_balance_time_slot]]  < SetUpScenarios.minimalModulationdDegree_HP /100:
                                                            outputVector_BT4_heatGenerationCoefficientSpaceHeating[index_BT4 - 1][highest_k_prices_onethird_day_combined_array [current_partition_of_the_day][current_price_inverval] [help_balance_time_slot]] = SetUpScenarios.minimalModulationdDegree_HP /100

                                if withRL:
                                    # Adust the arrays
                                    adjusted_percentage_array_loads_highest_prices_temp = percentage_array_loads_per_timeslot_highest_prices.reshape(12, 2)
                                    adjusted_percentage_array_loads_highest_prices = adjusted_percentage_array_loads_highest_prices_temp.sum(axis=1)

                                    adjusted_percentage_array_loads_lowest_prices_temp = percentage_array_loads_per_timeslot_lowest_prices.reshape(12, 2)
                                    adjusted_percentage_array_loads_lowest_prices = adjusted_percentage_array_loads_lowest_prices_temp.sum(axis=1)

                                    adjusted_array_lowest_prices = np.minimum(reshaped_array_lowest_prices[::2], reshaped_array_lowest_prices[1::2])
                                    adjusted_array_highest_prices = np.minimum(reshaped_array_highest_prices[::2],reshaped_array_highest_prices[1::2])

                                    # Parameters of the agent (action and state space)
                                    timeslots_for_state_load_percentages_costs = 5 # This number must be equal to the number of time slots which the agent was trained with
                                    number_of_discrete_shifting_actions = 20
                                    minimum_shifting_percentage = 20
                                    maximum_shifting_percentage = 40

                                    percentage_array_loads_per_timeslot_highest_prices_shortened = adjusted_percentage_array_loads_highest_prices [0:timeslots_for_state_load_percentages_costs]
                                    percentage_array_loads_per_timeslot_lowest_prices_shortened = adjusted_percentage_array_loads_lowest_prices [0:timeslots_for_state_load_percentages_costs]

                                    percentage_array_loads_per_timeslot_highest_prices_shortened_before_action = percentage_array_loads_per_timeslot_highest_prices_shortened
                                    percentage_array_loads_per_timeslot_lowest_prices_shortened_before_action = percentage_array_loads_per_timeslot_lowest_prices_shortened


                                    #Define the dimensionality of the observation space (dimensinality = timeslots_for_state_load_percentages_costs * 2)
                                    percentage_array_loads_per_timeslot_highest_prices_shortened = adjusted_percentage_array_loads_highest_prices [0:timeslots_for_state_load_percentages_costs]
                                    percentage_array_loads_per_timeslot_lowest_prices_shortened = adjusted_percentage_array_loads_lowest_prices [0:timeslots_for_state_load_percentages_costs]

                                    percentage_array_loads_per_timeslot_highest_prices_shortened_before_action = percentage_array_loads_per_timeslot_highest_prices_shortened
                                    percentage_array_loads_per_timeslot_lowest_prices_shortened_before_action = percentage_array_loads_per_timeslot_lowest_prices_shortened


                                    param_1_number_of_partitions_for_shifting = 1

                                    #Load the saved RL model from file for the price-shift operatore RL2
                                    from stable_baselines3 import PPO
                                    model_path_extension_RL2 = dir_price_shift_model #"RL2_Days12_SolSol10_SolIt10_ItDay2_ResStateTrue_StateTimeSlots5_ShiftActions20_PPO_RC/trained_PPO_model"#"RL2_Days12_SolSol10_SolIt10_ItDay3_ResStateTrue_StateTimeSlots5_ShiftActions20_PPO_tC/trained_PPO_model"
                                    model_RL2 = PPO.load(os.path.join(config.DIR_RL_MODELS, model_path_extension_RL2))

                                    observation_space = (np.concatenate((percentage_array_loads_per_timeslot_highest_prices_shortened, percentage_array_loads_per_timeslot_lowest_prices_shortened))).reshape(-1)


                                    for helpIterationRLAction in range (0,2):
                                        #Get the action from the RL agent
                                        action_rl_agent1, _ = model_RL2.predict(observation_space, deterministic=False)
                                        action_from_timeslot = action_rl_agent1[0]
                                        action_to_timeslot = action_rl_agent1[1]
                                        action_shifting_percentage = minimum_shifting_percentage + ((maximum_shifting_percentage - minimum_shifting_percentage)/ number_of_discrete_shifting_actions) * action_rl_agent1[2]


                                        for helpIteration in range (0,2):
                                            #Shift profiles cosindering the full day partition
                                            if param_1_number_of_partitions_for_shifting == 1:

                                                # Change the profiles of BT1
                                                for index_BT1 in indexOfBuildingsOverall_BT1:
                                                    help_balance_time_slot = helpIteration
                                                    #Shift Space heating power from highes price timeslot to lowest: Full day
                                                    help_output_mod_degree_old = outputVector_BT1_heatGenerationCoefficientSpaceHeating [index_BT1 - 1] [adjusted_array_highest_prices[action_from_timeslot] + helpIteration]
                                                    outputVector_BT1_heatGenerationCoefficientSpaceHeating [index_BT1 - 1] [adjusted_array_highest_prices[action_from_timeslot] + helpIteration]  = outputVector_BT1_heatGenerationCoefficientSpaceHeating [index_BT1 - 1] [reshaped_array_highest_prices[action_from_timeslot]]  - action_shifting_percentage/100 -0.0
                                                    help_output_mod_degree_new = outputVector_BT1_heatGenerationCoefficientSpaceHeating [index_BT1 - 1] [adjusted_array_highest_prices[action_from_timeslot] + helpIteration]
                                                    if help_output_mod_degree_new <0:
                                                        help_output_mod_degree_new = 0
                                                    changed_mod_degree = help_output_mod_degree_old - help_output_mod_degree_new
                                                    outputVector_BT1_heatGenerationCoefficientSpaceHeating [index_BT1 - 1] [adjusted_array_lowest_prices[action_to_timeslot] + helpIteration]  = outputVector_BT1_heatGenerationCoefficientSpaceHeating [index_BT1 - 1] [adjusted_array_lowest_prices[action_to_timeslot] + helpIteration]   +  changed_mod_degree

                                                    action_from_timeslot, action_to_timeslot, action_shifting_percentage

                                                    if outputVector_BT1_heatGenerationCoefficientSpaceHeating [index_BT1 - 1] [adjusted_array_lowest_prices[action_to_timeslot] + helpIteration]    < SetUpScenarios.minimalModulationdDegree_HP /100:
                                                        outputVector_BT1_heatGenerationCoefficientSpaceHeating[index_BT1 - 1][adjusted_array_lowest_prices[action_to_timeslot] + helpIteration]   = SetUpScenarios.minimalModulationdDegree_HP /100
                                                    if outputVector_BT1_heatGenerationCoefficientSpaceHeating [index_BT1 - 1] [adjusted_array_highest_prices[action_from_timeslot] + helpIteration]   < SetUpScenarios.minimalModulationdDegree_HP /100:
                                                        outputVector_BT1_heatGenerationCoefficientSpaceHeating[index_BT1 - 1][adjusted_array_highest_prices[action_from_timeslot] + helpIteration]  = 0

                                                    #Shift DHW heating power from highes price timeslot to lowest
                                                    if outputVector_BT1_heatGenerationCoefficientDHW [index_BT1 - 1]  [adjusted_array_highest_prices[action_from_timeslot] + helpIteration]  > 0.1:
                                                        help_output_mod_degree_old =  outputVector_BT1_heatGenerationCoefficientDHW [index_BT1 - 1]  [adjusted_array_highest_prices[action_from_timeslot] + helpIteration]
                                                        outputVector_BT1_heatGenerationCoefficientDHW [index_BT1 - 1]  [adjusted_array_highest_prices[action_from_timeslot] + helpIteration]  = outputVector_BT1_heatGenerationCoefficientDHW [index_BT1 - 1] [adjusted_array_highest_prices[action_from_timeslot] + helpIteration]    - action_shifting_percentage/100 -0.0
                                                        help_output_mod_degree_new =  outputVector_BT1_heatGenerationCoefficientDHW [index_BT1 - 1]  [adjusted_array_highest_prices[action_from_timeslot] + helpIteration]
                                                        if help_output_mod_degree_new <0:
                                                            help_output_mod_degree_new = 0
                                                        changed_mod_degree = help_output_mod_degree_old - help_output_mod_degree_new
                                                        outputVector_BT1_heatGenerationCoefficientDHW [index_BT1 - 1] [adjusted_array_lowest_prices[action_to_timeslot] + helpIteration]  = outputVector_BT1_heatGenerationCoefficientDHW [index_BT1 - 1]  [adjusted_array_lowest_prices[action_to_timeslot] + helpIteration]    + changed_mod_degree

                                                        if outputVector_BT1_heatGenerationCoefficientDHW [index_BT1 - 1] [adjusted_array_lowest_prices[action_to_timeslot] + helpIteration]  < SetUpScenarios.minimalModulationdDegree_HP /100:
                                                            outputVector_BT1_heatGenerationCoefficientDHW[index_BT1 - 1][adjusted_array_lowest_prices[action_to_timeslot] + helpIteration] = SetUpScenarios.minimalModulationdDegree_HP /100
                                                        if outputVector_BT1_heatGenerationCoefficientDHW [index_BT1 - 1] [adjusted_array_highest_prices[action_from_timeslot] + helpIteration]  < SetUpScenarios.minimalModulationdDegree_HP /100:
                                                            outputVector_BT1_heatGenerationCoefficientDHW[index_BT1 - 1][adjusted_array_highest_prices[action_from_timeslot] + helpIteration] = 0


                                                    # Shift charging power EV from highes price timeslot to lowest
                                                    help_output_EVpower_old = outputVector_BT1_chargingPowerEV [index_BT1 - 1] [adjusted_array_highest_prices[action_from_timeslot] + helpIteration]
                                                    outputVector_BT1_chargingPowerEV [index_BT1 - 1] [adjusted_array_highest_prices[action_from_timeslot] + helpIteration]  = outputVector_BT1_chargingPowerEV [index_BT1 - 1] [adjusted_array_highest_prices[action_from_timeslot] + helpIteration]    - (action_shifting_percentage*2/100) *SetUpScenarios.chargingPowerMaximal_EV
                                                    if outputVector_BT1_chargingPowerEV [index_BT1 - 1] [adjusted_array_highest_prices[action_from_timeslot] + helpIteration] <0:
                                                        outputVector_BT1_chargingPowerEV [index_BT1 - 1] [adjusted_array_highest_prices[action_from_timeslot] + helpIteration]  = 0
                                                    help_output_EVpower_new = outputVector_BT1_chargingPowerEV [index_BT1 - 1] [adjusted_array_highest_prices[action_from_timeslot] + helpIteration]
                                                    changed_EVpower = help_output_EVpower_old - help_output_EVpower_new

                                                    # Check if the EV is available at the desired timeslot for increasing the load
                                                    help_value_timeslot_addition = 0
                                                    while availability_pattern_EV_BT1 [index_BT1 - 1] [reshaped_array_lowest_prices[action_to_timeslot + help_value_timeslot_addition]] ==0 and action_to_timeslot + help_value_timeslot_addition<len(reshaped_array_lowest_prices) - 1:
                                                        help_value_timeslot_addition += 1
                                                    outputVector_BT1_chargingPowerEV [index_BT1 - 1] [reshaped_array_lowest_prices[action_to_timeslot + help_value_timeslot_addition]] = outputVector_BT1_chargingPowerEV [index_BT1 - 1]  [reshaped_array_lowest_prices[action_to_timeslot + help_value_timeslot_addition]]   + changed_EVpower


                                                # Change the profiles of BT2
                                                for index_BT2 in indexOfBuildingsOverall_BT2:
                                                    help_balance_time_slot = helpIteration
                                                    #Shift Space heating power from highes price timeslot to lowest: Full day
                                                    help_output_mod_degree_old = outputVector_BT2_heatGenerationCoefficientSpaceHeating [index_BT2 - 1] [adjusted_array_highest_prices[action_from_timeslot] + helpIteration]
                                                    outputVector_BT2_heatGenerationCoefficientSpaceHeating [index_BT2 - 1] [adjusted_array_highest_prices[action_from_timeslot] + helpIteration]  = outputVector_BT2_heatGenerationCoefficientSpaceHeating [index_BT2 - 1] [adjusted_array_highest_prices[action_from_timeslot] + helpIteration]    - action_shifting_percentage/100 -0.0
                                                    help_output_mod_degree_new = outputVector_BT2_heatGenerationCoefficientSpaceHeating [index_BT2 - 1] [adjusted_array_highest_prices[action_from_timeslot] + helpIteration]
                                                    if help_output_mod_degree_new <0:
                                                        help_output_mod_degree_new = 0
                                                    changed_mod_degree = help_output_mod_degree_old - help_output_mod_degree_new
                                                    outputVector_BT2_heatGenerationCoefficientSpaceHeating [index_BT2 - 1] [adjusted_array_lowest_prices[action_to_timeslot] + helpIteration]  = outputVector_BT2_heatGenerationCoefficientSpaceHeating [index_BT2 - 1]  [adjusted_array_lowest_prices[action_to_timeslot] + helpIteration]   +  changed_mod_degree


                                                    if outputVector_BT2_heatGenerationCoefficientSpaceHeating [index_BT2 - 1] [adjusted_array_lowest_prices[action_to_timeslot] + helpIteration]   < SetUpScenarios.minimalModulationdDegree_HP /100:
                                                        outputVector_BT2_heatGenerationCoefficientSpaceHeating[index_BT2 - 1][adjusted_array_lowest_prices[action_to_timeslot] + helpIteration]  = SetUpScenarios.minimalModulationdDegree_HP /100
                                                    if outputVector_BT2_heatGenerationCoefficientSpaceHeating [index_BT2 - 1] [adjusted_array_highest_prices[action_from_timeslot] + helpIteration]   < SetUpScenarios.minimalModulationdDegree_HP /100:
                                                        outputVector_BT2_heatGenerationCoefficientSpaceHeating[index_BT2 - 1][adjusted_array_highest_prices[action_from_timeslot] + helpIteration]  = 0
                                                    #Shift DHW heating power from highes price timeslot to lowest

                                                    if outputVector_BT2_heatGenerationCoefficientDHW [index_BT2 - 1]  [adjusted_array_highest_prices[action_from_timeslot] + helpIteration]  > 0.1:
                                                        help_output_mod_degree_old =  outputVector_BT2_heatGenerationCoefficientDHW [index_BT2 - 1]  [adjusted_array_highest_prices[action_from_timeslot] + helpIteration]
                                                        outputVector_BT2_heatGenerationCoefficientDHW [index_BT2 - 1]  [adjusted_array_highest_prices[action_from_timeslot] + helpIteration]  = outputVector_BT2_heatGenerationCoefficientDHW [index_BT2 - 1] [adjusted_array_highest_prices[action_from_timeslot] + helpIteration]    - action_shifting_percentage/100 -0.0
                                                        help_output_mod_degree_new =  outputVector_BT2_heatGenerationCoefficientDHW [index_BT2 - 1]  [adjusted_array_highest_prices[action_from_timeslot] + helpIteration]
                                                        if help_output_mod_degree_new <0:
                                                            help_output_mod_degree_new = 0
                                                        changed_mod_degree = help_output_mod_degree_old - help_output_mod_degree_new
                                                        outputVector_BT2_heatGenerationCoefficientDHW [index_BT2 - 1] [adjusted_array_lowest_prices[action_to_timeslot] + helpIteration]  = outputVector_BT2_heatGenerationCoefficientDHW [index_BT2 - 1]  [adjusted_array_lowest_prices[action_to_timeslot] + helpIteration]    + changed_mod_degree

                                                        if outputVector_BT2_heatGenerationCoefficientDHW [index_BT2 - 1] [adjusted_array_lowest_prices[action_to_timeslot] + helpIteration]  < SetUpScenarios.minimalModulationdDegree_HP /100:
                                                            outputVector_BT2_heatGenerationCoefficientDHW[index_BT2 - 1][adjusted_array_lowest_prices[action_to_timeslot] + helpIteration] = SetUpScenarios.minimalModulationdDegree_HP /100
                                                        if outputVector_BT2_heatGenerationCoefficientDHW [index_BT2 - 1] [adjusted_array_highest_prices[action_from_timeslot] + helpIteration]  < SetUpScenarios.minimalModulationdDegree_HP /100:
                                                            outputVector_BT2_heatGenerationCoefficientDHW[index_BT2 - 1][adjusted_array_highest_prices[action_from_timeslot] + helpIteration] = 0

                                                # Change the profiles of BT4
                                                for index_BT4 in indexOfBuildingsOverall_BT4:

                                                    help_output_mod_degree_old = outputVector_BT4_heatGenerationCoefficientSpaceHeating [index_BT4 - 1] [adjusted_array_highest_prices[action_from_timeslot] + helpIteration]
                                                    outputVector_BT4_heatGenerationCoefficientSpaceHeating [index_BT4 - 1] [adjusted_array_highest_prices[action_from_timeslot] + helpIteration]  = outputVector_BT4_heatGenerationCoefficientSpaceHeating [index_BT4 - 1] [adjusted_array_highest_prices[action_from_timeslot] + helpIteration]    - action_shifting_percentage/100 -0.0
                                                    help_output_mod_degree_new = outputVector_BT4_heatGenerationCoefficientSpaceHeating [index_BT4 - 1] [adjusted_array_highest_prices[action_from_timeslot] + helpIteration]

                                                    if outputVector_BT4_heatGenerationCoefficientSpaceHeating [index_BT4 - 1] [adjusted_array_lowest_prices[action_to_timeslot] + helpIteration]   < SetUpScenarios.minimalModulationdDegree_HP /100:
                                                        outputVector_BT4_heatGenerationCoefficientSpaceHeating[index_BT4 - 1][adjusted_array_lowest_prices[action_to_timeslot] + helpIteration]  = SetUpScenarios.minimalModulationdDegree_HP /100
                                                    if outputVector_BT4_heatGenerationCoefficientSpaceHeating [index_BT4 - 1] [adjusted_array_highest_prices[action_from_timeslot] + helpIteration]   < SetUpScenarios.minimalModulationdDegree_HP /100:
                                                        outputVector_BT4_heatGenerationCoefficientSpaceHeating[index_BT4 - 1][adjusted_array_highest_prices[action_from_timeslot] + helpIteration]  = 0

                                                    changed_mod_degree = help_output_mod_degree_old - help_output_mod_degree_new
                                                    outputVector_BT4_heatGenerationCoefficientSpaceHeating [index_BT4 - 1] [adjusted_array_lowest_prices[action_to_timeslot] + helpIteration]  = outputVector_BT4_heatGenerationCoefficientSpaceHeating [index_BT4 - 1]  [adjusted_array_lowest_prices[action_to_timeslot] + helpIteration]   +  changed_mod_degree


                            if optimize_peak_local_search == True:




                                # Determine highest load
                                help_array_electrical_load = electrical_load_profile_current_solution.copy()
                                help_price_array = price_array.copy()
                                max_load = np.max(electrical_load_profile_current_solution)
                                # Find the time slot indexes that are very close to the maxium load
                                list_timeslots_max_load = []
                                for t in range(0, len(electrical_load_profile_current_solution)):
                                    if electrical_load_profile_current_solution[t] + 50 >= max_load:
                                        help_array_electrical_load [t] = 0
                                        help_price_array [t] = 99999
                                        list_timeslots_max_load.append(t)

                                #Determine second highes load
                                max_load_second = np.max(help_array_electrical_load)
                                percentage_difference_highest_loads = ((max_load - max_load_second) / max_load)*100

                                # Find the timeslots with the highgest prices
                                k = int(60 /SetUpScenarios.timeResolution_InMinutes) #Price stages per part of timeslots: One price for every hour (day-ahead market) leads to the 60. In case of intraday market data it would be 15/timeResolutoin
                                sorted_indices_ascending = np.argsort(-help_price_array)
                                sorted_indices_descending = np.argsort(help_price_array)
                                highest_k_prices_array_full_day_1 = np.array([sorted_indices_ascending[:k],sorted_indices_ascending[k:2 * k],sorted_indices_ascending[2 * k:3 * k], sorted_indices_ascending[3 * k:4 * k],sorted_indices_ascending[4 * k:5 * k],sorted_indices_ascending[5 * k:6 * k],sorted_indices_ascending[6 * k:7 * k],sorted_indices_ascending[7 * k:8 * k],sorted_indices_ascending[8 * k:9 * k],sorted_indices_ascending[9 * k:10 * k],sorted_indices_ascending[10 * k:11 * k],sorted_indices_ascending[11 * k:12 * k]])
                                lowest_k_prices_array_full_day_1 = np.array([sorted_indices_descending[:k],sorted_indices_descending[k:2 * k],sorted_indices_descending[2 * k:3 * k], sorted_indices_descending[3 * k:4 * k],sorted_indices_descending[4 * k:5 * k],sorted_indices_descending[5 * k:6 * k],sorted_indices_descending[6 * k:7 * k],sorted_indices_descending[7 * k:8 * k],sorted_indices_descending[8 * k:9 * k],sorted_indices_descending[9 * k:10 * k],sorted_indices_descending[10 * k:11 * k],sorted_indices_descending[11 * k:12 * k]])


                                if not withRL:
                                    #Reduce load at all peaks
                                    #shifting_percentage_load = percentage_difference_highest_loads + random.uniform(2, 8)
                                    shifting_percentage_load = random.uniform(10-index_iteration, 25-index_iteration)
                                    if shifting_percentage_load < 0:
                                        shifting_percentage_load = 0

                                    for peak_timeslot in list_timeslots_max_load:

                                        # Shift Space heating power, DHW and EV charging from highes peak to lowest price
                                        for index_BT1 in indexOfBuildingsOverall_BT1:
                                            #Determine time slot to shift the load to
                                            random_number = random.uniform(0, 100)
                                            if random_number >= 0 and random_number < 20:
                                                assigned_price_interval = 0
                                            if random_number >= 20 and random_number <35:
                                                assigned_price_interval = 1
                                            if random_number >= 35 and random_number <50:
                                                assigned_price_interval = 2
                                            if random_number >= 50 and random_number <65:
                                                assigned_price_interval = 3
                                            if random_number >= 65 and random_number <80:
                                                assigned_price_interval = 4
                                            if random_number >= 80 and random_number <95:
                                                assigned_price_interval = 5
                                            if random_number >= 95 and random_number <100:
                                                assigned_price_interval = 6

                                            random_timeslot_in_assigned_interval = random.randint(0, k-1)

                                            #Shift Space heating power from highes price timeslot to lowest: Full day
                                            outputVector_BT1_heatGenerationCoefficientSpaceHeating [index_BT1 - 1] [peak_timeslot] = outputVector_BT1_heatGenerationCoefficientSpaceHeating [index_BT1 - 1] [peak_timeslot]   * ((100-shifting_percentage_load)/100)
                                            if lowest_k_prices_array_full_day_1[assigned_price_interval][random_timeslot_in_assigned_interval] not in list_timeslots_max_load:
                                                outputVector_BT1_heatGenerationCoefficientSpaceHeating [index_BT1 - 1] [lowest_k_prices_array_full_day_1[assigned_price_interval] [random_timeslot_in_assigned_interval]] = outputVector_BT1_heatGenerationCoefficientSpaceHeating [index_BT1 - 1]  [lowest_k_prices_array_full_day_1[assigned_price_interval] [random_timeslot_in_assigned_interval]]   * ((100+shifting_percentage_load)/100)
                                            if outputVector_BT1_heatGenerationCoefficientSpaceHeating [index_BT1 - 1] [lowest_k_prices_array_full_day_1[assigned_price_interval] [random_timeslot_in_assigned_interval]] < SetUpScenarios.minimalModulationdDegree_HP /100:
                                                outputVector_BT1_heatGenerationCoefficientSpaceHeating [index_BT1 - 1] [lowest_k_prices_array_full_day_1[assigned_price_interval] [random_timeslot_in_assigned_interval]]= SetUpScenarios.minimalModulationdDegree_HP /100
                                            if outputVector_BT1_heatGenerationCoefficientSpaceHeating [index_BT1 - 1] [peak_timeslot] < SetUpScenarios.minimalModulationdDegree_HP/100:
                                                outputVector_BT1_heatGenerationCoefficientSpaceHeating[index_BT1 - 1][peak_timeslot] = 0

                                                #Shift DHW heating power from highes price timeslot to lowest
                                            outputVector_BT1_heatGenerationCoefficientDHW [index_BT1 - 1]  [peak_timeslot] = outputVector_BT1_heatGenerationCoefficientDHW [index_BT1 - 1] [peak_timeslot]    * ((100-shifting_percentage_load)/100)
                                            if lowest_k_prices_array_full_day_1[assigned_price_interval] [random_timeslot_in_assigned_interval] not in list_timeslots_max_load:
                                                outputVector_BT1_heatGenerationCoefficientDHW [index_BT1 - 1] [lowest_k_prices_array_full_day_1[assigned_price_interval] [random_timeslot_in_assigned_interval]] = outputVector_BT1_heatGenerationCoefficientDHW [index_BT1 - 1]  [lowest_k_prices_array_full_day_1[assigned_price_interval] [random_timeslot_in_assigned_interval]]   * ((100+shifting_percentage_load)/100)
                                            if outputVector_BT1_heatGenerationCoefficientDHW [index_BT1 - 1] [lowest_k_prices_array_full_day_1[assigned_price_interval] [random_timeslot_in_assigned_interval]] < SetUpScenarios.minimalModulationdDegree_HP /100:
                                                outputVector_BT1_heatGenerationCoefficientDHW [index_BT1 - 1] [lowest_k_prices_array_full_day_1[assigned_price_interval] [random_timeslot_in_assigned_interval]]= SetUpScenarios.minimalModulationdDegree_HP /100
                                            if outputVector_BT1_heatGenerationCoefficientDHW [index_BT1 - 1] [peak_timeslot] < SetUpScenarios.minimalModulationdDegree_HP/100:
                                                outputVector_BT1_heatGenerationCoefficientDHW[index_BT1 - 1][peak_timeslot] = 0
                                            # Shift charging power EV from highes price timeslot to lowest
                                            outputVector_BT1_chargingPowerEV [index_BT1 - 1]  [peak_timeslot]  = outputVector_BT1_chargingPowerEV [index_BT1 - 1] [peak_timeslot]    * ((100-shifting_percentage_load)/100)
                                            if lowest_k_prices_array_full_day_1[assigned_price_interval] [random_timeslot_in_assigned_interval] not in list_timeslots_max_load:
                                                outputVector_BT1_chargingPowerEV [index_BT1 - 1] [lowest_k_prices_array_full_day_1[assigned_price_interval] [random_timeslot_in_assigned_interval]] = outputVector_BT1_chargingPowerEV [index_BT1 - 1] [lowest_k_prices_array_full_day_1[assigned_price_interval] [random_timeslot_in_assigned_interval]]   * ((100+shifting_percentage_load)/100)

                                        # Shift Space heating power and DHW from highes peak to lowest price
                                        for index_BT2 in indexOfBuildingsOverall_BT2:


                                            #Determine time slot to shift the load to
                                            random_number = random.uniform(0, 100)
                                            if random_number >= 0 and random_number < 20:
                                                assigned_price_interval = 0
                                            if random_number >= 20 and random_number <35:
                                                assigned_price_interval = 1
                                            if random_number >= 35 and random_number <50:
                                                assigned_price_interval = 2
                                            if random_number >= 50 and random_number <65:
                                                assigned_price_interval = 3
                                            if random_number >= 65 and random_number <80:
                                                assigned_price_interval = 4
                                            if random_number >= 80 and random_number <95:
                                                assigned_price_interval = 5
                                            if random_number >= 95 and random_number <100:
                                                assigned_price_interval = 6

                                            random_timeslot_in_assigned_interval = random.randint(0, k-1)

                                            #Shift Space heating power from highes price timeslot to lowest: Full day
                                            outputVector_BT2_heatGenerationCoefficientSpaceHeating [index_BT2 - 1] [peak_timeslot] = outputVector_BT2_heatGenerationCoefficientSpaceHeating [index_BT2 - 1] [peak_timeslot]   * ((100-shifting_percentage_load)/100)
                                            if lowest_k_prices_array_full_day_1[assigned_price_interval] [random_timeslot_in_assigned_interval] not in list_timeslots_max_load:
                                                outputVector_BT2_heatGenerationCoefficientSpaceHeating [index_BT2 - 1] [lowest_k_prices_array_full_day_1[assigned_price_interval] [random_timeslot_in_assigned_interval]] = outputVector_BT2_heatGenerationCoefficientSpaceHeating [index_BT2 - 1]  [lowest_k_prices_array_full_day_1[assigned_price_interval] [random_timeslot_in_assigned_interval]]   * ((100+shifting_percentage_load)/100)
                                            if outputVector_BT2_heatGenerationCoefficientSpaceHeating [index_BT2 - 1] [lowest_k_prices_array_full_day_1[assigned_price_interval] [random_timeslot_in_assigned_interval]] < SetUpScenarios.minimalModulationdDegree_HP /100:
                                                outputVector_BT2_heatGenerationCoefficientSpaceHeating [index_BT2 - 1] [lowest_k_prices_array_full_day_1[assigned_price_interval] [random_timeslot_in_assigned_interval]]= SetUpScenarios.minimalModulationdDegree_HP /100
                                            if outputVector_BT2_heatGenerationCoefficientSpaceHeating [index_BT2 - 1] [peak_timeslot] < SetUpScenarios.minimalModulationdDegree_HP/100:
                                                outputVector_BT2_heatGenerationCoefficientSpaceHeating[index_BT2 - 1][peak_timeslot] = 0


                                            #Shift DHW heating power from highes price timeslot to lowest
                                            outputVector_BT2_heatGenerationCoefficientDHW [index_BT2 - 1]  [peak_timeslot] = outputVector_BT2_heatGenerationCoefficientDHW [index_BT2 - 1] [peak_timeslot]    * ((100-shifting_percentage_load)/100)
                                            if lowest_k_prices_array_full_day_1[assigned_price_interval][random_timeslot_in_assigned_interval] not in list_timeslots_max_load:
                                                outputVector_BT2_heatGenerationCoefficientDHW [index_BT2 - 1] [lowest_k_prices_array_full_day_1[assigned_price_interval] [random_timeslot_in_assigned_interval]] = outputVector_BT2_heatGenerationCoefficientDHW [index_BT2 - 1]  [lowest_k_prices_array_full_day_1[assigned_price_interval] [random_timeslot_in_assigned_interval]]   * ((100+shifting_percentage_load)/100)
                                            if outputVector_BT2_heatGenerationCoefficientDHW [index_BT2 - 1] [lowest_k_prices_array_full_day_1[assigned_price_interval] [random_timeslot_in_assigned_interval]] < SetUpScenarios.minimalModulationdDegree_HP /100:
                                                outputVector_BT2_heatGenerationCoefficientDHW [index_BT2 - 1] [lowest_k_prices_array_full_day_1[assigned_price_interval] [random_timeslot_in_assigned_interval]]= SetUpScenarios.minimalModulationdDegree_HP /100
                                            if outputVector_BT2_heatGenerationCoefficientDHW [index_BT2 - 1] [peak_timeslot] < SetUpScenarios.minimalModulationdDegree_HP/100:
                                                outputVector_BT2_heatGenerationCoefficientDHW[index_BT2 - 1][peak_timeslot] = 0


                                        for index_BT4 in indexOfBuildingsOverall_BT4:
                                            #Shift Space heating power from highes peak to lowest price

                                            #Determine time slot to shift the load to
                                            random_number = random.uniform(0, 100)
                                            if random_number >= 0 and random_number < 20:
                                                assigned_price_interval = 0
                                            if random_number >= 20 and random_number <35:
                                                assigned_price_interval = 1
                                            if random_number >= 35 and random_number <50:
                                                assigned_price_interval = 2
                                            if random_number >= 50 and random_number <65:
                                                assigned_price_interval = 3
                                            if random_number >= 65 and random_number <80:
                                                assigned_price_interval = 4
                                            if random_number >= 80 and random_number <95:
                                                assigned_price_interval = 5
                                            if random_number >= 95 and random_number <100:
                                                assigned_price_interval = 6

                                            random_timeslot_in_assigned_interval = random.randint(0, k-1)

                                            #Shift Space heating power from highes price timeslot to lowest: Full day
                                            outputVector_BT4_heatGenerationCoefficientSpaceHeating [index_BT4 - 1] [peak_timeslot] = outputVector_BT4_heatGenerationCoefficientSpaceHeating [index_BT4 - 1] [peak_timeslot]   * ((100-shifting_percentage_load)/100)
                                            if lowest_k_prices_array_full_day_1[assigned_price_interval][random_timeslot_in_assigned_interval] not in list_timeslots_max_load:
                                                outputVector_BT4_heatGenerationCoefficientSpaceHeating [index_BT4 - 1] [lowest_k_prices_array_full_day_1[assigned_price_interval] [random_timeslot_in_assigned_interval]] = outputVector_BT4_heatGenerationCoefficientSpaceHeating [index_BT4 - 1]  [lowest_k_prices_array_full_day_1[assigned_price_interval] [random_timeslot_in_assigned_interval]]   * ((100+shifting_percentage_load)/100)

                                            if outputVector_BT4_heatGenerationCoefficientSpaceHeating [index_BT4 - 1] [lowest_k_prices_array_full_day_1[assigned_price_interval] [random_timeslot_in_assigned_interval]] < SetUpScenarios.minimalModulationdDegree_HP /100:
                                                outputVector_BT4_heatGenerationCoefficientSpaceHeating [index_BT4 - 1] [lowest_k_prices_array_full_day_1[assigned_price_interval] [random_timeslot_in_assigned_interval]]= SetUpScenarios.minimalModulationdDegree_HP /100
                                            if outputVector_BT4_heatGenerationCoefficientSpaceHeating [index_BT4 - 1] [peak_timeslot] < SetUpScenarios.minimalModulationdDegree_HP/100:
                                                outputVector_BT4_heatGenerationCoefficientSpaceHeating[index_BT4 - 1][peak_timeslot] = 0

                                if withRL:######
                                    #Calculate the load percentages (in relation to the maxium load) of the timeslots with the lowest prices
                                    reshaped_lowest_prices_fully_day = lowest_k_prices_array_full_day_1.reshape(-1)
                                    array_load_percentages_lowest_prices = np.zeros((len(reshaped_lowest_prices_fully_day)))
                                    for i in range(0, len(reshaped_lowest_prices_fully_day)):
                                        array_load_percentages_lowest_prices[i] = round((electrical_load_profile_current_solution[reshaped_lowest_prices_fully_day[i]] / max_load), 2) * 100

                                    # Parameters of the agent (action and state space)
                                    timeslots_for_state_load_percentages_peak = 5 #4 # This number must be equal to the number of time slots which the agent was trained with
                                    number_of_discrete_shifting_actions = 15
                                    minimum_shifting_percentage = 10
                                    maximum_shifting_percentage = 25

                                    array_load_percentages_lowest_prices_shortened_before = array_load_percentages_lowest_prices[0:timeslots_for_state_load_percentages_peak]

                                    #Load the saved RL model from file for the peak shift operatore RL3
                                    from stable_baselines3 import PPO
                                    model_path_extension_RL3 = dir_peak_shift_model #"RL3_Days12_SolSol10_SolIt10_ItDay2_ResStateTrue_StateTimeSlots5_ShiftActions15_PPO_qf/trained_PPO_model"#"RL3_Days12_SolSol11_SolIt11_ItDay3_ResStateTrue_StateTimeSlots4_ShiftActions15_PPO_wb/trained_PPO_model"
                                    model_RL3 = PPO.load(os.path.join(config.DIR_RL_MODELS, model_path_extension_RL3))


                                    for peak_timeslot in list_timeslots_max_load:


                                        # Get the action from the RL agent
                                        observation_space = array_load_percentages_lowest_prices_shortened_before.reshape(-1)
                                        action_rl_agent1, _ = model_RL3.predict(observation_space, deterministic=False )
                                        action_to_timeslot = action_rl_agent1[0]
                                        action_shifting_percentage = minimum_shifting_percentage + ((maximum_shifting_percentage - minimum_shifting_percentage)/ number_of_discrete_shifting_actions) * action_rl_agent1[1]


                                        #Reduce load at all peaks
                                        shifting_percentage_load = action_shifting_percentage

                                        # Shift Space heating power, DHW and EV charging from highes peak to lowest price
                                        for index_BT1 in indexOfBuildingsOverall_BT1:



                                            #Shift Space heating power from highes price timeslot to lowest: Full day
                                            outputVector_BT1_heatGenerationCoefficientSpaceHeating [index_BT1 - 1] [peak_timeslot] = outputVector_BT1_heatGenerationCoefficientSpaceHeating [index_BT1 - 1] [peak_timeslot]   * ((100-shifting_percentage_load)/100)
                                            if reshaped_lowest_prices_fully_day [action_to_timeslot] not in list_timeslots_max_load:
                                                outputVector_BT1_heatGenerationCoefficientSpaceHeating [index_BT1 - 1] [reshaped_lowest_prices_fully_day [action_to_timeslot]] = outputVector_BT1_heatGenerationCoefficientSpaceHeating [index_BT1 - 1]  [reshaped_lowest_prices_fully_day [action_to_timeslot]]   * ((100+shifting_percentage_load)/100)
                                            if outputVector_BT1_heatGenerationCoefficientSpaceHeating [index_BT1 - 1] [reshaped_lowest_prices_fully_day [action_to_timeslot]] < SetUpScenarios.minimalModulationdDegree_HP /100:
                                                outputVector_BT1_heatGenerationCoefficientSpaceHeating [index_BT1 - 1] [reshaped_lowest_prices_fully_day [action_to_timeslot]]= SetUpScenarios.minimalModulationdDegree_HP /100
                                            if outputVector_BT1_heatGenerationCoefficientSpaceHeating [index_BT1 - 1] [peak_timeslot] < SetUpScenarios.minimalModulationdDegree_HP/100:
                                                outputVector_BT1_heatGenerationCoefficientSpaceHeating[index_BT1 - 1][peak_timeslot] = 0

                                                #Shift DHW heating power from highes price timeslot to lowest
                                            outputVector_BT1_heatGenerationCoefficientDHW [index_BT1 - 1]  [peak_timeslot] = outputVector_BT1_heatGenerationCoefficientDHW [index_BT1 - 1] [peak_timeslot]    * ((100-shifting_percentage_load)/100)
                                            if reshaped_lowest_prices_fully_day [action_to_timeslot] not in list_timeslots_max_load:
                                                outputVector_BT1_heatGenerationCoefficientDHW [index_BT1 - 1] [reshaped_lowest_prices_fully_day [action_to_timeslot]] = outputVector_BT1_heatGenerationCoefficientDHW [index_BT1 - 1]  [reshaped_lowest_prices_fully_day [action_to_timeslot]]   * ((100+shifting_percentage_load)/100)
                                            if outputVector_BT1_heatGenerationCoefficientDHW [index_BT1 - 1] [reshaped_lowest_prices_fully_day [action_to_timeslot]] < SetUpScenarios.minimalModulationdDegree_HP /100:
                                                outputVector_BT1_heatGenerationCoefficientDHW [index_BT1 - 1] [reshaped_lowest_prices_fully_day [action_to_timeslot]]= SetUpScenarios.minimalModulationdDegree_HP /100
                                            if outputVector_BT1_heatGenerationCoefficientDHW [index_BT1 - 1] [peak_timeslot] < SetUpScenarios.minimalModulationdDegree_HP/100:
                                                outputVector_BT1_heatGenerationCoefficientDHW[index_BT1 - 1][peak_timeslot] = 0
                                            # Shift charging power EV from highes price timeslot to lowest
                                            outputVector_BT1_chargingPowerEV [index_BT1 - 1]  [peak_timeslot]  = outputVector_BT1_chargingPowerEV [index_BT1 - 1] [peak_timeslot]    * ((100-shifting_percentage_load)/100)
                                            if reshaped_lowest_prices_fully_day [action_to_timeslot] not in list_timeslots_max_load:
                                                outputVector_BT1_chargingPowerEV [index_BT1 - 1] [reshaped_lowest_prices_fully_day [action_to_timeslot]] = outputVector_BT1_chargingPowerEV [index_BT1 - 1] [reshaped_lowest_prices_fully_day [action_to_timeslot]]   * ((100+shifting_percentage_load)/100)

                                        # Shift Space heating power and DHW from highes peak to lowest price
                                        for index_BT2 in indexOfBuildingsOverall_BT2:




                                            #Shift Space heating power from highes price timeslot to lowest: Full day
                                            outputVector_BT2_heatGenerationCoefficientSpaceHeating [index_BT2 - 1] [peak_timeslot] = outputVector_BT2_heatGenerationCoefficientSpaceHeating [index_BT2 - 1] [peak_timeslot]   * ((100-shifting_percentage_load)/100)
                                            if reshaped_lowest_prices_fully_day [action_to_timeslot] not in list_timeslots_max_load:
                                                outputVector_BT2_heatGenerationCoefficientSpaceHeating [index_BT2 - 1] [reshaped_lowest_prices_fully_day [action_to_timeslot]] = outputVector_BT2_heatGenerationCoefficientSpaceHeating [index_BT2 - 1]  [reshaped_lowest_prices_fully_day [action_to_timeslot]]   * ((100+shifting_percentage_load)/100)
                                            if outputVector_BT2_heatGenerationCoefficientSpaceHeating [index_BT2 - 1] [reshaped_lowest_prices_fully_day [action_to_timeslot]] < SetUpScenarios.minimalModulationdDegree_HP /100:
                                                outputVector_BT2_heatGenerationCoefficientSpaceHeating [index_BT2 - 1] [reshaped_lowest_prices_fully_day [action_to_timeslot]]= SetUpScenarios.minimalModulationdDegree_HP /100
                                            if outputVector_BT2_heatGenerationCoefficientSpaceHeating [index_BT2 - 1] [peak_timeslot] < SetUpScenarios.minimalModulationdDegree_HP/100:
                                                outputVector_BT2_heatGenerationCoefficientSpaceHeating[index_BT2 - 1][peak_timeslot] = 0


                                            #Shift DHW heating power from highes price timeslot to lowest
                                            outputVector_BT2_heatGenerationCoefficientDHW [index_BT2 - 1]  [peak_timeslot] = outputVector_BT2_heatGenerationCoefficientDHW [index_BT2 - 1] [peak_timeslot]    * ((100-shifting_percentage_load)/100)
                                            if reshaped_lowest_prices_fully_day [action_to_timeslot] not in list_timeslots_max_load:
                                                outputVector_BT2_heatGenerationCoefficientDHW [index_BT2 - 1] [reshaped_lowest_prices_fully_day [action_to_timeslot]] = outputVector_BT2_heatGenerationCoefficientDHW [index_BT2 - 1]  [reshaped_lowest_prices_fully_day [action_to_timeslot]]   * ((100+shifting_percentage_load)/100)
                                            if outputVector_BT2_heatGenerationCoefficientDHW [index_BT2 - 1] [reshaped_lowest_prices_fully_day [action_to_timeslot]] < SetUpScenarios.minimalModulationdDegree_HP /100:
                                                outputVector_BT2_heatGenerationCoefficientDHW [index_BT2 - 1] [reshaped_lowest_prices_fully_day [action_to_timeslot]]= SetUpScenarios.minimalModulationdDegree_HP /100
                                            if outputVector_BT2_heatGenerationCoefficientDHW [index_BT2 - 1] [peak_timeslot] < SetUpScenarios.minimalModulationdDegree_HP/100:
                                                outputVector_BT2_heatGenerationCoefficientDHW[index_BT2 - 1][peak_timeslot] = 0


                                        for index_BT4 in indexOfBuildingsOverall_BT4:
                                            #Shift Space heating power from highes peak to lowest price

                                            #Shift Space heating power from highes price timeslot to lowest: Full day
                                            outputVector_BT4_heatGenerationCoefficientSpaceHeating [index_BT4 - 1] [peak_timeslot] = outputVector_BT4_heatGenerationCoefficientSpaceHeating [index_BT4 - 1] [peak_timeslot]   * ((100-shifting_percentage_load)/100)
                                            if reshaped_lowest_prices_fully_day [action_to_timeslot] not in list_timeslots_max_load:
                                                outputVector_BT4_heatGenerationCoefficientSpaceHeating [index_BT4 - 1] [reshaped_lowest_prices_fully_day [action_to_timeslot]] = outputVector_BT4_heatGenerationCoefficientSpaceHeating [index_BT4 - 1]  [reshaped_lowest_prices_fully_day [action_to_timeslot]]   * ((100+shifting_percentage_load)/100)

                                            if outputVector_BT4_heatGenerationCoefficientSpaceHeating [index_BT4 - 1] [reshaped_lowest_prices_fully_day [action_to_timeslot]] < SetUpScenarios.minimalModulationdDegree_HP /100:
                                                outputVector_BT4_heatGenerationCoefficientSpaceHeating [index_BT4 - 1] [reshaped_lowest_prices_fully_day [action_to_timeslot]]= SetUpScenarios.minimalModulationdDegree_HP /100
                                            if outputVector_BT4_heatGenerationCoefficientSpaceHeating [index_BT4 - 1] [peak_timeslot] < SetUpScenarios.minimalModulationdDegree_HP/100:
                                                outputVector_BT4_heatGenerationCoefficientSpaceHeating[index_BT4 - 1][peak_timeslot] = 0


                            if optimize_comfort_local_search == True:
                                #Optimize the comfort for BT1
                                for index_BT1 in indexOfBuildingsOverall_BT1:
                                    help_adjusted_timeslots_space_heating_BT1 = np.zeros(len(outputVector_BT1_heatGenerationCoefficientSpaceHeating[index_BT1 - 1] ))
                                    help_adjusted_timeslots_dhw_BT1 = np.zeros(len(outputVector_BT1_heatGenerationCoefficientDHW[index_BT1 - 1] ))

                                    #Get the timeslots with the highest values of the thermal discomfort
                                    abs_values = np.abs(thermal_discomfort_space_heating_BT1[0])
                                    random_number_of_timeslots_correction = random.randint(int(30/SetUpScenarios.timeResolution_InMinutes), int(90/SetUpScenarios.timeResolution_InMinutes))
                                    top_absolute_values = np.sort(abs_values)[-random_number_of_timeslots_correction:]
                                    top_indices = np.argpartition(abs_values, -random_number_of_timeslots_correction)[-random_number_of_timeslots_correction:]
                                    sorted_indices = np.argsort(np.abs(thermal_discomfort_space_heating_BT1[0][top_indices]),kind='mergesort')[::-1]
                                    sorted_top_indices = top_indices[sorted_indices]

                                    for index_timeslot in sorted_top_indices:
                                        #When space heating temperature too high, stop heating in the previous time slots
                                        if thermal_discomfort_space_heating_BT1  [index_BT1-1][index_timeslot] > 0.25 :
                                            random_number_timeslots = random.randint(int(30/SetUpScenarios.timeResolution_InMinutes), int(90/SetUpScenarios.timeResolution_InMinutes))
                                            for i in range (0, random_number_timeslots):
                                                if index_timeslot - i > 0 and help_adjusted_timeslots_space_heating_BT1 [index_timeslot - i] ==0:
                                                    outputVector_BT1_heatGenerationCoefficientSpaceHeating[index_BT1 - 1] [index_timeslot - i] =  0
                                                    help_adjusted_timeslots_space_heating_BT1 [index_timeslot - i] =  1
                                        #When space heating temperature too low,  heat stronger in the previous time slots
                                        if thermal_discomfort_space_heating_BT1 [index_BT1-1][index_timeslot] < - 0.25 :
                                            random_number_timeslots = random.randint(int(30/SetUpScenarios.timeResolution_InMinutes), int(120/SetUpScenarios.timeResolution_InMinutes))
                                            for i in range (0, random_number_timeslots):
                                                if index_timeslot - i > 0 and help_adjusted_timeslots_space_heating_BT1[index_timeslot - i] == 0:
                                                    outputVector_BT1_heatGenerationCoefficientSpaceHeating[index_BT1 - 1][index_timeslot - i] = outputVector_BT1_heatGenerationCoefficientSpaceHeating[index_BT1 - 1][index_timeslot - i] * 1.1 + SetUpScenarios.minimalModulationdDegree_HP/100
                                                    help_adjusted_timeslots_space_heating_BT1[index_timeslot - i] = 1
                                        # When DHW volume  too high, stop heating in the previous time slots
                                        if thermal_discomfort_dhw_BT1 [index_BT1-1][index_timeslot] > 0.25 :
                                            random_number_timeslots = random.randint(int(30/SetUpScenarios.timeResolution_InMinutes), int(120/SetUpScenarios.timeResolution_InMinutes))
                                            for i in range (0, random_number_timeslots):
                                                if index_timeslot - i > 0 and help_adjusted_timeslots_dhw_BT1[index_timeslot - i] == 0:
                                                    outputVector_BT1_heatGenerationCoefficientDHW [index_BT1 - 1][index_timeslot - i] = 0
                                                    help_adjusted_timeslots_dhw_BT1[index_timeslot - i] = 1
                                        #When DHW volume  too low,  heat stronger in the previous time slots
                                        if thermal_discomfort_dhw_BT1 [index_BT1-1][index_timeslot] < -0.25:
                                            random_number_timeslots = random.randint(int(30/SetUpScenarios.timeResolution_InMinutes), int(120/SetUpScenarios.timeResolution_InMinutes))
                                            for i in range (0, random_number_timeslots):
                                                if index_timeslot - i > 0 and help_adjusted_timeslots_dhw_BT1[index_timeslot - i] == 0:
                                                    outputVector_BT1_heatGenerationCoefficientDHW[index_BT1 - 1][index_timeslot - i] = outputVector_BT1_heatGenerationCoefficientDHW[index_BT1 - 1][index_timeslot - i] * 1.2 + SetUpScenarios.minimalModulationdDegree_HP/100
                                                    help_adjusted_timeslots_dhw_BT1[index_timeslot - i] = 1

                                #Optimize the comfort for BT2
                                for index_BT2 in indexOfBuildingsOverall_BT2:
                                    help_adjusted_timeslots_space_heating_BT2 = np.zeros(len(outputVector_BT2_heatGenerationCoefficientSpaceHeating[index_BT2 - 1] ))
                                    help_adjusted_timeslots_dhw_BT2 = np.zeros(len(outputVector_BT2_heatGenerationCoefficientDHW[index_BT2 - 1] ))

                                    #Get the timeslots with the highest values of the thermal discomfort
                                    abs_values = np.abs(thermal_discomfort_space_heating_BT2[0])
                                    random_number_of_timeslots_correction = random.randint(int(30/SetUpScenarios.timeResolution_InMinutes), int(90/SetUpScenarios.timeResolution_InMinutes))
                                    top_absolute_values = np.sort(abs_values)[-random_number_of_timeslots_correction:]
                                    top_indices = np.argpartition(abs_values, -random_number_of_timeslots_correction)[-random_number_of_timeslots_correction:]
                                    sorted_indices = np.argsort(np.abs(thermal_discomfort_space_heating_BT2[0][top_indices]),kind='mergesort')[::-1]
                                    sorted_top_indices = top_indices[sorted_indices]

                                    for index_timeslot in sorted_top_indices:
                                        #When space heating temperature too high, stop heating in the previous time slots
                                        if thermal_discomfort_space_heating_BT2 [index_BT2-1][index_timeslot] > 0.25:
                                            random_number_timeslots = random.randint(int(30/SetUpScenarios.timeResolution_InMinutes), int(120/SetUpScenarios.timeResolution_InMinutes))
                                            for i in range (0, random_number_timeslots):
                                                if index_timeslot - i > 0 and help_adjusted_timeslots_space_heating_BT2  [index_timeslot - i] ==0:
                                                    outputVector_BT2_heatGenerationCoefficientSpaceHeating[index_BT2 - 1] [index_timeslot - i] =  0
                                                    help_adjusted_timeslots_space_heating_BT2 [index_timeslot - i] =1
                                        #When space heating temperature too low,  heat stronger in the previous time slots
                                        if thermal_discomfort_space_heating_BT2 [index_BT2-1][index_timeslot] < -0.25:
                                            random_number_timeslots = random.randint(int(30/SetUpScenarios.timeResolution_InMinutes), int(120/SetUpScenarios.timeResolution_InMinutes))
                                            for i in range (0, random_number_timeslots):
                                                if index_timeslot - i > 0 and help_adjusted_timeslots_space_heating_BT2 [index_timeslot - i] ==0:
                                                    outputVector_BT2_heatGenerationCoefficientSpaceHeating[index_BT2 - 1][index_timeslot - i] = outputVector_BT2_heatGenerationCoefficientSpaceHeating[index_BT2 - 1][index_timeslot - i] * 1.1 + SetUpScenarios.minimalModulationdDegree_HP/100
                                                    help_adjusted_timeslots_space_heating_BT2[index_timeslot - i] = 1
                                        # When DHW volume  too high, stop heating in the previous time slots
                                        if thermal_discomfort_dhw_BT2 [index_BT2-1][index_timeslot] > 0.25:
                                            random_number_timeslots = random.randint(int(30/SetUpScenarios.timeResolution_InMinutes), int(120/SetUpScenarios.timeResolution_InMinutes))
                                            for i in range (0, random_number_timeslots):
                                                if index_timeslot - i > 0 and help_adjusted_timeslots_dhw_BT2 [index_timeslot - i] ==0:
                                                    outputVector_BT2_heatGenerationCoefficientDHW [index_BT2 - 1][index_timeslot - i] = 0
                                                    help_adjusted_timeslots_dhw_BT2[index_timeslot - i] = 1
                                        #When DHW volume  too low,  heat stronger in the previous time slots
                                        if thermal_discomfort_dhw_BT2 [index_BT2-1][index_timeslot] < -0.25:
                                            random_number_timeslots = random.randint(int(30/SetUpScenarios.timeResolution_InMinutes), int(120/SetUpScenarios.timeResolution_InMinutes))
                                            for i in range (0, random_number_timeslots):
                                                if index_timeslot - i > 0 and help_adjusted_timeslots_dhw_BT2 [index_timeslot - i] ==0:
                                                    outputVector_BT2_heatGenerationCoefficientDHW[index_BT2 - 1][index_timeslot - i] = outputVector_BT2_heatGenerationCoefficientDHW[index_BT2 - 1][index_timeslot - i] * 1.2 + SetUpScenarios.minimalModulationdDegree_HP/100
                                                    help_adjusted_timeslots_dhw_BT2[index_timeslot - i] = 1
                                #Optimize the comfort for BT4
                                for index_BT4 in indexOfBuildingsOverall_BT4:
                                    help_adjusted_timeslots_space_heating_BT4 = np.zeros(len(outputVector_BT4_heatGenerationCoefficientSpaceHeating[index_BT4 - 1] ))

                                    #Get the timeslots with the highest values of the thermal discomfort
                                    abs_values = np.abs(thermal_discomfort_space_heating_BT4[0])
                                    random_number_of_timeslots_correction = random.randint(int(30/SetUpScenarios.timeResolution_InMinutes), int(90/SetUpScenarios.timeResolution_InMinutes))
                                    top_absolute_values = np.sort(abs_values)[-random_number_of_timeslots_correction:]
                                    top_indices = np.argpartition(abs_values, -random_number_of_timeslots_correction)[-random_number_of_timeslots_correction:]
                                    sorted_indices = np.argsort(np.abs(thermal_discomfort_space_heating_BT4[0][top_indices]),kind='mergesort')[::-1]
                                    sorted_top_indices = top_indices[sorted_indices]

                                    for index_timeslot in sorted_top_indices:
                                        #When space heating temperature too high, stop heating in the previous time slots
                                        if thermal_discomfort_space_heating_BT4 [index_BT4-1][index_timeslot] > 0.25:
                                            random_number_timeslots = random.randint(int(30/SetUpScenarios.timeResolution_InMinutes), int(120/SetUpScenarios.timeResolution_InMinutes))
                                            for i in range (0, random_number_timeslots):
                                                if index_timeslot - i > 0 and help_adjusted_timeslots_space_heating_BT4  [index_timeslot - i] ==  0:
                                                    outputVector_BT4_heatGenerationCoefficientSpaceHeating[index_BT4 - 1] [index_timeslot - i] =  0
                                                    help_adjusted_timeslots_space_heating_BT4  [index_timeslot - i] =  1
                                        #When space heating temperature too low,  heat stronger in the previous time slots
                                        if thermal_discomfort_space_heating_BT4 [index_BT4-1][index_timeslot] < -0.25:
                                            random_number_timeslots = random.randint(int(30/SetUpScenarios.timeResolution_InMinutes), int(120/SetUpScenarios.timeResolution_InMinutes))
                                            for i in range (0, random_number_timeslots):
                                                if index_timeslot - i > 0 and help_adjusted_timeslots_space_heating_BT4 [index_timeslot - i] ==  0:
                                                    outputVector_BT4_heatGenerationCoefficientSpaceHeating[index_BT4 - 1][index_timeslot - i] = outputVector_BT4_heatGenerationCoefficientSpaceHeating[index_BT4 - 1][index_timeslot - i] * 1.1 + SetUpScenarios.minimalModulationdDegree_HP/100
                                                    help_adjusted_timeslots_space_heating_BT4[index_timeslot - i] = 1


                            #Run simulation with new solution
                            returned_objects = ICSimulation.simulateDays_WithLightController_Schedule(indexOfBuildingsOverall_BT1, indexOfBuildingsOverall_BT2, indexOfBuildingsOverall_BT3, indexOfBuildingsOverall_BT4, indexOfBuildingsOverall_BT5, indexOfBuildingsOverall_BT6, indexOfBuildingsOverall_BT7, currentDay, outputVector_BT1_heatGenerationCoefficientSpaceHeating, outputVector_BT1_heatGenerationCoefficientDHW, outputVector_BT1_chargingPowerEV, outputVector_BT2_heatGenerationCoefficientSpaceHeating, outputVector_BT2_heatGenerationCoefficientDHW, outputVector_BT3_chargingPowerEV, outputVector_BT4_heatGenerationCoefficientSpaceHeating, outputVector_BT5_chargingPowerBAT, outputVector_BT5_disChargingPowerBAT, outputVector_BT6_heatGenerationCoefficient_GasBoiler, outputVector_BT6_heatGenerationCoefficient_ElectricalHeatingElement, outputVector_BT6_heatTransferCoefficient_StorageToRoom, outputVector_BT7_heatGenerationCoefficient_GasBoiler, outputVector_BT7_electricalPowerFanHeater, pathForCreatingTheResultData, preCorrectSchedules_AvoidingFrequentStarts, optParameters, useLocalSearch)
                            results_dict = {"simulationObjective_surplusEnergy_kWh_combined": returned_objects[0], "simulationObjective_maximumLoad_kW_combined": returned_objects[1], "simulationObjective_thermalDiscomfort_combined": returned_objects[2], "simulationObjective_gasConsumptionkWh_combined": returned_objects[3], "simulationObjective_costs_Euro_combined": returned_objects[4], "simulationObjective_combinedScore_combined": returned_objects[5], "simulationResult_electricalLoad_combined": returned_objects[6], "price_array": returned_objects[7], "simulationInput_BT1_availabilityPattern": returned_objects[8], "combined_array_thermal_discomfort": returned_objects[9], "outputVector_BT1_heatGenerationCoefficientSpaceHeating": returned_objects[10], "outputVector_BT1_heatGenerationCoefficientDHW": returned_objects[11], "outputVector_BT1_chargingPowerEV": returned_objects[12], "outputVector_BT2_heatGenerationCoefficientSpaceHeating": returned_objects[13], "outputVector_BT2_heatGenerationCoefficientDHW": returned_objects[14], "outputVector_BT3_chargingPowerEV": returned_objects[15], "outputVector_BT4_heatGenerationCoefficientSpaceHeating": returned_objects[16], "outputVector_BT5_chargingPowerBAT": returned_objects[17], "outputVector_BT5_disChargingPowerBAT": returned_objects[18], "outputVector_BT6_heatGenerationCoefficient_GasBoiler": returned_objects[19], "outputVector_BT6_heatGenerationCoefficient_ElectricalHeatingElement": returned_objects[20], "outputVector_BT6_heatTransferCoefficient_StorageToRoom": returned_objects[21], "outputVector_BT7_heatGenerationCoefficient_GasBoiler": returned_objects[22], "outputVector_BT7_electricalPowerFanHeater": returned_objects[23], "combined_array_thermal_discomfort": returned_objects[24], "thermal_discomfort_space_heating_BT1": returned_objects[25], "thermal_discomfort_dhw_BT1": returned_objects[26], "thermal_discomfort_space_heating_BT2": returned_objects[27], "thermal_discomfort_dhw_BT2": returned_objects[28], "thermal_discomfort_space_heating_BT4": returned_objects[29], "id of the run": id_of_the_run}

                            help_print_optimization_operator = ""
                            if optimize_costs_local_search == True:
                                help_print_optimization_operator = "cost_opt"
                            if optimize_peak_local_search == True:
                                help_print_optimization_operator = "peak_opt"
                            if optimize_comfort_local_search == True:
                                help_print_optimization_operator = "comfort_opt"

                            print(f"id:{id_of_the_run}, Costs:{returned_objects[4]}, Peak:{returned_objects[1]}, DC:{returned_objects[2]}, {help_print_optimization_operator}")
                            results_list_per_iteration.append(results_dict)
                            df_results_per_iteration.loc[id_of_the_run - 1] = [id_of_the_run, results_dict['simulationObjective_costs_Euro_combined'],  results_dict['simulationObjective_maximumLoad_kW_combined'], results_dict['simulationObjective_thermalDiscomfort_combined']]

                            current_time = time.time()
                            current_execution_time = current_time - start_time

                            if current_execution_time >= time_limit_in_seconds_for_local_search:
                                termination_flag_active = True


                    #Identify the non-dominated solutions considering the population for the population of the next iteration
                    df_merged_population_current_iteration = pd.concat([df_results_per_iteration, df_results_population],ignore_index=True)
                    merged_list_population_current_iteration = results_list_per_iteration.copy()  # Create a copy of the first list
                    merged_list_population_current_iteration.extend(results_list_population)

                    df_merged_population_current_iteration['Combined Score'] = (df_merged_population_current_iteration['Costs'] / help_value_normalization_cost_conventional + df_merged_population_current_iteration['Peak Load'] / help_value_normalization_maxiumLoad_conventional)


                    # Convert elements with brackets to numeric format
                    df_merged_population_current_iteration = df_merged_population_current_iteration.applymap(lambda x: float(x.strip('[]')) if isinstance(x, str) else x)
                    # Boolean indexing to filter rows based on the condition
                    df_merged_population_current_iteration = df_merged_population_current_iteration[df_merged_population_current_iteration['Thermal Discomfort'] < threshold_discomfort_local_search]



                    # create a new DataFrame to store the Pareto-optimal solutions
                    pareto_df = pd.DataFrame(columns=df_merged_population_current_iteration.columns)

                    for i, row in df_merged_population_current_iteration.iterrows():
                        is_dominated = False
                        is_duplicate = False
                        for j, other_row in df_merged_population_current_iteration.iterrows():

                            if i == j:
                                continue
                            # Check if the other solution dominates the current solution
                            if (other_row['Costs'] < row['Costs'] and other_row['Peak Load'] < row['Peak Load']) or \
                                    (other_row['Costs'] <= row['Costs'] and other_row['Peak Load'] < row[
                                        'Peak Load']) or \
                                    (other_row['Costs'] < row['Costs'] and other_row['Peak Load'] <= row['Peak Load']):
                                # The other solution dominates the current solution
                                is_dominated = True
                                break
                            # Check if the other solution is a duplicate
                            if (other_row['Costs'] == row['Costs'] and other_row['Peak Load'] == row['Peak Load'] and i > j):
                                is_duplicate = True
                                break

                        if not is_dominated and not is_duplicate and row['Thermal Discomfort'] < threshold_discomfort_local_search:
                            # The current solution is Pareto-optimal, not a duplicate, and meets the discomfort threshold
                            row_df = pd.DataFrame([row])
                            pareto_df = pd.concat([pareto_df, row_df], ignore_index=True)


                    #Build the population for the new generation
                    if index_iteration == 0:
                        results_list_population = merged_list_population_current_iteration.copy()
                    else:
                        #Add pareto optimal solution into the next generation
                        results_list_population_temp = []
                        desired_ids_pareto_optimal = pareto_df['id of the run'].values.tolist()
                        added_ids = set()
                        for item in merged_list_population_current_iteration:
                            if int(item['id of the run']) in desired_ids_pareto_optimal:
                                run_id = int(item['id of the run'])
                                if run_id not in added_ids:
                                    results_list_population_temp.append(item)
                                    added_ids.add(run_id)

                        a = 1
                        #Add additional dominated solutions if the number of pareto optimal solutions is smaller than the population size
                        if len(pareto_df) < max_population_size:
                            desired_ids_dominated = []
                            if len(pareto_df) < number_of_pareto_optimal_solutions_in_population:
                                number_of_pareto_optimal_duplicate_solutions = number_of_pareto_optimal_solutions_in_population - len(pareto_df)
                                number_of_further_solutions = max_population_size  - (number_of_pareto_optimal_duplicate_solutions +len(pareto_df))
                            else:
                                number_of_further_solutions = 0

                            filtered_df = df_merged_population_current_iteration[~df_merged_population_current_iteration['id of the run'].isin(desired_ids_pareto_optimal)]
                            sorted_df = filtered_df.sort_values('Combined Score')
                            selected_rows = sorted_df.head(number_of_further_solutions)
                            desired_ids_dominated.extend(selected_rows['id of the run'].tolist())
                            added_ids = set()
                            for item in merged_list_population_current_iteration:
                                if int(item['id of the run']) in desired_ids_dominated:
                                    run_id = int(item['id of the run'])
                                    if run_id not in added_ids:
                                        results_list_population_temp.append(item)
                                        added_ids.add(run_id)

                            #Randomly choose duplicate pareto-optimal solutions to be in the population of the next generation
                            list_pareto_optimal_duplicate_solutions = []
                            if  len(pareto_df) > 0:
                                for i in range (0, number_of_pareto_optimal_duplicate_solutions):
                                    random_index = random.randint(0, len(pareto_df) - 1)
                                    list_pareto_optimal_duplicate_solutions.append(pareto_df.loc [random_index, 'id of the run'])

                            #Add pareto-optimal duplicate solutions to the temporary list for the popuplation
                            for run_id in list_pareto_optimal_duplicate_solutions:
                                for item in merged_list_population_current_iteration:
                                    if int(item['id of the run']) == run_id:
                                        results_list_population_temp.append(item)
                                        break



                        #Choose the most diverse pareto optimal solutions if the number of pareto optimal soutions is higher than the max population size
                        if len(pareto_df) > max_population_size:
                            import pandas as pd
                            from sklearn.metrics.pairwise import euclidean_distances

                            # Select only the "Costs", "Peak Load", and "id of the run" columns from the pareto_df DataFrame
                            selected_columns = ["Costs", "Peak Load", "id of the run"]
                            data = pareto_df[selected_columns]

                            # Check if the DataFrame is empty or if there are missing values in the relevant columns
                            if data.empty or data[selected_columns[:-1]].isnull().values.any():
                                print("Error: DataFrame is empty or contains missing values.")
                            else:
                                # Initialize the array to store selected id_of_the_run values
                                selected_ids = []
                                pareto_df_help = pareto_df.copy()

                                while len(selected_ids) < max_population_size:
                                    #Assign first solution randomly
                                    if len(selected_ids) == 0:
                                        random_number = random.randint(1, len(pareto_df_help))
                                        selected_ids.append(random_number)
                                        pareto_df_help.loc[pareto_df_help['id of the run'] == random_number, 'Combined Score'] = -1
                                        continue
                                    #Calculate distance of all pareto optimal solution regarding "Combined Score" to all already choosen solutions
                                    if not withRL:
                                        distance_array_to_all_selected_solutions = np.zeros((len(pareto_df_help), len(selected_ids)))
                                    for i in range (0, len (pareto_df_help)):
                                        for j in range(0, len(selected_ids)):
                                            if withRL:
                                                distance_array_to_all_selected_solutions [i] [j] = abs(pareto_df_help.loc[i, "Combined Score"] - pareto_df_help.loc[selected_ids[j], "Combined Score"])
                                            try:
                                                distance = abs( pareto_df_help.loc[i, "Combined Score"] - pareto_df_help.loc[selected_ids[j], "Combined Score"])
                                                distance_array_to_all_selected_solutions[i][j] = distance
                                            except KeyError as e:
                                                # Handle the KeyError (index not found in DataFrame)
                                                print(f"KeyError: {e}. Index {i} or {selected_ids[j]} not found in pareto_df_help.")
                                                distance_array_to_all_selected_solutions[i][j] = -1

                                            if pareto_df_help.loc[i, "Combined Score"] < 0:
                                                distance_array_to_all_selected_solutions[i][j] = -1


                                    row_sums = np.sum(distance_array_to_all_selected_solutions,axis=1)  # Compute the sum along the second dimension
                                    max_sum_index = np.argmax(row_sums)  # Find the index of the maximum sum

                                    selected_ids.append(pareto_df_help.loc[max_sum_index, "id of the run"])
                                    pareto_df_help.loc[pareto_df_help['id of the run'] == selected_ids[len(selected_ids)- 1], 'Combined Score'] = -1

                            #Add  solutions ot the results list
                            results_list_population_temp = []
                            for item in merged_list_population_current_iteration:
                                if int(item['id of the run']) in selected_ids:
                                    results_list_population_temp.append(item)



                        #Elimnate dublicates from the temporary results list
                        results_list_population =  results_list_population_temp
                        df_results_population
                        data = []
                        for entry in results_list_population:
                            id_of_the_run = entry['id of the run']
                            costs = round(entry['simulationObjective_costs_Euro_combined'][0],2)
                            peak_load = round(entry['simulationObjective_maximumLoad_kW_combined'][0],2)
                            thermal_discomfort = round(entry['simulationObjective_thermalDiscomfort_combined'][0],2)

                            data.append([id_of_the_run, costs, peak_load,thermal_discomfort])

                        df_results_population = pd.DataFrame(data, columns=['id of the run', 'Costs', 'Peak Load','Thermal Discomfort'])
                        a = 2
                        print("")
                        print(f"pareto_df: {pareto_df}")
                    a = 1
                pareto_front = pareto_df
                # Plot the results into a file
                # Print results to csv
                titleOfThePlot = "Local Search - Day: " + str(currentDay) + " - "
                appendixResultFile = "Local Search" + "_Day" + str(currentDay) + ""
                if SetUpScenarios.numberOfBuildings_BT1 > 0:
                    titleOfThePlot = titleOfThePlot + "BT1: " + str(SetUpScenarios.numberOfBuildings_BT1) + ", "
                    appendixResultFile = appendixResultFile + "_BT1_" + str(SetUpScenarios.numberOfBuildings_BT1)
                if SetUpScenarios.numberOfBuildings_BT2 > 0:
                    titleOfThePlot = titleOfThePlot + "BT2: " + str(SetUpScenarios.numberOfBuildings_BT2) + ", "
                    appendixResultFile = appendixResultFile + "_BT2_" + str(SetUpScenarios.numberOfBuildings_BT2)
                if SetUpScenarios.numberOfBuildings_BT3 > 0:
                    titleOfThePlot = titleOfThePlot + "BT3: " + str(SetUpScenarios.numberOfBuildings_BT3) + ", "
                    appendixResultFile = appendixResultFile + "_BT3_" + str(SetUpScenarios.numberOfBuildings_BT3)
                if SetUpScenarios.numberOfBuildings_BT4 > 0:
                    titleOfThePlot = titleOfThePlot + "BT4: " + str(SetUpScenarios.numberOfBuildings_BT4) + ", "
                    appendixResultFile = appendixResultFile + "_BT4_" + str(SetUpScenarios.numberOfBuildings_BT4)
                if SetUpScenarios.numberOfBuildings_BT5 > 0:
                    titleOfThePlot = titleOfThePlot + "BT5: " + str(SetUpScenarios.numberOfBuildings_BT5) + ", "
                    appendixResultFile = appendixResultFile + "_BT5_" + str(SetUpScenarios.numberOfBuildings_BT5)
                if SetUpScenarios.numberOfBuildings_BT6 > 0:
                    titleOfThePlot = titleOfThePlot + "BT6: " + str(SetUpScenarios.numberOfBuildings_BT6) + ", "
                    appendixResultFile = appendixResultFile + "_BT6_" + str(SetUpScenarios.numberOfBuildings_BT6)
                if SetUpScenarios.numberOfBuildings_BT7 > 0:
                    titleOfThePlot = titleOfThePlot + "BT7: " + str(SetUpScenarios.numberOfBuildings_BT7) + ", "
                    appendixResultFile = appendixResultFile + "_BT7_" + str(SetUpScenarios.numberOfBuildings_BT7)
                titleOfThePlot = titleOfThePlot[:-2]


                pareto_front.to_csv(pathForCreatingTheResultData_LocalSearch + "/ParetoFront_" + appendixResultFile + ".csv",index=False, sep=";")


                if not withRL:
                    # Plot Pareto efficient solutions with line
                    import matplotlib.pyplot as plt
                    import matplotlib

                    plt.scatter(pareto_front['Costs'], pareto_front['Peak Load'], color='blue')
                    plt.plot(pareto_front['Costs'], pareto_front['Peak Load'], color='red')
                    plt.xlabel('Costs', fontsize=15)
                    plt.ylabel('Peak Load', fontsize=15)
                    if font_size_title_Pareto_Plot > 0:
                        plt.title(titleOfThePlot, fontsize=font_size_title_Pareto_Plot)
                    plt.tick_params(axis='both', which='major', labelsize=11)
                    plt.subplots_adjust(left=0.15)
                    plt.savefig(pathForCreatingTheResultData_LocalSearch + '/PFront_Line_' + appendixResultFile + '.png', dpi=100)

                    # Clear current figure
                    plt.clf()
                    # Plot Pareto efficient solutions without line
                    plt.scatter(pareto_front['Costs'], pareto_front['Peak Load'], color='blue')
                    plt.xlabel('Costs', fontsize=14)
                    plt.ylabel('Peak Load', fontsize=14)
                    plt.subplots_adjust(left=0.15)
                    if font_size_title_Pareto_Plot > 0:
                        plt.title(titleOfThePlot, fontsize=font_size_title_Pareto_Plot)
                    plt.tick_params(axis='both', which='major', labelsize=11)
                    plt.savefig(pathForCreatingTheResultData_LocalSearch + '/PFront_' + appendixResultFile + '.png', dpi=100)

                #Calculate parto front metrics for comparisons if desired
                if calculate_pareto_front_comparisons == True:
                    #Read pareto front dataframe from file
                    file_path = os.path.join(config.DIR_PARETO_FRONT_FULL,'ParetoFront_' + appendixResultFile + '.csv')
                    file_path_adjusted = file_path.replace("Local Search_", "")
                    pareto_front_full = pd.read_csv(file_path_adjusted, sep=';')
                    pareto_front_approximation = pareto_front

                    #Get the values from the dataframes
                    pareto_front_approximation['Costs'] = pareto_front_approximation['Costs'].apply(lambda x: x[0] if isinstance(x, (list, np.ndarray)) and len(x) > 0 else x)
                    pareto_front_approximation['Peak Load'] = pareto_front_approximation['Peak Load'].apply(lambda x: x[0] if isinstance(x, (list, np.ndarray)) and len(x) > 0 else x)

                    pareto_front_approximation_values = pareto_front_approximation[['Costs', 'Peak Load']].values
                    pareto_front_full_values = pareto_front_full[['Costs', 'Peak Load']].values

                    # Calculate the Generational Distance
                    from pymoo.indicators.gd import GD
                    ind = GD(pareto_front_full_values)
                    print("")
                    generational_distance_appximated_front = round(ind(pareto_front_approximation_values), 1)
                    print("Generational Distance PF_Approximation", generational_distance_appximated_front)

                    ref_point = np.array([help_value_normalization_cost_conventional, help_value_normalization_maxiumLoad_conventional])
                    reshaped_ref_point = np.reshape(ref_point, (1, 2))
                    generational_distance_ref_point = round(ind(reshaped_ref_point), 1)


                    # Calculate the Hypervolume
                    from pymoo.indicators.hv import HV
                    ref_point = np.array([help_value_normalization_cost_conventional, help_value_normalization_maxiumLoad_conventional ])
                    ind = HV(ref_point=ref_point)
                    result_ind = ind(pareto_front_approximation_values)
                    if isinstance(result_ind, (list, np.ndarray)):
                        # If it's a list or NumPy array, use the first element if available
                        if len(result_ind) > 0:
                            hypervolume_approximated_front = round(result_ind[0], 1)
                        else:
                            print("Empty list or array")
                    else:
                        # Handle other cases, e.g., when ind() returns a float
                        hypervolume_approximated_front = round(result_ind, 1)
                    hypervolume_full_front = round(ind(pareto_front_full_values)[0], 1)
                    hypervolume_ratio_from_approximation_of_full_pareto_front = round((hypervolume_approximated_front / hypervolume_full_front) *100, 1)
                    print("")
                    print("Hypervolume PF_Approximation", hypervolume_approximated_front)
                    print("Hypervolume PF_Full", hypervolume_full_front)
                    print("Hypervolume Percentage PF_Approximation ", hypervolume_ratio_from_approximation_of_full_pareto_front)


                    #Add results of multiobt into csv file
                    new_row = {'Day': currentDay, 'GD_Conventional': generational_distance_ref_point,  'GD PF_Approx': generational_distance_appximated_front,'HV PF_Approx': hypervolume_approximated_front,'HV PF_Full': hypervolume_full_front,'HV_Ratio': hypervolume_ratio_from_approximation_of_full_pareto_front}
                    df_results_multiopt_local_search = pd.concat([df_results_multiopt_local_search, pd.DataFrame(new_row, index=[0])], ignore_index=True)



                    #Plot the combined pareto front
                    import matplotlib.pyplot as plt
                    import matplotlib

                    plt.clf()
                    plt.scatter(pareto_front['Costs'], pareto_front['Peak Load'], color='blue', label='Pareto Front PLS')
                    plt.scatter(pareto_front_full['Costs'], pareto_front_full['Peak Load'], color='palegreen', label='Pareto Front True')

                    ref_point_reshaped = np.reshape(ref_point, (1, 2))
                    plt.scatter(ref_point_reshaped[:, 0], ref_point_reshaped[:, 1], color='red',label='Conventional Control')

                    plt.xlabel('Costs', fontsize=14)
                    plt.ylabel('Peak Load', fontsize=14)
                    plt.subplots_adjust(left=0.15)
                    if font_size_title_Pareto_Plot > 0:
                        plt.title(titleOfThePlot, fontsize=font_size_title_Pareto_Plot)
                    plt.tick_params(axis='both', which='major', labelsize=11)

                    # Add a legend to distinguish between points
                    plt.legend(loc='upper right')


                    # Save the combined Pareto diagram
                    plt.savefig(pathForCreatingTheResultData_LocalSearch + '/PFrontCombined_' + appendixResultFile + '.png', dpi=100)


                end_time = time.time()
                execution_time = end_time - start_time
                end_cpu = time.process_time()
                execution_cpu = end_cpu - start_cpu
                print("")
                hours, remainder = divmod(execution_time, 3600)
                minutes, seconds = divmod(remainder, 60)
                formatted_time = "{:02d} hours, {:02d} minutes, {:02d} seconds".format(int(hours), int(minutes), int(seconds))
                print("Normal Execution Time:", formatted_time)

                hours, remainder = divmod(execution_cpu, 3600)
                minutes, seconds = divmod(remainder, 60)
                formatted_time = "{:02d} hours, {:02d} minutes, {:02d} seconds".format(int(hours), int(minutes), int(seconds))
                print("CPU  Time:", formatted_time)

                #Print end profiles of the pareto optimal solutions
                for i in range(0, len(pareto_front)):
                    indexOfBuildingsOverall_BT1 = [i for i in range(1, SetUpScenarios.numberOfBuildings_BT1 + 1)]
                    indexOfBuildingsOverall_BT2 = [i for i in range(1, SetUpScenarios.numberOfBuildings_BT2 + 1)]
                    indexOfBuildingsOverall_BT3 = [i for i in range(1, SetUpScenarios.numberOfBuildings_BT3 + 1)]
                    indexOfBuildingsOverall_BT4 = [i for i in range(1, SetUpScenarios.numberOfBuildings_BT4 + 1)]
                    indexOfBuildingsOverall_BT5 = [i for i in range(1, SetUpScenarios.numberOfBuildings_BT5 + 1)]
                    indexOfBuildingsOverall_BT6 = [i for i in range(1, SetUpScenarios.numberOfBuildings_BT6 + 1)]
                    indexOfBuildingsOverall_BT7 = [i for i in range(1, SetUpScenarios.numberOfBuildings_BT7 + 1)]

                    useLocalSearch = False

                    desired_id_of_the_run = pareto_front.loc[i, "id of the run"]

                    # Iterate over the list and find the matching dictionary entry
                    for entry in results_list_population:
                        if entry["id of the run"] == desired_id_of_the_run:
                            outputVector_heatGenerationCoefficientSpaceHeating_BT1 = entry["outputVector_BT1_heatGenerationCoefficientSpaceHeating"]
                            outputVector_heatGenerationCoefficientDHW_BT1 = entry["outputVector_BT1_heatGenerationCoefficientDHW"]
                            outputVector_chargingPowerEV_BT1 = entry["outputVector_BT1_chargingPowerEV"]
                            outputVector_heatGenerationCoefficientSpaceHeating_BT2 = entry["outputVector_BT2_heatGenerationCoefficientSpaceHeating"]
                            outputVector_heatGenerationCoefficientDHW_BT2 = entry["outputVector_BT2_heatGenerationCoefficientDHW"]
                            outputVector_chargingPowerEV_BT3 = entry["outputVector_BT3_chargingPowerEV"]
                            outputVector_heatGenerationCoefficientSpaceHeating_BT4 = entry["outputVector_BT4_heatGenerationCoefficientSpaceHeating"]
                            outputVector_chargingPowerBAT_BT5 = entry["outputVector_BT5_chargingPowerBAT"]
                            outputVector_disChargingPowerBAT_BT5 = entry["outputVector_BT5_disChargingPowerBAT"]
                            outputVector_heatGenerationCoefficient_GasBoiler_BT6 = entry["outputVector_BT6_heatGenerationCoefficient_GasBoiler"]
                            outputVector_heatGenerationCoefficient_ElectricalHeatingElement_BT6 = entry["outputVector_BT6_heatGenerationCoefficient_ElectricalHeatingElement"]
                            outputVector_heatTransferCoefficient_StorageToRoom_BT6 = entry["outputVector_BT6_heatTransferCoefficient_StorageToRoom"]
                            outputVector_heatGenerationCoefficient_GasBoiler_BT7 = entry["outputVector_BT7_heatGenerationCoefficient_GasBoiler"]
                            outputVector_electricalPowerFanHeater_BT7 = entry["outputVector_BT7_electricalPowerFanHeater"]
                            break


                    if i == 0:
                        # create "Individual Solutions" folder
                        folder_path_individual_solutions = os.path.join(pathForCreatingTheResultData_LocalSearch, "Individual Solutions")
                        try:
                            os.makedirs(folder_path_individual_solutions)
                        except OSError:
                            print(f"Creation of the directory {folder_path_individual_solutions} failed")

                    # create "Solution_" subfolders inside "Individual Solutions" folder
                    subfolder_name = "Solution_" + str(i + 1)
                    folder_path_single_result = os.path.join(folder_path_individual_solutions, subfolder_name)
                    try:
                        os.makedirs(folder_path_single_result)
                    except OSError:
                        print(f"Creation of the directory {folder_path_single_result} failed")

                    simulationObjective_surplusEnergy_kWh_combined, simulationObjective_maximumLoad_kW_combined, simulationObjective_costs_Euro_combined, simulationObjective_thermalDiscomfort_combined, simulationObjective_gasConsumptionkWh_combined, simulationObjective_combinedScore_combined = ICSimulation.simulateDays_WithLightController_Schedule(
                        indexOfBuildingsOverall_BT1, indexOfBuildingsOverall_BT2, indexOfBuildingsOverall_BT3,
                        indexOfBuildingsOverall_BT4, indexOfBuildingsOverall_BT5, indexOfBuildingsOverall_BT6,
                        indexOfBuildingsOverall_BT7, currentDay, outputVector_heatGenerationCoefficientSpaceHeating_BT1,
                        outputVector_heatGenerationCoefficientDHW_BT1, outputVector_chargingPowerEV_BT1,
                        outputVector_heatGenerationCoefficientSpaceHeating_BT2,
                        outputVector_heatGenerationCoefficientDHW_BT2, outputVector_chargingPowerEV_BT3,
                        outputVector_heatGenerationCoefficientSpaceHeating_BT4, outputVector_chargingPowerBAT_BT5,
                        outputVector_disChargingPowerBAT_BT5, outputVector_heatGenerationCoefficient_GasBoiler_BT6,
                        outputVector_heatGenerationCoefficient_ElectricalHeatingElement_BT6,
                        outputVector_heatTransferCoefficient_StorageToRoom_BT6,
                        outputVector_heatGenerationCoefficient_GasBoiler_BT7, outputVector_electricalPowerFanHeater_BT7,
                        folder_path_single_result, preCorrectSchedules_AvoidingFrequentStarts, optParameters,
                        useLocalSearch)
                    useLocalSearch = True
            if not withRL:
                # Exectue the dichotomous method for the goals: Costs and Peak
                if useDichotomicMethodCentralized_Cost_Peak == True:
                    id_of_the_run = 0
                    df_results = pd.DataFrame(columns=["id of the run", "Costs", "Peak Load", "Weight Costs", "Weight Peak Load", "MIP Gap", "Solving time"])
                    df_results_after_internal_controller = pd.DataFrame(columns=["id of the run", "Costs", "Peak Load","Thermal Discomfort"])

                    df_activePoints = pd.DataFrame(columns=["id", "Costs", "Peak Load"])
                    list_activePoints = []
                    list_usedWeightes_minimizeCosts = []

                    #Optimize the first goal: Costs
                    optParameters['optimizationGoal_minimizePeakLoad'] = True
                    optParameters['optimizationGoal_minimizeCosts'] = True
                    optParameters['optimizationGoal_minimizeGas'] = False
                    optParameters['optimizationGoal_minimizeThermalDiscomfort'] = False
                    optParameters['optimization_1Objective'] = False
                    optParameters['optimization_2Objective'] = True
                    optParameters['optimization_3Objectives'] = False
                    optParameters['optimization_4Objectives'] = False
                    optParameters['objective_minimizeCosts_weight'] = 0.99
                    optParameters['objective_minimizePeakLoad_weight'] = 1 - optParameters['objective_minimizeCosts_weight']
                    list_usedWeightes_minimizeCosts.append(optParameters['objective_minimizeCosts_weight'])

                    includeObjectivesInReturnStatementCentralized = True
                    outputVector_heatGenerationCoefficientSpaceHeating_BT1, outputVector_heatGenerationCoefficientDHW_BT1, outputVector_chargingPowerEV_BT1, outputVector_heatGenerationCoefficientSpaceHeating_BT2, outputVector_heatGenerationCoefficientDHW_BT2, outputVector_chargingPowerEV_BT3, outputVector_heatGenerationCoefficientSpaceHeating_BT4, outputVector_chargingPowerBAT_BT5, outputVector_disChargingPowerBAT_BT5, outputVector_heatGenerationCoefficient_GasBoiler_BT6, outputVector_heatGenerationCoefficient_ElectricalHeatingElement_BT6, outputVector_heatTransferCoefficient_StorageToRoom_BT6, outputVector_heatGenerationCoefficient_GasBoiler_BT7, outputVector_electricalPowerFanHeater_BT7,  objectiveMaximumLoad_OP1, objectiveSurplusEnergy_OP1, objectiveCosts_OP1, objectiveThermalDiscomfort_OP1, mipGapOfFoundSolution, timeForFindingOptimalSolution =   Building_Optimization_Problem.optimizeOneDay(indexOfBuildingsOverall_BT1, indexOfBuildingsOverall_BT2, indexOfBuildingsOverall_BT3, indexOfBuildingsOverall_BT4, indexOfBuildingsOverall_BT5,indexOfBuildingsOverall_BT6, indexOfBuildingsOverall_BT7, currentDay, includeObjectivesInReturnStatementCentralized, optParameters)
                    id_of_the_run += 1
                    df_results.loc[id_of_the_run - 1] = [id_of_the_run, objectiveCosts_OP1, objectiveMaximumLoad_OP1, optParameters['objective_minimizeCosts_weight'], optParameters['objective_minimizePeakLoad_weight'], mipGapOfFoundSolution, timeForFindingOptimalSolution]
                    list_activePoints.append({"id": id_of_the_run, "Costs": objectiveCosts_OP1, "Maximum_Load": objectiveMaximumLoad_OP1})

                    if create_result_load_profiles_multi_opt == True:
                        preCorrectSchedules_AvoidingFrequentStarts = False
                        overruleActions = False
                        pathForCreatingTheResultData = pathForCreatingTheResultData_Dichotromic + "/Profiles/Solution_ID" + str(id_of_the_run)
                        useLocalSearchHelp = False
                        simulationObjective_surplusEnergy_kWh_combined, simulationObjective_maximumLoad_kW_combined, simulationObjective_thermalDiscomfort_combined, simulationObjective_gasConsumptionkWh_combined, simulationObjective_costs_Euro_combined, simulationObjective_combinedScore_combined = ICSimulation.simulateDays_WithLightController_Schedule(indexOfBuildingsOverall_BT1, indexOfBuildingsOverall_BT2, indexOfBuildingsOverall_BT3, indexOfBuildingsOverall_BT4, indexOfBuildingsOverall_BT5, indexOfBuildingsOverall_BT6, indexOfBuildingsOverall_BT7, currentDay, outputVector_heatGenerationCoefficientSpaceHeating_BT1, outputVector_heatGenerationCoefficientDHW_BT1, outputVector_chargingPowerEV_BT1, outputVector_heatGenerationCoefficientSpaceHeating_BT2, outputVector_heatGenerationCoefficientDHW_BT2, outputVector_chargingPowerEV_BT3, outputVector_heatGenerationCoefficientSpaceHeating_BT4, outputVector_chargingPowerBAT_BT5, outputVector_disChargingPowerBAT_BT5, outputVector_heatGenerationCoefficient_GasBoiler_BT6, outputVector_heatGenerationCoefficient_ElectricalHeatingElement_BT6, outputVector_heatTransferCoefficient_StorageToRoom_BT6, outputVector_heatGenerationCoefficient_GasBoiler_BT7, outputVector_electricalPowerFanHeater_BT7, pathForCreatingTheResultData, preCorrectSchedules_AvoidingFrequentStarts, optParameters, useLocalSearchHelp)
                        df_results_after_internal_controller.loc[id_of_the_run - 1] = [id_of_the_run, simulationObjective_costs_Euro_combined, simulationObjective_maximumLoad_kW_combined, simulationObjective_thermalDiscomfort_combined]


                    # Optimize the second goal: Peak load
                    optParameters['objective_minimizeCosts_weight'] = 0.01
                    optParameters['objective_minimizePeakLoad_weight'] = 1 - optParameters['objective_minimizeCosts_weight']
                    list_usedWeightes_minimizeCosts.append(optParameters['objective_minimizeCosts_weight'])


                    outputVector_heatGenerationCoefficientSpaceHeating_BT1, outputVector_heatGenerationCoefficientDHW_BT1, outputVector_chargingPowerEV_BT1, outputVector_heatGenerationCoefficientSpaceHeating_BT2, outputVector_heatGenerationCoefficientDHW_BT2, outputVector_chargingPowerEV_BT3, outputVector_heatGenerationCoefficientSpaceHeating_BT4, outputVector_chargingPowerBAT_BT5, outputVector_disChargingPowerBAT_BT5, outputVector_heatGenerationCoefficient_GasBoiler_BT6, outputVector_heatGenerationCoefficient_ElectricalHeatingElement_BT6, outputVector_heatTransferCoefficient_StorageToRoom_BT6, outputVector_heatGenerationCoefficient_GasBoiler_BT7, outputVector_electricalPowerFanHeater_BT7,  objectiveMaximumLoad_OP2, objectiveSurplusEnergy_OP2, objectiveCosts_OP2,objectiveThermalDiscomfort_OP2, mipGapOfFoundSolution, timeForFindingOptimalSolution =   Building_Optimization_Problem.optimizeOneDay(indexOfBuildingsOverall_BT1, indexOfBuildingsOverall_BT2, indexOfBuildingsOverall_BT3, indexOfBuildingsOverall_BT4, indexOfBuildingsOverall_BT5,indexOfBuildingsOverall_BT6, indexOfBuildingsOverall_BT7, currentDay, includeObjectivesInReturnStatementCentralized, optParameters)


                    id_of_the_run += 1
                    df_results.loc[id_of_the_run - 1] = [id_of_the_run, objectiveCosts_OP2, objectiveMaximumLoad_OP2, optParameters['objective_minimizeCosts_weight'], optParameters['objective_minimizePeakLoad_weight'], mipGapOfFoundSolution, timeForFindingOptimalSolution]
                    list_activePoints.append({"id": id_of_the_run, "Costs": objectiveCosts_OP2, "Maximum_Load": objectiveMaximumLoad_OP2})

                    if create_result_load_profiles_multi_opt == True:
                        preCorrectSchedules_AvoidingFrequentStarts = False
                        overruleActions = False
                        pathForCreatingTheResultData = pathForCreatingTheResultData_Dichotromic + "/Profiles/Solution_ID" + str(id_of_the_run)
                        useLocalSearchHelp = False
                        simulationObjective_surplusEnergy_kWh_combined, simulationObjective_maximumLoad_kW_combined, simulationObjective_thermalDiscomfort_combined, simulationObjective_gasConsumptionkWh_combined, simulationObjective_costs_Euro_combined, simulationObjective_combinedScore_combined= ICSimulation.simulateDays_WithLightController_Schedule(indexOfBuildingsOverall_BT1, indexOfBuildingsOverall_BT2, indexOfBuildingsOverall_BT3, indexOfBuildingsOverall_BT4, indexOfBuildingsOverall_BT5, indexOfBuildingsOverall_BT6, indexOfBuildingsOverall_BT7, currentDay, outputVector_heatGenerationCoefficientSpaceHeating_BT1, outputVector_heatGenerationCoefficientDHW_BT1, outputVector_chargingPowerEV_BT1, outputVector_heatGenerationCoefficientSpaceHeating_BT2, outputVector_heatGenerationCoefficientDHW_BT2, outputVector_chargingPowerEV_BT3, outputVector_heatGenerationCoefficientSpaceHeating_BT4, outputVector_chargingPowerBAT_BT5, outputVector_disChargingPowerBAT_BT5, outputVector_heatGenerationCoefficient_GasBoiler_BT6, outputVector_heatGenerationCoefficient_ElectricalHeatingElement_BT6, outputVector_heatTransferCoefficient_StorageToRoom_BT6, outputVector_heatGenerationCoefficient_GasBoiler_BT7, outputVector_electricalPowerFanHeater_BT7, pathForCreatingTheResultData, preCorrectSchedules_AvoidingFrequentStarts, optParameters, useLocalSearchHelp)
                        df_results_after_internal_controller.loc[id_of_the_run - 1] = [id_of_the_run, simulationObjective_costs_Euro_combined, simulationObjective_maximumLoad_kW_combined, simulationObjective_thermalDiscomfort_combined]


                    #Do the dichotomic iteration
                    terminationConditionReached = False
                    iterationCounter = 0
                    helpCounterWeightAlreadyUsed = 0
                    if scaleResultsWeightedSum ==True:
                        optParameters['objective_minimizeCosts_normalizationValue'] =  objectiveCosts_OP1
                        optParameters['objective_minimizePeakLoad_normalizationValue'] = objectiveMaximumLoad_OP2


                    while terminationConditionReached == False:

                        #Write list of active points to csv (if desired for testing)
                        if dichotomicMethodprintListOfActivePoints == True:
                            import csv
                            file_path = pathForCreatingTheResultData_Dichotromic + '/activePoints_It_' + str(iterationCounter) + '.csv'

                            try:
                                with open(file_path, 'w', newline='') as file:
                                    writer = csv.writer(file, delimiter=';', quoting=csv.QUOTE_NONE)
                                    writer.writerow(['id', 'Costs', 'Maximum_Load'])
                                    for row in list_activePoints:
                                        writer.writerow([row['id'], round(row['Costs'], 0), round(row['Maximum_Load'], 0)])
                                print("Data successfully written to CSV file.")
                            except Exception as e:
                                print(f"Error writing data to CSV file: {e}")

                        if len(list_activePoints) <2 or iterationCounter>dichotomicMethodTermination_NumberOfIterations:
                            terminationConditionReached = True
                            break

                        #Create new optimization problem
                        iterationCounter += 1

                        optParameters['optimizationGoal_minimizePeakLoad'] = True
                        optParameters['optimizationGoal_minimizeCosts'] = True
                        try:
                            optParameters['objective_minimizeCosts_weight'] = (list_activePoints[0]["Maximum_Load"] - list_activePoints[1]["Maximum_Load"])/((list_activePoints[1]["Costs"] - list_activePoints[0]["Costs"] )  + (list_activePoints[0]["Maximum_Load"] - list_activePoints[1]["Maximum_Load"]))
                        except:
                            optParameters['objective_minimizeCosts_weight'] = 1

                        # Loop through the list and check if the value is within the range of dichotomicMethod_toleranceLambdaNewSolution
                        weightAlreadyUsed = False
                        for num in list_usedWeightes_minimizeCosts:
                            if abs(num - optParameters['objective_minimizeCosts_weight']) < dichotomicMethod_toleranceLambdaNewSolution:
                                weightAlreadyUsed = True
                                helpCounterWeightAlreadyUsed += 1
                                break


                        if weightAlreadyUsed ==True:
                            #Remove entry at the first position of active points
                            list_activePoints.pop(0)
                            continue
                        else:
                            list_usedWeightes_minimizeCosts.append(optParameters['objective_minimizeCosts_weight'])


                        if optParameters['objective_minimizeCosts_weight'] > 1:
                            optParameters['objective_minimizeCosts_weight'] = 1
                        if optParameters['objective_minimizeCosts_weight'] < 0:
                            optParameters['objective_minimizeCosts_weight'] = 0
                        optParameters['objective_minimizePeakLoad_weight'] = 1 - optParameters['objective_minimizeCosts_weight']

                        try:
                            outputVector_heatGenerationCoefficientSpaceHeating_BT1, outputVector_heatGenerationCoefficientDHW_BT1, outputVector_chargingPowerEV_BT1, outputVector_heatGenerationCoefficientSpaceHeating_BT2, outputVector_heatGenerationCoefficientDHW_BT2, outputVector_chargingPowerEV_BT3, outputVector_heatGenerationCoefficientSpaceHeating_BT4, outputVector_chargingPowerBAT_BT5, outputVector_disChargingPowerBAT_BT5, outputVector_heatGenerationCoefficient_GasBoiler_BT6, outputVector_heatGenerationCoefficient_ElectricalHeatingElement_BT6, outputVector_heatTransferCoefficient_StorageToRoom_BT6, outputVector_heatGenerationCoefficient_GasBoiler_BT7, outputVector_electricalPowerFanHeater_BT7,  objectiveMaximumLoad_OPStar, objectiveSurplusEnergy_OPStar, objectiveCosts_OPStar, objectiveThermalDiscomfort_OPStar, mipGapOfFoundSolution, timeForFindingOptimalSolution =   Building_Optimization_Problem.optimizeOneDay(indexOfBuildingsOverall_BT1, indexOfBuildingsOverall_BT2, indexOfBuildingsOverall_BT3, indexOfBuildingsOverall_BT4, indexOfBuildingsOverall_BT5,indexOfBuildingsOverall_BT6, indexOfBuildingsOverall_BT7, currentDay, includeObjectivesInReturnStatementCentralized, optParameters)
                        except:
                            break
                        id_of_the_run += 1
                        df_results.loc[id_of_the_run - 1] = [id_of_the_run, objectiveCosts_OPStar, objectiveMaximumLoad_OPStar, optParameters['objective_minimizeCosts_weight'], optParameters['objective_minimizePeakLoad_weight'], mipGapOfFoundSolution, timeForFindingOptimalSolution]

                        if create_result_load_profiles_multi_opt == True:
                            preCorrectSchedules_AvoidingFrequentStarts = False
                            overruleActions = False
                            pathForCreatingTheResultData = pathForCreatingTheResultData_Dichotromic + "/Profiles/Solution_ID" + str(id_of_the_run)
                            useLocalSearchHelp = False
                            simulationObjective_surplusEnergy_kWh_combined, simulationObjective_maximumLoad_kW_combined, simulationObjective_thermalDiscomfort_combined, simulationObjective_gasConsumptionkWh_combined, simulationObjective_costs_Euro_combined, simulationObjective_combinedScore_combined = ICSimulation.simulateDays_WithLightController_Schedule(indexOfBuildingsOverall_BT1, indexOfBuildingsOverall_BT2, indexOfBuildingsOverall_BT3, indexOfBuildingsOverall_BT4, indexOfBuildingsOverall_BT5, indexOfBuildingsOverall_BT6, indexOfBuildingsOverall_BT7, currentDay, outputVector_heatGenerationCoefficientSpaceHeating_BT1, outputVector_heatGenerationCoefficientDHW_BT1, outputVector_chargingPowerEV_BT1, outputVector_heatGenerationCoefficientSpaceHeating_BT2, outputVector_heatGenerationCoefficientDHW_BT2, outputVector_chargingPowerEV_BT3, outputVector_heatGenerationCoefficientSpaceHeating_BT4, outputVector_chargingPowerBAT_BT5, outputVector_disChargingPowerBAT_BT5, outputVector_heatGenerationCoefficient_GasBoiler_BT6, outputVector_heatGenerationCoefficient_ElectricalHeatingElement_BT6, outputVector_heatTransferCoefficient_StorageToRoom_BT6, outputVector_heatGenerationCoefficient_GasBoiler_BT7, outputVector_electricalPowerFanHeater_BT7, pathForCreatingTheResultData, preCorrectSchedules_AvoidingFrequentStarts, optParameters, useLocalSearchHelp)
                            df_results_after_internal_controller.loc[id_of_the_run - 1] = [id_of_the_run, simulationObjective_costs_Euro_combined, simulationObjective_maximumLoad_kW_combined, simulationObjective_thermalDiscomfort_combined]


                        # Print results to csv
                        df_results['Costs'] = df_results['Costs'].round(0)
                        df_results['Peak Load'] = df_results['Peak Load'].round(0)
                        df_results['Solving time'] = df_results['Solving time'].round(0)
                        df_results['Weight Costs'] = df_results['Weight Costs'].round(4)
                        df_results['Weight Peak Load'] = df_results['Weight Peak Load'].round(4)
                        df_results.to_csv(pathForCreatingTheResultData_Dichotromic + "/DT_Peak_Costs"+ ".csv", index=False, sep=";")



                        #Check stopping criterion
                        firstTermLeft =  optParameters['objective_minimizeCosts_weight'] * objectiveCosts_OPStar
                        secondTermLeft = (1-optParameters['objective_minimizeCosts_weight']) * objectiveMaximumLoad_OPStar
                        wholeTermLEft = optParameters['objective_minimizeCosts_weight'] * objectiveCosts_OPStar  + (1-optParameters['objective_minimizeCosts_weight']) * objectiveMaximumLoad_OPStar
                        firstTermRight =optParameters['objective_minimizeCosts_weight'] *  list_activePoints[0]["Costs"]
                        secondTermRight = (1- optParameters['objective_minimizeCosts_weight'] ) *  list_activePoints[0]["Maximum_Load"]
                        wholeTermRight =optParameters['objective_minimizeCosts_weight'] *  list_activePoints[0]["Costs"]  + (1- optParameters['objective_minimizeCosts_weight'] ) *  list_activePoints[0]["Maximum_Load"]

                        if optParameters['objective_minimizeCosts_weight'] * objectiveCosts_OPStar  + (1-optParameters['objective_minimizeCosts_weight']) * objectiveMaximumLoad_OPStar < optParameters['objective_minimizeCosts_weight'] *  list_activePoints[0]["Costs"]  + (1- optParameters['objective_minimizeCosts_weight'] ) *  list_activePoints[0]["Maximum_Load"]:

                            #Add new active point as the second entry
                            list_activePoints.insert(1, {"id": id_of_the_run, "Costs": objectiveCosts_OPStar, "Maximum_Load": objectiveMaximumLoad_OPStar})

                        else:
                            #Remove entry at the first position of active points
                            list_activePoints.pop(0)

                        # Print results to csv
                        titleOfThePlot = "Dichotromic Method - Day: " + str(currentDay) + " - "
                        appendixResultFile = "DC_Day" + str(currentDay) + ""
                        if SetUpScenarios.numberOfBuildings_BT1 > 0:
                            titleOfThePlot = titleOfThePlot + "BT1: " + str(SetUpScenarios.numberOfBuildings_BT1) + ", "
                            appendixResultFile = appendixResultFile + "_BT1_" + str(SetUpScenarios.numberOfBuildings_BT1)
                        if SetUpScenarios.numberOfBuildings_BT2 > 0:
                            titleOfThePlot = titleOfThePlot + "BT2: " + str(SetUpScenarios.numberOfBuildings_BT2) + ", "
                            appendixResultFile = appendixResultFile + "_BT2_" + str(SetUpScenarios.numberOfBuildings_BT2)
                        if SetUpScenarios.numberOfBuildings_BT3 > 0:
                            titleOfThePlot = titleOfThePlot + "BT3: " + str(SetUpScenarios.numberOfBuildings_BT3) + ", "
                            appendixResultFile = appendixResultFile + "_BT3_" + str(SetUpScenarios.numberOfBuildings_BT3)
                        if SetUpScenarios.numberOfBuildings_BT4 > 0:
                            titleOfThePlot = titleOfThePlot + "BT4: " + str(SetUpScenarios.numberOfBuildings_BT4) + ", "
                            appendixResultFile = appendixResultFile + "_BT4_" + str(SetUpScenarios.numberOfBuildings_BT4)
                        if SetUpScenarios.numberOfBuildings_BT5 > 0:
                            titleOfThePlot = titleOfThePlot + "BT5: " + str(SetUpScenarios.numberOfBuildings_BT5) + ", "
                            appendixResultFile = appendixResultFile + "_BT5_" + str(SetUpScenarios.numberOfBuildings_BT5)
                        if SetUpScenarios.numberOfBuildings_BT6 > 0:
                            titleOfThePlot = titleOfThePlot + "BT6: " + str(SetUpScenarios.numberOfBuildings_BT6) + ", "
                            appendixResultFile = appendixResultFile + "_BT6_" + str(SetUpScenarios.numberOfBuildings_BT6)
                        if SetUpScenarios.numberOfBuildings_BT7 > 0:
                            titleOfThePlot = titleOfThePlot + "BT7: " + str(SetUpScenarios.numberOfBuildings_BT7) + ", "
                            appendixResultFile = appendixResultFile + "_BT7_" + str(SetUpScenarios.numberOfBuildings_BT7)
                        titleOfThePlot = titleOfThePlot [:-2]

                        df_results['Costs'] = df_results['Costs'].round(0)
                        df_results['Peak Load'] = df_results['Peak Load'].round(0)
                        df_results['Solving time'] = df_results['Solving time'].round(0)
                        df_results['Weight Costs'] = df_results['Weight Costs'].round(4)
                        df_results['Weight Peak Load'] = df_results['Weight Peak Load'].round(4)
                        df_results.to_csv(pathForCreatingTheResultData_Dichotromic + "/Peak_Costs"+ appendixResultFile + ".csv", index=False, sep=";")



                    print(f"\niterationCounter DichotomicMethod: {iterationCounter}")


                    # Create figure with pareto front
                    df_results['Costs'] = df_results['Costs'] / 100
                    df_results['Costs'] = df_results['Costs'].round(2)
                    df_results['Peak Load'] = df_results['Peak Load'] / 1000
                    df_results['Peak Load'] = df_results['Peak Load'].round(2)
                    df_results['Solving time'] = df_results['Solving time'].round(0)
                    df_results['Weight Costs'] = df_results['Weight Costs'].round(4)
                    df_results['Weight Peak Load'] = df_results['Weight Peak Load'].round(4)
                    #df_results.to_csv(pathForCreatingTheResultData_Dichotromic + "/Peak_Costs" + appendixResultFile + ".csv", index=False, sep=";")


                    #Adjust the values of the result_v2 dataframes which contains the values after the simulation with the internal controller
                    columns_to_extract = ['Peak Load', 'Costs', 'Thermal Discomfort']
                    # Use the apply method to extract the first element from each list or array
                    for column in columns_to_extract:
                        df_results_after_internal_controller[column] = df_results_after_internal_controller[column].apply(lambda x: x[0])

                    df_results_after_internal_controller.to_csv(pathForCreatingTheResultData_Dichotromic + "/Peak_Costs" + appendixResultFile + ".csv", index=False, sep=";")

                   #print pareto front
                    import pandas as pd
                    import matplotlib.pyplot as plt
                    import matplotlib

                    matplotlib.use('Agg')

                    if df_results_after_internal_controller.empty:
                        print("Error: df_results_after_internal_controller DataFrame is empty.")
                    else:
                        # Create an empty DataFrame to store the Pareto-efficient solutions
                        pareto_front = pd.DataFrame(columns=['Costs', 'Peak Load'])

                        # Loop through all solutions in the DataFrame
                        for i, row in df_results_after_internal_controller.iterrows():
                            # Assume the current solution is Pareto-efficient until proven otherwise
                            is_efficient = True
                            # Loop through all existing solutions in the Pareto front
                            for j, pareto_row in pareto_front.iterrows():
                                # Check if the current solution is dominated by any existing solution in the Pareto front
                                if (row['Costs'] >= pareto_row['Costs'] and row['Peak Load'] >= pareto_row['Peak Load']):
                                    is_efficient = False
                                    break
                                # Check if the current solution dominates any existing solution in the Pareto front
                                elif (row['Costs'] <= pareto_row['Costs'] and row['Peak Load'] <= pareto_row['Peak Load']):
                                    pareto_front = pareto_front.drop(j)
                            # If the current solution is Pareto-efficient, add it to the Pareto front
                            if is_efficient and row['Thermal Discomfort'] <= threshold_discomfort_local_search:
                                pareto_front = pd.concat([pareto_front, row[['Costs', 'Peak Load']].to_frame().T], ignore_index=True)

                        if pareto_front.empty:
                            print("Error: pareto_front DataFrame is empty.")
                        else:
                            # Sort the Pareto front by 'Costs' and 'Peak Load'
                            pareto_front = pareto_front.sort_values(['Costs', 'Peak Load'], ascending=[True, True])
                            # Reset the index of the Pareto front DataFrame
                            pareto_front = pareto_front.reset_index(drop=True)

                            # Plot Pareto efficient solutions with line
                            plt.scatter(pareto_front['Costs'], pareto_front['Peak Load'], color='blue')
                            plt.plot(pareto_front['Costs'], pareto_front['Peak Load'], color='red')
                            plt.xlabel('Costs [€]', fontsize=15)
                            plt.ylabel('Peak Load [kW]', fontsize=15)
                            if font_size_title_Pareto_Plot > 0:
                                plt.title(titleOfThePlot, fontsize=font_size_title_Pareto_Plot)
                            plt.tick_params(axis='both', which='major', labelsize=11)
                            plt.subplots_adjust(left=0.15)
                            plt.savefig(pathForCreatingTheResultData_Dichotromic + '/PFront_Line_' + appendixResultFile + '.png',dpi=100)
                            # Clear current figure
                            plt.clf()
                            # Plot Pareto efficient solutions without line
                            plt.scatter(pareto_front['Costs'], pareto_front['Peak Load'], color='blue')
                            plt.xlabel('Costs [€]', fontsize=14)
                            plt.ylabel('Peak Load [kW]', fontsize=14)
                            plt.subplots_adjust(left=0.15)
                            if font_size_title_Pareto_Plot > 0:
                                plt.title(titleOfThePlot, fontsize=font_size_title_Pareto_Plot)
                            plt.tick_params(axis='both', which='major', labelsize=11)
                            plt.savefig(pathForCreatingTheResultData_Dichotromic + '/PFront_' + appendixResultFile + '.png',dpi=100)

                # Exectue the box method for the goals: Costs and Peak
                elif useBoxMethodCentralized_Cost_Peak == True:
                    id_of_the_run = 0
                    df_results = pd.DataFrame(columns=["id of the run", "Costs", "Peak Load","epsilon_MaximumLoad_TargetValue", "MIP Gap", "Solving time"])

                    df_results_after_internal_controller = pd.DataFrame(columns=["id of the run", "Costs", "Peak Load","Thermal Discomfort"])


                    #Optimize the first goal: Costs
                    optParameters['optimizationGoal_minimizePeakLoad'] = True
                    optParameters['optimizationGoal_minimizeCosts'] = True
                    optParameters['optimizationGoal_minimizeGas'] = False
                    optParameters['optimizationGoal_minimizeThermalDiscomfort'] = False
                    optParameters['optimization_1Objective'] = False
                    optParameters['optimization_2Objective'] = True
                    optParameters['optimization_3Objectives'] = False
                    optParameters['optimization_4Objectives'] = False
                    optParameters['objective_minimizeCosts_weight'] = 0.99
                    optParameters['objective_minimizePeakLoad_weight'] = 1 - optParameters['objective_minimizeCosts_weight']

                    includeObjectivesInReturnStatementCentralized = True
                    outputVector_heatGenerationCoefficientSpaceHeating_BT1, outputVector_heatGenerationCoefficientDHW_BT1, outputVector_chargingPowerEV_BT1, outputVector_heatGenerationCoefficientSpaceHeating_BT2, outputVector_heatGenerationCoefficientDHW_BT2, outputVector_chargingPowerEV_BT3, outputVector_heatGenerationCoefficientSpaceHeating_BT4, outputVector_chargingPowerBAT_BT5, outputVector_disChargingPowerBAT_BT5, outputVector_heatGenerationCoefficient_GasBoiler_BT6, outputVector_heatGenerationCoefficient_ElectricalHeatingElement_BT6, outputVector_heatTransferCoefficient_StorageToRoom_BT6, outputVector_heatGenerationCoefficient_GasBoiler_BT7, outputVector_electricalPowerFanHeater_BT7,  objectiveMaximumLoad_OP1, objectiveSurplusEnergy_OP1, objectiveCosts_OP1, objectiveThermalDiscomfort_OP1, mipGapOfFoundSolution, timeForFindingOptimalSolution =   Building_Optimization_Problem.optimizeOneDay(indexOfBuildingsOverall_BT1, indexOfBuildingsOverall_BT2, indexOfBuildingsOverall_BT3, indexOfBuildingsOverall_BT4, indexOfBuildingsOverall_BT5,indexOfBuildingsOverall_BT6, indexOfBuildingsOverall_BT7, currentDay, includeObjectivesInReturnStatementCentralized, optParameters)
                    id_of_the_run += 1
                    df_results.loc[id_of_the_run - 1] = [id_of_the_run, objectiveCosts_OP1, objectiveMaximumLoad_OP1, optParameters["epsilon_objective_minimizeMaximumLoad_TargetValue"] , mipGapOfFoundSolution, timeForFindingOptimalSolution]

                    if create_result_load_profiles_multi_opt == True:
                        preCorrectSchedules_AvoidingFrequentStarts = False
                        overruleActions = False
                        pathForCreatingTheResultData = pathForCreatingTheResultData_Box + "/Profiles/Solution_ID" + str(id_of_the_run)
                        useLocalSearchHelp = False
                        simulationObjective_surplusEnergy_kWh_combined , simulationObjective_maximumLoad_kW_combined,simulationObjective_thermalDiscomfort_combined, simulationObjective_gasConsumptionkWh_combined, simulationObjective_costs_Euro_combined,  simulationObjective_combinedScore_combined= ICSimulation.simulateDays_WithLightController_Schedule(indexOfBuildingsOverall_BT1, indexOfBuildingsOverall_BT2, indexOfBuildingsOverall_BT3, indexOfBuildingsOverall_BT4, indexOfBuildingsOverall_BT5, indexOfBuildingsOverall_BT6, indexOfBuildingsOverall_BT7, currentDay, outputVector_heatGenerationCoefficientSpaceHeating_BT1, outputVector_heatGenerationCoefficientDHW_BT1, outputVector_chargingPowerEV_BT1, outputVector_heatGenerationCoefficientSpaceHeating_BT2, outputVector_heatGenerationCoefficientDHW_BT2, outputVector_chargingPowerEV_BT3, outputVector_heatGenerationCoefficientSpaceHeating_BT4, outputVector_chargingPowerBAT_BT5, outputVector_disChargingPowerBAT_BT5, outputVector_heatGenerationCoefficient_GasBoiler_BT6, outputVector_heatGenerationCoefficient_ElectricalHeatingElement_BT6, outputVector_heatTransferCoefficient_StorageToRoom_BT6, outputVector_heatGenerationCoefficient_GasBoiler_BT7, outputVector_electricalPowerFanHeater_BT7, pathForCreatingTheResultData, preCorrectSchedules_AvoidingFrequentStarts, optParameters, useLocalSearchHelp)
                        df_results_after_internal_controller.loc[id_of_the_run - 1] = [id_of_the_run, simulationObjective_costs_Euro_combined, simulationObjective_maximumLoad_kW_combined, simulationObjective_thermalDiscomfort_combined]



                    # Optimize the second goal: Peak load
                    optParameters['objective_minimizeCosts_weight'] = 0.01
                    optParameters['objective_minimizePeakLoad_weight'] = 1 - optParameters['objective_minimizeCosts_weight']



                    outputVector_heatGenerationCoefficientSpaceHeating_BT1, outputVector_heatGenerationCoefficientDHW_BT1, outputVector_chargingPowerEV_BT1, outputVector_heatGenerationCoefficientSpaceHeating_BT2, outputVector_heatGenerationCoefficientDHW_BT2, outputVector_chargingPowerEV_BT3, outputVector_heatGenerationCoefficientSpaceHeating_BT4, outputVector_chargingPowerBAT_BT5, outputVector_disChargingPowerBAT_BT5, outputVector_heatGenerationCoefficient_GasBoiler_BT6, outputVector_heatGenerationCoefficient_ElectricalHeatingElement_BT6, outputVector_heatTransferCoefficient_StorageToRoom_BT6, outputVector_heatGenerationCoefficient_GasBoiler_BT7, outputVector_electricalPowerFanHeater_BT7,  objectiveMaximumLoad_OP2, objectiveSurplusEnergy_OP2, objectiveCosts_OP2,objectiveThermalDiscomfort_OP2, mipGapOfFoundSolution, timeForFindingOptimalSolution =   Building_Optimization_Problem.optimizeOneDay(indexOfBuildingsOverall_BT1, indexOfBuildingsOverall_BT2, indexOfBuildingsOverall_BT3, indexOfBuildingsOverall_BT4, indexOfBuildingsOverall_BT5,indexOfBuildingsOverall_BT6, indexOfBuildingsOverall_BT7, currentDay, includeObjectivesInReturnStatementCentralized, optParameters)
                    id_of_the_run += 1
                    df_results.loc[id_of_the_run - 1] = [id_of_the_run, objectiveCosts_OP2, objectiveMaximumLoad_OP2, optParameters["epsilon_objective_minimizeMaximumLoad_TargetValue"] , mipGapOfFoundSolution, timeForFindingOptimalSolution]

                    if create_result_load_profiles_multi_opt == True:
                        preCorrectSchedules_AvoidingFrequentStarts = False
                        overruleActions = False
                        pathForCreatingTheResultData = pathForCreatingTheResultData_Box + "/Profiles/Solution_ID" + str(id_of_the_run)
                        useLocalSearchHelp = False
                        simulationObjective_surplusEnergy_kWh_combined , simulationObjective_maximumLoad_kW_combined,simulationObjective_thermalDiscomfort_combined, simulationObjective_gasConsumptionkWh_combined, simulationObjective_costs_Euro_combined,  simulationObjective_combinedScore_combined= ICSimulation.simulateDays_WithLightController_Schedule(indexOfBuildingsOverall_BT1, indexOfBuildingsOverall_BT2, indexOfBuildingsOverall_BT3, indexOfBuildingsOverall_BT4, indexOfBuildingsOverall_BT5, indexOfBuildingsOverall_BT6, indexOfBuildingsOverall_BT7, currentDay, outputVector_heatGenerationCoefficientSpaceHeating_BT1, outputVector_heatGenerationCoefficientDHW_BT1, outputVector_chargingPowerEV_BT1, outputVector_heatGenerationCoefficientSpaceHeating_BT2, outputVector_heatGenerationCoefficientDHW_BT2, outputVector_chargingPowerEV_BT3, outputVector_heatGenerationCoefficientSpaceHeating_BT4, outputVector_chargingPowerBAT_BT5, outputVector_disChargingPowerBAT_BT5, outputVector_heatGenerationCoefficient_GasBoiler_BT6, outputVector_heatGenerationCoefficient_ElectricalHeatingElement_BT6, outputVector_heatTransferCoefficient_StorageToRoom_BT6, outputVector_heatGenerationCoefficient_GasBoiler_BT7, outputVector_electricalPowerFanHeater_BT7, pathForCreatingTheResultData, preCorrectSchedules_AvoidingFrequentStarts, optParameters, useLocalSearchHelp)
                        df_results_after_internal_controller.loc[id_of_the_run - 1] = [id_of_the_run, simulationObjective_costs_Euro_combined, simulationObjective_maximumLoad_kW_combined, simulationObjective_thermalDiscomfort_combined]


                    #Do the box-method iteration
                    terminationConditionReached = False
                    iterationCounter = 0
                    if scaleResultsWeightedSum ==True:
                        optParameters['objective_minimizeCosts_normalizationValue'] =  objectiveCosts_OP1
                        optParameters['objective_minimizePeakLoad_normalizationValue'] = objectiveMaximumLoad_OP2



                    while terminationConditionReached == False:
                        # Sort the temporary results
                        df_results.sort_values(by=["Peak Load"], ascending=False, inplace=True)
                        num_solution = df_results.shape[0]

                        #Calculate average difference between the peak load values of all solutions
                        peak_load_diff = df_results["Peak Load"].diff().abs()  # compute absolute differences
                        peak_load_diff = peak_load_diff.dropna()  # drop any NaN values
                        avg_peak_load_diff = peak_load_diff.mean()  # compute mean of differences


                        if avg_peak_load_diff < boxMethodTermination_AverageDifference or num_solution > boxMethodTermination_NumberOfSolutions:
                            terminationConditionReached = True
                            break
                        for i in range (0, num_solution - 1):

                            optParameters['optimizationGoal_minimizePeakLoad'] = False
                            optParameters['optimizationGoal_minimizeCosts'] = True
                            optParameters['optimization_1Objective'] = True
                            optParameters['optimization_2Objective'] = False

                            optParameters["epsilon_objective_minimizeCosts_Active"] = False
                            optParameters["epsilon_objective_minimizePeakLoad_Active"] = True
                            optParameters["epsilon_objective_minimizeGasConsumption_Active"] = False
                            optParameters["epsilon_objective_minimizeThermalDiscomfort_Active"] = False
                            optParameters["epsilon_objective_minimizeCosts_TargetValue"] = 9999999
                            optParameters["epsilon_objective_minimizeMaximumLoad_TargetValue"] =  (df_results.iloc[i, df_results.columns.get_loc("Peak Load")] + df_results.iloc[i + 1, df_results.columns.get_loc("Peak Load")])/2
                            optParameters["epsilon_objective_minimizeGasConsumption_TargetValue"] = 9999999
                            optParameters["epsilon_objective_minimizeThermalDiscomfort_TargetValue"] = 9999999
                            boxSize =  optParameters["epsilon_objective_minimizeMaximumLoad_TargetValue"]


                            try:
                                outputVector_heatGenerationCoefficientSpaceHeating_BT1, outputVector_heatGenerationCoefficientDHW_BT1, outputVector_chargingPowerEV_BT1, outputVector_heatGenerationCoefficientSpaceHeating_BT2, outputVector_heatGenerationCoefficientDHW_BT2, outputVector_chargingPowerEV_BT3, outputVector_heatGenerationCoefficientSpaceHeating_BT4, outputVector_chargingPowerBAT_BT5, outputVector_disChargingPowerBAT_BT5, outputVector_heatGenerationCoefficient_GasBoiler_BT6, outputVector_heatGenerationCoefficient_ElectricalHeatingElement_BT6, outputVector_heatTransferCoefficient_StorageToRoom_BT6, outputVector_heatGenerationCoefficient_GasBoiler_BT7, outputVector_electricalPowerFanHeater_BT7,  objectiveMaximumLoad_OPEpsilon, objectiveSurplusEnergy_OPEpsilon, objectiveCosts_OPEpsilon,objectiveThermalDiscomfort_OPEpsilon, mipGapOfFoundSolution, timeForFindingOptimalSolution =   Building_Optimization_Problem.optimizeOneDay(indexOfBuildingsOverall_BT1, indexOfBuildingsOverall_BT2, indexOfBuildingsOverall_BT3, indexOfBuildingsOverall_BT4, indexOfBuildingsOverall_BT5,indexOfBuildingsOverall_BT6, indexOfBuildingsOverall_BT7, currentDay, includeObjectivesInReturnStatementCentralized, optParameters)
                            except:
                                break
                            df_results.loc[id_of_the_run - 1] = [id_of_the_run, objectiveCosts_OPEpsilon, objectiveMaximumLoad_OPEpsilon, optParameters["epsilon_objective_minimizeMaximumLoad_TargetValue"] , mipGapOfFoundSolution, timeForFindingOptimalSolution]
                            id_of_the_run += 1


                            if create_result_load_profiles_multi_opt == True:
                                preCorrectSchedules_AvoidingFrequentStarts = False
                                overruleActions = False
                                pathForCreatingTheResultData = pathForCreatingTheResultData_Box + "/Profiles/Solution_ID" + str(id_of_the_run)
                                useLocalSearchHelp = False
                                simulationObjective_surplusEnergy_kWh_combined , simulationObjective_maximumLoad_kW_combined,simulationObjective_thermalDiscomfort_combined, simulationObjective_gasConsumptionkWh_combined, simulationObjective_costs_Euro_combined,  simulationObjective_combinedScore_combined= ICSimulation.simulateDays_WithLightController_Schedule(indexOfBuildingsOverall_BT1, indexOfBuildingsOverall_BT2, indexOfBuildingsOverall_BT3, indexOfBuildingsOverall_BT4, indexOfBuildingsOverall_BT5, indexOfBuildingsOverall_BT6, indexOfBuildingsOverall_BT7, currentDay, outputVector_heatGenerationCoefficientSpaceHeating_BT1, outputVector_heatGenerationCoefficientDHW_BT1, outputVector_chargingPowerEV_BT1, outputVector_heatGenerationCoefficientSpaceHeating_BT2, outputVector_heatGenerationCoefficientDHW_BT2, outputVector_chargingPowerEV_BT3, outputVector_heatGenerationCoefficientSpaceHeating_BT4, outputVector_chargingPowerBAT_BT5, outputVector_disChargingPowerBAT_BT5, outputVector_heatGenerationCoefficient_GasBoiler_BT6, outputVector_heatGenerationCoefficient_ElectricalHeatingElement_BT6, outputVector_heatTransferCoefficient_StorageToRoom_BT6, outputVector_heatGenerationCoefficient_GasBoiler_BT7, outputVector_electricalPowerFanHeater_BT7, pathForCreatingTheResultData, preCorrectSchedules_AvoidingFrequentStarts, optParameters, useLocalSearchHelp)
                                df_results_after_internal_controller.loc[id_of_the_run - 1] = [id_of_the_run, simulationObjective_costs_Euro_combined, simulationObjective_maximumLoad_kW_combined, simulationObjective_thermalDiscomfort_combined]



                            if id_of_the_run > boxMethodTermination_NumberOfSolutions:
                                terminationConditionReached = True
                                df_results.sort_values(by=["Peak Load"], ascending=False, inplace=True)
                                break

                    # Print results to csv
                    titleOfThePlot = "Box Method - Day: " + str(currentDay) + " - "
                    appendixResultFile = "Box" + str(boxMethodTermination_NumberOfSolutions) + "_Day" + str(currentDay) + ""
                    if SetUpScenarios.numberOfBuildings_BT1 > 0:
                        titleOfThePlot = titleOfThePlot + "BT1: " + str(SetUpScenarios.numberOfBuildings_BT1) + ", "
                        appendixResultFile = appendixResultFile + "_BT1_" + str(SetUpScenarios.numberOfBuildings_BT1)
                    if SetUpScenarios.numberOfBuildings_BT2 > 0:
                        titleOfThePlot = titleOfThePlot + "BT2: " + str(SetUpScenarios.numberOfBuildings_BT2) + ", "
                        appendixResultFile = appendixResultFile + "_BT2_" + str(SetUpScenarios.numberOfBuildings_BT2)
                    if SetUpScenarios.numberOfBuildings_BT3 > 0:
                        titleOfThePlot = titleOfThePlot + "BT3: " + str(SetUpScenarios.numberOfBuildings_BT3) + ", "
                        appendixResultFile = appendixResultFile + "_BT3_" + str(SetUpScenarios.numberOfBuildings_BT3)
                    if SetUpScenarios.numberOfBuildings_BT4 > 0:
                        titleOfThePlot = titleOfThePlot + "BT4: " + str(SetUpScenarios.numberOfBuildings_BT4) + ", "
                        appendixResultFile = appendixResultFile + "_BT4_" + str(SetUpScenarios.numberOfBuildings_BT4)
                    if SetUpScenarios.numberOfBuildings_BT5 > 0:
                        titleOfThePlot = titleOfThePlot + "BT5: " + str(SetUpScenarios.numberOfBuildings_BT5) + ", "
                        appendixResultFile = appendixResultFile + "_BT5_" + str(SetUpScenarios.numberOfBuildings_BT5)
                    if SetUpScenarios.numberOfBuildings_BT6 > 0:
                        titleOfThePlot = titleOfThePlot + "BT6: " + str(SetUpScenarios.numberOfBuildings_BT6) + ", "
                        appendixResultFile = appendixResultFile + "_BT6_" + str(SetUpScenarios.numberOfBuildings_BT6)
                    if SetUpScenarios.numberOfBuildings_BT7 > 0:
                        titleOfThePlot = titleOfThePlot + "BT7: " + str(SetUpScenarios.numberOfBuildings_BT7) + ", "
                        appendixResultFile = appendixResultFile + "_BT7_" + str(SetUpScenarios.numberOfBuildings_BT7)
                    titleOfThePlot = titleOfThePlot[:-2]


                    #Round the values of the result dataframe
                    df_results['Costs'] = df_results['Costs'] / 100
                    df_results['Costs'] = df_results['Costs'].round(2)
                    df_results['Peak Load'] = df_results['Peak Load'] / 1000
                    df_results['Peak Load'] = df_results['Peak Load'].round(2)
                    df_results['Solving time'] = df_results['Solving time'].round(0)
                    df_results['epsilon_MaximumLoad_TargetValue'] = df_results['epsilon_MaximumLoad_TargetValue'].round(0)


                    #Adjust the values of the result_v2 dataframes which contains the values after the simulation with the internal controller
                    columns_to_extract = ['Peak Load', 'Costs', 'Thermal Discomfort']
                    # Use the apply method to extract the first element from each list or array
                    for column in columns_to_extract:
                        df_results_after_internal_controller[column] = df_results_after_internal_controller[column].apply(lambda x: x[0])


                    #Print the result to a csv file
                    #df_results.to_csv(pathForCreatingTheResultData_Box + "/Peak_Costs" + appendixResultFile + ".csv",index=False, sep=";")
                    df_results_after_internal_controller.to_csv(pathForCreatingTheResultData_Box + "/Peak_Costs" + appendixResultFile + "_v2.csv",index=False, sep=";")




                    # Create figure with pareto front
                    import pandas as pd

                    import pandas as pd
                    import matplotlib.pyplot as plt
                    import matplotlib

                    matplotlib.use('Agg')

                    if df_results_after_internal_controller.empty:
                        print("Error: df_results_after_internal_controller DataFrame is empty.")
                    else:
                        # Create an empty DataFrame to store the Pareto-efficient solutions
                        pareto_front = pd.DataFrame(columns=['Costs', 'Peak Load'])

                        # Loop through all solutions in the DataFrame
                        for i, row in df_results_after_internal_controller.iterrows():
                            # Assume the current solution is Pareto-efficient until proven otherwise
                            is_efficient = True
                            # Loop through all existing solutions in the Pareto front
                            for j, pareto_row in pareto_front.iterrows():
                                # Check if the current solution is dominated by any existing solution in the Pareto front
                                if (row['Costs'] >= pareto_row['Costs'] and row['Peak Load'] >= pareto_row['Peak Load']):
                                    is_efficient = False
                                    break
                                # Check if the current solution dominates any existing solution in the Pareto front
                                elif (row['Costs'] <= pareto_row['Costs'] and row['Peak Load'] <= pareto_row['Peak Load']):
                                    pareto_front = pareto_front.drop(j)
                            # If the current solution is Pareto-efficient, add it to the Pareto front
                            if is_efficient and row['Thermal Discomfort'] <= threshold_discomfort_local_search:
                                pareto_front = pd.concat([pareto_front, row.to_frame().T], ignore_index=True)

                        if pareto_front.empty:
                            print("Error: pareto_front DataFrame is empty.")
                        else:
                            # Sort the Pareto front by 'Costs' and 'Peak Load'
                            pareto_front = pareto_front.sort_values(['Costs', 'Peak Load'], ascending=[True, True])
                            # Reset the index of the Pareto front DataFrame
                            pareto_front = pareto_front.reset_index(drop=True)

                            #Print the pareto optimal solution into a csv file
                            pareto_front.to_csv(pathForCreatingTheResultData_Box + "/ParetoFront_" + appendixResultFile + ".csv",index=False, sep=";")

                            # Plot Pareto efficient solutions with line
                            plt.scatter(pareto_front['Costs'], pareto_front['Peak Load'], color='blue')
                            plt.plot(pareto_front['Costs'], pareto_front['Peak Load'], color='red')
                            plt.xlabel('Costs', fontsize=15)
                            plt.ylabel('Peak Load', fontsize=15)
                            if font_size_title_Pareto_Plot > 0:
                                plt.title(titleOfThePlot, fontsize=font_size_title_Pareto_Plot)
                            plt.tick_params(axis='both', which='major', labelsize=11)
                            plt.subplots_adjust(left=0.15)
                            plt.savefig(pathForCreatingTheResultData_Box + '/PFront_Line_' + appendixResultFile + '.png', dpi=100)
                            # Clear current figure
                            plt.clf()
                            # Plot Pareto efficient solutions without line
                            plt.scatter(pareto_front['Costs'], pareto_front['Peak Load'], color='blue')
                            plt.xlabel('Costs', fontsize=14)
                            plt.ylabel('Peak Load', fontsize=14)
                            plt.subplots_adjust(left=0.15)
                            if font_size_title_Pareto_Plot > 0:
                                plt.title(titleOfThePlot, fontsize=font_size_title_Pareto_Plot)
                            plt.tick_params(axis='both', which='major', labelsize=11)
                            plt.savefig(pathForCreatingTheResultData_Box + '/PFront_' + appendixResultFile + '.png',dpi=100)

                if useLocalSearch == False and useDichotomicMethodCentralized_Cost_Peak == False and  useDichotomicMethodCentralized_Cost_Comfort == False and  useBoxMethodCentralized_Cost_Peak == False and  useBoxMethodCentralized_Cost_Comfort == False:
                    includeObjectivesInReturnStatementCentralized = True
                    outputVector_heatGenerationCoefficientSpaceHeating_BT1, outputVector_heatGenerationCoefficientDHW_BT1, outputVector_chargingPowerEV_BT1, outputVector_heatGenerationCoefficientSpaceHeating_BT2, outputVector_heatGenerationCoefficientDHW_BT2, outputVector_chargingPowerEV_BT3, outputVector_heatGenerationCoefficientSpaceHeating_BT4, outputVector_chargingPowerBAT_BT5, outputVector_disChargingPowerBAT_BT5, outputVector_heatGenerationCoefficient_GasBoiler_BT6, outputVector_heatGenerationCoefficient_ElectricalHeatingElement_BT6, outputVector_heatTransferCoefficient_StorageToRoom_BT6, outputVector_heatGenerationCoefficient_GasBoiler_BT7, outputVector_electricalPowerFanHeater_BT7, objectiveMaximumLoad_OP, objectiveSurplusEnergy_OP, objectiveCosts_OP,objectiveThermalDiscomfort_OP, mipGapPercentOfFoundSolution, timeForFindingOptimalSolution =   Building_Optimization_Problem.optimizeOneDay(indexOfBuildingsOverall_BT1, indexOfBuildingsOverall_BT2, indexOfBuildingsOverall_BT3, indexOfBuildingsOverall_BT4, indexOfBuildingsOverall_BT5,indexOfBuildingsOverall_BT6, indexOfBuildingsOverall_BT7, currentDay, includeObjectivesInReturnStatementCentralized, optParameters)
                    useLocalSearchHelp = False
                    simulationObjective_surplusEnergy_kWh_combined , simulationObjective_maximumLoad_kW_combined, simulationObjective_costs_Euro_combined,simulationObjective_thermalDiscomfort_combined,simulationObjective_gasConsumptionkWh_combined,  simulationObjective_combinedScore_combined = ICSimulation.simulateDays_WithLightController_Schedule(indexOfBuildingsOverall_BT1, indexOfBuildingsOverall_BT2, indexOfBuildingsOverall_BT3, indexOfBuildingsOverall_BT4, indexOfBuildingsOverall_BT5, indexOfBuildingsOverall_BT6, indexOfBuildingsOverall_BT7, currentDay, outputVector_heatGenerationCoefficientSpaceHeating_BT1, outputVector_heatGenerationCoefficientDHW_BT1, outputVector_chargingPowerEV_BT1, outputVector_heatGenerationCoefficientSpaceHeating_BT2, outputVector_heatGenerationCoefficientDHW_BT2, outputVector_chargingPowerEV_BT3, outputVector_heatGenerationCoefficientSpaceHeating_BT4, outputVector_chargingPowerBAT_BT5, outputVector_disChargingPowerBAT_BT5,outputVector_heatGenerationCoefficient_GasBoiler_BT6, outputVector_heatGenerationCoefficient_ElectricalHeatingElement_BT6, outputVector_heatTransferCoefficient_StorageToRoom_BT6, outputVector_heatGenerationCoefficient_GasBoiler_BT7, outputVector_electricalPowerFanHeater_BT7, pathForCreatingTheResultData_Centralized, preCorrectSchedules_AvoidingFrequentStarts, optParameters, useLocalSearchHelp)


            # Conventional Control
            if useConventionalControl == True:
                print("\n--------------Conventional Control------------\n")

                indexOfBuildingsOverall_BT1 = [i + 1 for i in range (0, SetUpScenarios.numberOfBuildings_BT1)]
                indexOfBuildingsOverall_BT2 = [i + 1 for i in range (0, SetUpScenarios.numberOfBuildings_BT2)]
                indexOfBuildingsOverall_BT3 = [i + 1 for i in range (0, SetUpScenarios.numberOfBuildings_BT3)]
                indexOfBuildingsOverall_BT4 = [i + 1 for i in range (0, SetUpScenarios.numberOfBuildings_BT4)]
                indexOfBuildingsOverall_BT5 = [i + 1 for i in range (0, SetUpScenarios.numberOfBuildings_BT5)]
                indexOfBuildingsOverall_BT6 = [i + 1 for i in range (0, SetUpScenarios.numberOfBuildings_BT6)]
                indexOfBuildingsOverall_BT7 = [i + 1 for i in range (0, SetUpScenarios.numberOfBuildings_BT7)]
                useLocalSearch = False
                simulationObjective_surplusEnergy_kWh_combined , simulationObjective_maximumLoad_kW_combined, simulationObjective_thermalDiscomfort_combined, simulationObjective_gasConsumptionkWh_combined, simulationObjective_costs_Euro_combined, simulationObjective_combinedScore_combined = ICSimulation.simulateDays_ConventionalControl(indexOfBuildingsOverall_BT1, indexOfBuildingsOverall_BT2, indexOfBuildingsOverall_BT3, indexOfBuildingsOverall_BT4,indexOfBuildingsOverall_BT5,indexOfBuildingsOverall_BT6, indexOfBuildingsOverall_BT7, currentDay, pathForCreatingTheResultData_Conventional, useLocalSearch)

    #Print result of multi objective optimization from local search and calculate averages over all days
    if useLocalSearch==True or withRL:
        averages = df_results_multiopt_local_search.iloc[:, 1:].mean()
        #averages = averages.round(1)
        average_row = pd.DataFrame(averages).T
        average_row["Day"] = "Average"
        df_results_multiopt_local_search = pd.concat([df_results_multiopt_local_search, pd.DataFrame(average_row, index=[0])], ignore_index=True)
        time_limit_in_minutes = int(time_limit_in_seconds_for_local_search / 60)
        df_results_multiopt_local_search.to_csv(
            folderPath_resultFile_multiOpt + "/RL2_3_result_multiopt_" + "min" + str(time_limit_in_minutes) + ".csv",
            sep=';', index=False) if withRL else df_results_multiopt_local_search.to_csv(folderPath_resultFile_multiOpt + "/LS_result_multiopt_" + "min" + str(time_limit_in_minutes) + ".csv", sep=';',index=False)

        #Print the parameters of the run as a txt file
        folderPath_resultFile_multiOpt_txt_file = folderPath_resultFile_multiOpt + "/Run_Parameters.txt"
        with open(folderPath_resultFile_multiOpt_txt_file, 'w') as file:
            file.write(f"max_population_size: {max_population_size}\n")
            file.write(f"number_of_pareto_optimal_solutions_in_population: {number_of_pareto_optimal_solutions_in_population}\n")
            file.write(f"number_of_new_solutions_per_solution_in_iteration: {number_of_new_solutions_per_solution_in_iteration}\n")
            file.write(f"number_of_iterations_local_search: {number_of_iterations_local_search}\n")
            file.write(f"time_limit_in_seconds_for_local_search: {time_limit_in_seconds_for_local_search}\n")
            if withRL:
                file.write(f"model RL2: {model_path_extension_RL2}\n")
                file.write(f"model RL3: {model_path_extension_RL3}\n")




    if not withRL:
        #Calculate the end time and print it
        end_time = time.time()
        duration_seconds = end_time - start_time
        hours, remainder = divmod(duration_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        time_str = f"{int(hours)} hours, {int(minutes)} minutes, {int(seconds)} seconds"
        print(f"Time taken: {time_str}")

    # only used for plots
    return folderPath_resultFile_multiOpt, appendixResultFile

if __name__ == '__main__':
    run_simulations(withRL=True)
