
"""
This file contains two methods that exectue the actions of the two operators for local search (price-shift and peak-shift) for the Reinforcement
learning agents.

The results (including resulting load profiles) are store on file (path specified by the variable "pathForCreatingTheResultData") and returned at the end of the functions
"""
from random import random

import SetUpScenarios
import numpy as np
import os
import ICSimulation
import config



#Objectives and scenarios
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
    'epsilon_objective_minimizeCosts_Active': False,
    'epsilon_objective_minimizePeakLoad_Active': False,
    'epsilon_objective_minimizeGasConsumption_Active': False ,
    'epsilon_objective_minimizeThermalDiscomfort_Active':False,
    'epsilon_objective_minimizeCosts_TargetValue': 9999999,
    'epsilon_objective_minimizeMaximumLoad_TargetValue': 9999999,
    'epsilon_objective_minimizeGasConsumption_TargetValue': 9999999,
    'epsilon_objective_minimizeThermalDiscomfort_TargetValue': 9999999
}
threshold_discomfort_local_search = 0.3

def execute_single_modification_operator_decision_RL3_peak_shift (base_solution, action_to_timeslot, action_shifting_percentage, current_day, timeslots_for_state_load_percentages):
    return execute_single_modification_operator_decision_RL2_price_or_peak_shift(base_solution, None, action_to_timeslot, action_shifting_percentage, current_day, None, timeslots_for_state_load_percentages, False)

def execute_single_modification_operator_decision_RL2_price_shift (base_solution, action_from_timeslot, action_to_timeslot, action_shifting_percentage, current_day, timeslots_for_state_load_percentages_costs):
    return execute_single_modification_operator_decision_RL2_price_or_peak_shift(base_solution, action_from_timeslot, action_to_timeslot, action_shifting_percentage, current_day, timeslots_for_state_load_percentages_costs, None, True)


#This function exectues one single modification for the price shift operator on a current solution (base_solution) using an RL agent
def execute_single_modification_operator_decision_RL2_price_or_peak_shift(base_solution, action_from_timeslot, action_to_timeslot, action_shifting_percentage, current_day, timeslots_for_state_load_percentages_costs, timeslots_for_state_load_percentages, is_price_shift_operator):

    indexOfBuildingsOverall_BT1 = [i + 1 for i in range(0, SetUpScenarios.numberOfBuildings_BT1)]
    indexOfBuildingsOverall_BT2 = [i + 1 for i in range(0, SetUpScenarios.numberOfBuildings_BT2)]
    indexOfBuildingsOverall_BT3 = [i + 1 for i in range(0, SetUpScenarios.numberOfBuildings_BT3)]
    indexOfBuildingsOverall_BT4 = [i + 1 for i in range(0, SetUpScenarios.numberOfBuildings_BT4)]
    indexOfBuildingsOverall_BT5 = [i + 1 for i in range(0, SetUpScenarios.numberOfBuildings_BT5)]
    indexOfBuildingsOverall_BT6 = [i + 1 for i in range(0, SetUpScenarios.numberOfBuildings_BT6)]
    indexOfBuildingsOverall_BT7 = [i + 1 for i in range(0, SetUpScenarios.numberOfBuildings_BT7)]
    price_array = base_solution['price_array'].copy()
    electrical_load_combined = base_solution['simulationResult_electricalLoad_combined']
    sum_of_electrical_loads = np.sum(electrical_load_combined)

    #Do the local search iteration

    # set the solution vector (original solution) for the next solution
    outputVector_BT1_heatGenerationCoefficientSpaceHeating_original = base_solution['outputVector_BT1_heatGenerationCoefficientSpaceHeating'].copy()
    outputVector_BT1_heatGenerationCoefficientDHW_original = base_solution['outputVector_BT1_heatGenerationCoefficientDHW'].copy()
    outputVector_BT1_chargingPowerEV_original = base_solution['outputVector_BT1_chargingPowerEV'].copy()
    outputVector_BT2_heatGenerationCoefficientSpaceHeating_original = base_solution['outputVector_BT2_heatGenerationCoefficientSpaceHeating'].copy()
    outputVector_BT2_heatGenerationCoefficientDHW_original = base_solution['outputVector_BT2_heatGenerationCoefficientDHW'].copy()
    outputVector_BT3_chargingPowerEV_original = base_solution['outputVector_BT3_chargingPowerEV'].copy()
    outputVector_BT4_heatGenerationCoefficientSpaceHeating_original = base_solution['outputVector_BT4_heatGenerationCoefficientSpaceHeating'].copy()
    outputVector_BT5_chargingPowerBAT_original = base_solution['outputVector_BT5_chargingPowerBAT'].copy()
    outputVector_BT5_disChargingPowerBAT_original = base_solution['outputVector_BT5_disChargingPowerBAT'].copy()
    outputVector_BT6_heatGenerationCoefficient_GasBoiler_original =base_solution['outputVector_BT6_heatGenerationCoefficient_GasBoiler'].copy()
    outputVector_BT6_heatGenerationCoefficient_ElectricalHeatingElement_original = base_solution['outputVector_BT6_heatGenerationCoefficient_ElectricalHeatingElement'].copy()
    outputVector_BT6_heatTransferCoefficient_StorageToRoom_original = base_solution['outputVector_BT6_heatTransferCoefficient_StorageToRoom'].copy()
    outputVector_BT7_heatGenerationCoefficient_GasBoiler_original = base_solution['outputVector_BT7_heatGenerationCoefficient_GasBoiler'].copy()
    outputVector_BT7_electricalPowerFanHeater_original = base_solution['outputVector_BT7_electricalPowerFanHeater'].copy()


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
    outputVector_BT7_electricalPowerFanHeater = outputVector_BT7_electricalPowerFanHeater_original.copy()


    pathForCreatingTheResultData = os.path.join(config.DIR_RESULT, "RL_Temp")
    preCorrectSchedules_AvoidingFrequentStarts = False
    use_local_search = True
    returned_objects = ICSimulation.simulateDays_WithLightController_Schedule(indexOfBuildingsOverall_BT1, indexOfBuildingsOverall_BT2, indexOfBuildingsOverall_BT3, indexOfBuildingsOverall_BT4, indexOfBuildingsOverall_BT5, indexOfBuildingsOverall_BT6, indexOfBuildingsOverall_BT7, current_day, outputVector_BT1_heatGenerationCoefficientSpaceHeating, outputVector_BT1_heatGenerationCoefficientDHW, outputVector_BT1_chargingPowerEV, outputVector_BT2_heatGenerationCoefficientSpaceHeating, outputVector_BT2_heatGenerationCoefficientDHW, outputVector_BT3_chargingPowerEV, outputVector_BT4_heatGenerationCoefficientSpaceHeating, outputVector_BT5_chargingPowerBAT, outputVector_BT5_disChargingPowerBAT, outputVector_BT6_heatGenerationCoefficient_GasBoiler, outputVector_BT6_heatGenerationCoefficient_ElectricalHeatingElement, outputVector_BT6_heatTransferCoefficient_StorageToRoom, outputVector_BT7_heatGenerationCoefficient_GasBoiler, outputVector_BT7_electricalPowerFanHeater, pathForCreatingTheResultData, preCorrectSchedules_AvoidingFrequentStarts, optParameters, use_local_search)

    electrical_load_profile_current_solution = returned_objects[6]
    availability_pattern_EV_BT1 = returned_objects [8]
    thermal_discomfort_space_heating_BT1 = returned_objects[25]
    thermal_discomfort_dhw_BT1 = returned_objects[26]
    thermal_discomfort_space_heating_BT2= returned_objects[27]
    thermal_discomfort_dhw_BT2= returned_objects[28]
    thermal_discomfort_space_heating_BT4= returned_objects[29]



    #Choose the search direction (optimization goal)
    # price shift operator: True,False, False; peak shift operator: False, True, False
    optimize_costs_local_search = is_price_shift_operator
    optimize_peak_local_search = not is_price_shift_operator
    optimize_comfort_local_search = False



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

        #Adust the arrays
        adjusted_percentage_array_loads_highest_prices_temp = percentage_array_loads_per_timeslot_highest_prices.reshape(12, 2)
        adjusted_percentage_array_loads_highest_prices = adjusted_percentage_array_loads_highest_prices_temp.sum(axis=1)

        adjusted_percentage_array_loads_lowest_prices_temp = percentage_array_loads_per_timeslot_lowest_prices.reshape(12, 2)
        adjusted_percentage_array_loads_lowest_prices = adjusted_percentage_array_loads_lowest_prices_temp.sum(axis=1)

        adjusted_array_lowest_prices = np.minimum(reshaped_array_lowest_prices[::2], reshaped_array_lowest_prices[1::2])
        adjusted_array_highest_prices = np.minimum(reshaped_array_highest_prices[::2],reshaped_array_highest_prices[1::2])

        percentage_array_loads_per_timeslot_highest_prices_shortened = adjusted_percentage_array_loads_highest_prices [0:timeslots_for_state_load_percentages_costs]
        percentage_array_loads_per_timeslot_lowest_prices_shortened = adjusted_percentage_array_loads_lowest_prices [0:timeslots_for_state_load_percentages_costs]

        percentage_array_loads_per_timeslot_highest_prices_shortened_before_action = percentage_array_loads_per_timeslot_highest_prices_shortened
        percentage_array_loads_per_timeslot_lowest_prices_shortened_before_action = percentage_array_loads_per_timeslot_lowest_prices_shortened


        param_1_number_of_partitions_for_shifting = 1


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


    #local search direction peak reduction (not used in this function)
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

        if is_price_shift_operator:
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
        if not is_price_shift_operator:
            # Calculate the load percentages (in relation to the maxium load) of the timeslots with the lowest prices
            reshaped_lowest_prices_fully_day = lowest_k_prices_array_full_day_1.reshape(-1)
            array_load_percentages_lowest_prices = np.zeros((len(reshaped_lowest_prices_fully_day)))
            for i in range(0, len(reshaped_lowest_prices_fully_day)):
                array_load_percentages_lowest_prices[i] = round((electrical_load_profile_current_solution[reshaped_lowest_prices_fully_day[i]] / max_load), 2) * 100

            array_load_percentages_lowest_prices_shortened_before = array_load_percentages_lowest_prices[0:timeslots_for_state_load_percentages]

        #Reduce load at all peaks
            shifting_percentage_load = action_shifting_percentage
            for peak_timeslot in list_timeslots_max_load:

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

                        # Shift DHW heating power from highes price timeslot to lowest
                    outputVector_BT1_heatGenerationCoefficientDHW [index_BT1 - 1]  [peak_timeslot] = outputVector_BT1_heatGenerationCoefficientDHW [index_BT1 - 1] [peak_timeslot]    * ((100-shifting_percentage_load)/100)
                    if reshaped_lowest_prices_fully_day[action_to_timeslot] not in list_timeslots_max_load:
                        outputVector_BT1_heatGenerationCoefficientDHW [index_BT1 - 1] [reshaped_lowest_prices_fully_day [action_to_timeslot]] = outputVector_BT1_heatGenerationCoefficientDHW [index_BT1 - 1]  [reshaped_lowest_prices_fully_day [action_to_timeslot]]   * ((100+shifting_percentage_load)/100)
                    if outputVector_BT1_heatGenerationCoefficientDHW[index_BT1 - 1][reshaped_lowest_prices_fully_day[action_to_timeslot]] < SetUpScenarios.minimalModulationdDegree_HP / 100:
                        outputVector_BT1_heatGenerationCoefficientDHW[index_BT1 - 1][reshaped_lowest_prices_fully_day[action_to_timeslot]] = SetUpScenarios.minimalModulationdDegree_HP / 100
                    if outputVector_BT1_heatGenerationCoefficientDHW[index_BT1 - 1][peak_timeslot] < SetUpScenarios.minimalModulationdDegree_HP / 100:
                        outputVector_BT1_heatGenerationCoefficientDHW[index_BT1 - 1][peak_timeslot] = 0
                    # Shift charging power EV from highes price timeslot to lowest
                    outputVector_BT1_chargingPowerEV [index_BT1 - 1]  [peak_timeslot]  = outputVector_BT1_chargingPowerEV [index_BT1 - 1] [peak_timeslot]    * ((100-shifting_percentage_load)/100)
                    if reshaped_lowest_prices_fully_day[action_to_timeslot] not in list_timeslots_max_load:
                        outputVector_BT1_chargingPowerEV [index_BT1 - 1] [reshaped_lowest_prices_fully_day [action_to_timeslot]] = outputVector_BT1_chargingPowerEV [index_BT1 - 1] [reshaped_lowest_prices_fully_day [action_to_timeslot]]   * ((100+shifting_percentage_load)/100)

                # Shift Space heating power and DHW from highes peak to lowest price
                for index_BT2 in indexOfBuildingsOverall_BT2:


                    # Shift Space heating power from highes price timeslot to lowest: Full day
                    outputVector_BT2_heatGenerationCoefficientSpaceHeating [index_BT2 - 1] [peak_timeslot] = outputVector_BT2_heatGenerationCoefficientSpaceHeating [index_BT2 - 1] [peak_timeslot]   * ((100-shifting_percentage_load)/100)
                    if reshaped_lowest_prices_fully_day[action_to_timeslot] not in list_timeslots_max_load:
                        outputVector_BT2_heatGenerationCoefficientSpaceHeating [index_BT2 - 1] [reshaped_lowest_prices_fully_day [action_to_timeslot]] = outputVector_BT2_heatGenerationCoefficientSpaceHeating [index_BT2 - 1]  [reshaped_lowest_prices_fully_day [action_to_timeslot]]   * ((100+shifting_percentage_load)/100)
                    if outputVector_BT2_heatGenerationCoefficientSpaceHeating [index_BT2 - 1] [reshaped_lowest_prices_fully_day [action_to_timeslot]] < SetUpScenarios.minimalModulationdDegree_HP /100:
                        outputVector_BT2_heatGenerationCoefficientSpaceHeating [index_BT2 - 1] [reshaped_lowest_prices_fully_day [action_to_timeslot]]= SetUpScenarios.minimalModulationdDegree_HP /100
                    if outputVector_BT2_heatGenerationCoefficientSpaceHeating [index_BT2 - 1] [peak_timeslot] < SetUpScenarios.minimalModulationdDegree_HP/100:
                        outputVector_BT2_heatGenerationCoefficientSpaceHeating[index_BT2 - 1][peak_timeslot] = 0


                    # Shift DHW heating power from highes price timeslot to lowest
                    outputVector_BT2_heatGenerationCoefficientDHW [index_BT2 - 1]  [peak_timeslot] = outputVector_BT2_heatGenerationCoefficientDHW [index_BT2 - 1] [peak_timeslot]    * ((100-shifting_percentage_load)/100)
                    if reshaped_lowest_prices_fully_day[action_to_timeslot] not in list_timeslots_max_load:
                        outputVector_BT2_heatGenerationCoefficientDHW [index_BT2 - 1] [reshaped_lowest_prices_fully_day [action_to_timeslot]] = outputVector_BT2_heatGenerationCoefficientDHW [index_BT2 - 1]  [reshaped_lowest_prices_fully_day [action_to_timeslot]]   * ((100+shifting_percentage_load)/100)
                    if outputVector_BT2_heatGenerationCoefficientDHW [index_BT2 - 1] [reshaped_lowest_prices_fully_day [action_to_timeslot]] < SetUpScenarios.minimalModulationdDegree_HP /100:
                        outputVector_BT2_heatGenerationCoefficientDHW [index_BT2 - 1] [reshaped_lowest_prices_fully_day [action_to_timeslot]]= SetUpScenarios.minimalModulationdDegree_HP /100
                    if outputVector_BT2_heatGenerationCoefficientDHW [index_BT2 - 1] [peak_timeslot] < SetUpScenarios.minimalModulationdDegree_HP/100:
                        outputVector_BT2_heatGenerationCoefficientDHW[index_BT2 - 1][peak_timeslot] = 0


                for index_BT4 in indexOfBuildingsOverall_BT4:
                    # Shift Space heating power from highes peak to lowest price

                    # Shift Space heating power from highes price timeslot to lowest: Full day
                    outputVector_BT4_heatGenerationCoefficientSpaceHeating [index_BT4 - 1] [peak_timeslot] = outputVector_BT4_heatGenerationCoefficientSpaceHeating [index_BT4 - 1] [peak_timeslot]   * ((100-shifting_percentage_load)/100)
                    if reshaped_lowest_prices_fully_day[action_to_timeslot] not in list_timeslots_max_load:
                        outputVector_BT4_heatGenerationCoefficientSpaceHeating [index_BT4 - 1] [reshaped_lowest_prices_fully_day [action_to_timeslot]] = outputVector_BT4_heatGenerationCoefficientSpaceHeating [index_BT4 - 1]  [reshaped_lowest_prices_fully_day [action_to_timeslot]]   * ((100+shifting_percentage_load)/100)

                    if outputVector_BT4_heatGenerationCoefficientSpaceHeating [index_BT4 - 1] [reshaped_lowest_prices_fully_day [action_to_timeslot]] < SetUpScenarios.minimalModulationdDegree_HP /100:
                        outputVector_BT4_heatGenerationCoefficientSpaceHeating [index_BT4 - 1] [reshaped_lowest_prices_fully_day [action_to_timeslot]]= SetUpScenarios.minimalModulationdDegree_HP /100
                    if outputVector_BT4_heatGenerationCoefficientSpaceHeating [index_BT4 - 1] [peak_timeslot] < SetUpScenarios.minimalModulationdDegree_HP/100:
                        outputVector_BT4_heatGenerationCoefficientSpaceHeating[index_BT4 - 1][peak_timeslot] = 0

    # local search direction comfort improvement (not used in this function)
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
    returned_objects = ICSimulation.simulateDays_WithLightController_Schedule(indexOfBuildingsOverall_BT1, indexOfBuildingsOverall_BT2, indexOfBuildingsOverall_BT3, indexOfBuildingsOverall_BT4, indexOfBuildingsOverall_BT5, indexOfBuildingsOverall_BT6, indexOfBuildingsOverall_BT7, current_day, outputVector_BT1_heatGenerationCoefficientSpaceHeating, outputVector_BT1_heatGenerationCoefficientDHW, outputVector_BT1_chargingPowerEV, outputVector_BT2_heatGenerationCoefficientSpaceHeating, outputVector_BT2_heatGenerationCoefficientDHW, outputVector_BT3_chargingPowerEV, outputVector_BT4_heatGenerationCoefficientSpaceHeating, outputVector_BT5_chargingPowerBAT, outputVector_BT5_disChargingPowerBAT, outputVector_BT6_heatGenerationCoefficient_GasBoiler, outputVector_BT6_heatGenerationCoefficient_ElectricalHeatingElement, outputVector_BT6_heatTransferCoefficient_StorageToRoom, outputVector_BT7_heatGenerationCoefficient_GasBoiler, outputVector_BT7_electricalPowerFanHeater, pathForCreatingTheResultData, preCorrectSchedules_AvoidingFrequentStarts, optParameters, use_local_search)
    results_dict = {"simulationObjective_surplusEnergy_kWh_combined": returned_objects[0], "simulationObjective_maximumLoad_kW_combined": returned_objects[1], "simulationObjective_thermalDiscomfort_combined": returned_objects[2], "simulationObjective_gasConsumptionkWh_combined": returned_objects[3], "simulationObjective_costs_Euro_combined": returned_objects[4], "simulationObjective_combinedScore_combined": returned_objects[5], "simulationResult_electricalLoad_combined": returned_objects[6], "price_array": returned_objects[7], "simulationInput_BT1_availabilityPattern": returned_objects[8], "combined_array_thermal_discomfort": returned_objects[9], "outputVector_BT1_heatGenerationCoefficientSpaceHeating": returned_objects[10], "outputVector_BT1_heatGenerationCoefficientDHW": returned_objects[11], "outputVector_BT1_chargingPowerEV": returned_objects[12], "outputVector_BT2_heatGenerationCoefficientSpaceHeating": returned_objects[13], "outputVector_BT2_heatGenerationCoefficientDHW": returned_objects[14], "outputVector_BT3_chargingPowerEV": returned_objects[15], "outputVector_BT4_heatGenerationCoefficientSpaceHeating": returned_objects[16], "outputVector_BT5_chargingPowerBAT": returned_objects[17], "outputVector_BT5_disChargingPowerBAT": returned_objects[18], "outputVector_BT6_heatGenerationCoefficient_GasBoiler": returned_objects[19], "outputVector_BT6_heatGenerationCoefficient_ElectricalHeatingElement": returned_objects[20], "outputVector_BT6_heatTransferCoefficient_StorageToRoom": returned_objects[21], "outputVector_BT7_heatGenerationCoefficient_GasBoiler": returned_objects[22], "outputVector_BT7_electricalPowerFanHeater": returned_objects[23], "combined_array_thermal_discomfort": returned_objects[24], "thermal_discomfort_space_heating_BT1": returned_objects[25], "thermal_discomfort_dhw_BT1": returned_objects[26], "thermal_discomfort_space_heating_BT2": returned_objects[27], "thermal_discomfort_dhw_BT2": returned_objects[28], "thermal_discomfort_space_heating_BT4": returned_objects[29]}

    help_print_optimization_operator = ""
    if optimize_costs_local_search == True:
        help_print_optimization_operator = "cost_opt"
    if optimize_peak_local_search == True:
        help_print_optimization_operator = "peak_opt"
    if optimize_comfort_local_search == True:
        help_print_optimization_operator = "comfort_opt"


    result_costs = returned_objects[4]
    result_peak = returned_objects[1]
    result_DC  = returned_objects[2]


    if is_price_shift_operator:
        #####
        # Find the timeslots with the highgest prices
        electrical_load_profile_current_solution = returned_objects[6]
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


        #Adust the arrays
        adjusted_percentage_array_loads_highest_prices_temp = percentage_array_loads_per_timeslot_highest_prices.reshape(12, 2)
        adjusted_percentage_array_loads_highest_prices = adjusted_percentage_array_loads_highest_prices_temp.sum(axis=1)

        adjusted_percentage_array_loads_lowest_prices_temp = percentage_array_loads_per_timeslot_lowest_prices.reshape(12, 2)
        adjusted_percentage_array_loads_lowest_prices = adjusted_percentage_array_loads_lowest_prices_temp.sum(axis=1)

        adjusted_array_lowest_prices = np.minimum(reshaped_array_lowest_prices[::2], reshaped_array_lowest_prices[1::2])
        adjusted_array_highest_prices = np.minimum(reshaped_array_highest_prices[::2],reshaped_array_highest_prices[1::2])

        percentage_array_loads_per_timeslot_highest_prices_shortened = adjusted_percentage_array_loads_highest_prices [0:timeslots_for_state_load_percentages_costs]
        percentage_array_loads_per_timeslot_lowest_prices_shortened = adjusted_percentage_array_loads_lowest_prices [0:timeslots_for_state_load_percentages_costs]


        return  result_costs, result_peak, result_DC, results_dict, percentage_array_loads_per_timeslot_highest_prices_shortened, percentage_array_loads_per_timeslot_lowest_prices_shortened, percentage_array_loads_per_timeslot_highest_prices_shortened_before_action, percentage_array_loads_per_timeslot_lowest_prices_shortened_before_action
    if not is_price_shift_operator:
#####
        # Find the timeslots with the lowest prices and calculate their load pecentage
        electrical_load_profile_current_solution = returned_objects[6]
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

        #Calculate the load percentages (in relation to the maxium load) of the timeslots with the lowest prices
        reshaped_lowest_prices_fully_day = lowest_k_prices_array_full_day_1.reshape(-1)
        array_load_percentages_lowest_prices = np.zeros((len(reshaped_lowest_prices_fully_day)))
        for i in range(0, len(reshaped_lowest_prices_fully_day)):
            array_load_percentages_lowest_prices[i] = round((electrical_load_profile_current_solution[reshaped_lowest_prices_fully_day[i]] / max_load), 2) * 100

        array_load_percentages_lowest_prices_shortened_after = array_load_percentages_lowest_prices[0:timeslots_for_state_load_percentages]


        return  result_costs, result_peak, result_DC, results_dict, array_load_percentages_lowest_prices_shortened_after, array_load_percentages_lowest_prices_shortened_before
















