"""
Centralized optimization problem for mutiple buildings implemented with the package pyomo
7 building types:
    - Building Type 1: single-familiy building with modulating air-source heat pump (mHP) and electric vehicle (EV) --> used in Paper as BT1
    - Building Type 2: single-familiy building with modulating air-source heat pump (mHP)  --> used in Paper as BT2
    - Builidng Type 3: single-familiy building with electric vehicle  --> not used in Paper
    - Building Type 4: multi-familiy building with modulating air-source heat pump (mHP) only for space heating (and not domestic hot water)  --> used in Paper as BT3
    - Building Type 5: Building with battery  --> not used in Paper
    - Building Type 6: Gas heating with hot water tank and additional electrical heating element. 2 Storage systems (Building mass, hot water tank)  --> not used in Paper
    - Building Type 7: Gas heating without hot water tank and with additional electrical fan heater.  --> not used in Paper

Price, temperature and demand data for the buildings (space heating, domestic hot water, electricity) are read from csv files with a 1-minute resolution

The results (including resulting load profiles) are both stored on file (path is specified by the variable "folderPath" in the function) and returned by the function (at the end of this script)
"""
import os
import sys

import SetUpScenarios

import config

def optimizeOneDay(indexOfBuildingsOverall_BT1, indexOfBuildingsOverall_BT2, indexOfBuildingsOverall_BT3, indexOfBuildingsOverall_BT4, indexOfBuildingsOverall_BT5, indexOfBuildingsOverall_BT6, indexOfBuildingsOverall_BT7, currentDay, includeObjectivesInReturnStatementCentralized, optParameters):
    """
    This method uses as solver to solve the optimization problem of a residential area with different building types for one day

    Args:
        indexOfBuildingsOverall_BT1 (list): List of buildings for type 1.
        indexOfBuildingsOverall_BT2 (list): List of buildings for type 2.
        indexOfBuildingsOverall_BT3 (list): List of buildings for type 3.
        indexOfBuildingsOverall_BT4 (list): List of buildings for type 4.
        indexOfBuildingsOverall_BT5 (list): List of buildings for type 5.
        indexOfBuildingsOverall_BT6 (list): List of buildings for type 6.
        indexOfBuildingsOverall_BT7 (list): List of buildings for type 7.
        currentDay (int): The current day.
        includeObjectivesInReturnStatementCentralized (bool): Whether to include objectives in the return statement.
        optParameters (dict): Optimization parameters.

    """

    import SetUpScenarios
    import Run_Simulations_Combined as Run_Simulations
    import pyomo.environ as pyo
    import pandas as pd
    from pyomo.util.infeasible import log_infeasible_constraints
    from pyomo.opt import SolverStatus, TerminationCondition
    import numpy as np
    
    import sys
    import os
    from datetime import datetime
    from time import sleep

    
    
    
    # specify if the output should be printed on a separate log file or on the console
    printLogToFile = False
    
    if printLogToFile == True:
        #Specify output file for the logs
        prev_stdout = sys.stdout
        sys.stdout = open(config.DIR_LOG_FILE, 'w')
    
    
    
    # define the directory to be created for the result files
    currentDatetimeString = datetime.today().strftime('%d_%m_%Y_Time_%H_%M_%S')
    folderName = currentDatetimeString + "_BTCombined_" + str(SetUpScenarios.numberOfBuildings_Total) 
    folderPath = os.path.join(config.DIR_CENTRAL_OPT_INSTANCE_BASE, folderName)
    
    try:
        os.makedirs(folderPath)
    except OSError:
        print ("Creation of the directory %s failed" % folderPath)
    else:
        print ("Successfully created the directory %s" % folderPath)
    
    sleep(0.5)
    
    
    #Define the model
    model = pyo.ConcreteModel()
    
    
    #Define the sets
    model.set_timeslots = pyo.RangeSet(1, SetUpScenarios.numberOfTimeSlotsPerDay)
    model.set_buildings_BT1 = pyo.RangeSet(1, SetUpScenarios.numberOfBuildings_BT1)
    model.set_buildings_BT2 = pyo.RangeSet(1, SetUpScenarios.numberOfBuildings_BT2)
    model.set_buildings_BT3 = pyo.RangeSet(1, SetUpScenarios.numberOfBuildings_BT3)
    model.set_buildings_BT4 = pyo.RangeSet(1, SetUpScenarios.numberOfBuildings_BT4)
    model.set_buildings_BT5 = pyo.RangeSet(1, SetUpScenarios.numberOfBuildings_BT5)
    model.set_buildings_BT6 = pyo.RangeSet(1, SetUpScenarios.numberOfBuildings_BT6)
    model.set_buildings_BT7 = pyo.RangeSet(1, SetUpScenarios.numberOfBuildings_BT7)

    
    #Reading of the price data
    df_priceData_original = pd.read_csv(os.path.join(config.DIR_PRICE_ONE_MINUTE_DAYS, SetUpScenarios.typeOfPriceData + '/Price_' + SetUpScenarios.typeOfPriceData + '_1Minute_Day' + str(currentDay) + '.csv'), sep =";")
    df_priceData_original['Time'] = pd.to_datetime(df_priceData_original['Time'], format = '%d.%m.%Y %H:%M')
    df_priceData = df_priceData_original.set_index('Time').resample(str(SetUpScenarios.timeResolution_InMinutes) +'Min').mean()
    arrayTimeSlots = [i for i in range (1,SetUpScenarios.numberOfTimeSlotsPerDay + 1)]
    df_priceData['Timeslot'] = arrayTimeSlots
    df_priceData = df_priceData.set_index('Timeslot')
    
    #Reading outside temperature data
    df_outsideTemperatureData_original = pd.read_csv(os.path.join(config.DIR_OUTSIDE_TEMPERATURE_ONE_MINUTE_DAYS, 'Outside_Temperature_1Minute_Day' + str(currentDay) + '.csv'), sep =";")
    df_outsideTemperatureData_original['Time'] = pd.to_datetime(df_outsideTemperatureData_original['Time'], format = '%d.%m.%Y %H:%M')
    df_outsideTemperatureData = df_outsideTemperatureData_original.set_index('Time').resample(str(SetUpScenarios.timeResolution_InMinutes) +'Min').mean()
    df_outsideTemperatureData['Timeslot'] = arrayTimeSlots
    df_outsideTemperatureData = df_outsideTemperatureData.set_index('Timeslot')
    
    dictionaryTemperature_In_C= df_outsideTemperatureData['Temperature [C]'].to_dict()
    cop_heatPump_SpaceHeating, cop_heatPump_DHW = SetUpScenarios.calculateCOP(df_outsideTemperatureData["Temperature [C]"])

    #Create the price data
    df_priceData_original['Time'] = pd.to_datetime(df_priceData_original['Time'], format = '%d.%m.%Y %H:%M')
    df_priceData = df_priceData_original.set_index('Time').resample(str(SetUpScenarios.timeResolution_InMinutes) +'Min').mean()
    df_priceData['Timeslot'] = arrayTimeSlots
    df_priceData = df_priceData.set_index('Timeslot')

    dictionaryPrice_Cents= df_priceData['Price [Cent/kWh]'].to_dict()
    model.param_electricityPrice_In_Cents = pyo.Param(model.set_timeslots, initialize=dictionaryPrice_Cents)
    
    
    #Reading of the building data
    list_df_buildingData_BT1_original= [pd.read_csv(os.path.join(config.DIR_INPUT_DATA, "BT1_mHP_EV_SFH_1Minute_Days/HH" + str(index) + "/HH" + str(index) + "_Day" + str(currentDay) + ".csv"), sep =";") for index in indexOfBuildingsOverall_BT1]
    list_df_buildingData_BT2_original= [pd.read_csv(os.path.join(config.DIR_INPUT_DATA, "BT2_mHP_SFH_1Minute_Days/HH" + str(index) + "/HH" + str(index) + "_Day" + str(currentDay) + ".csv"), sep =";") for index in indexOfBuildingsOverall_BT2]
    list_df_buildingData_BT3_original= [pd.read_csv(os.path.join(config.DIR_INPUT_DATA, "BT3_EV_SFH_1Minute_Days/HH" + str(index) + "/HH" + str(index) + "_Day" + str(currentDay) + ".csv"), sep =";") for index in indexOfBuildingsOverall_BT3]
    list_df_buildingData_BT4_original= [pd.read_csv(os.path.join(config.DIR_INPUT_DATA, "BT4_mHP_MFH_1Minute_Days/HH" + str(index) + "/HH" + str(index) + "_Day" + str(currentDay) + ".csv"), sep =";") for index in indexOfBuildingsOverall_BT4]
    list_df_buildingData_BT5_original= [pd.read_csv(os.path.join(config.DIR_INPUT_DATA, "BT5_BAT_SFH_1Minute_Days/HH" + str(index) + "/HH" + str(index) + "_Day" + str(currentDay) + ".csv"), sep =";") for index in indexOfBuildingsOverall_BT5]
    list_df_buildingData_BT6_original= [pd.read_csv(os.path.join(config.DIR_INPUT_DATA, "BT6_mGas_mElement_SFH_1_Minute_Days/HH" + str(index) + "/HH" + str(index) + "_Day" + str(currentDay) + ".csv"), sep =";") for index in indexOfBuildingsOverall_BT6]
    list_df_buildingData_BT7_original= [pd.read_csv(os.path.join(config.DIR_INPUT_DATA, "BT7_mGas_Fan_SFH_1Minute_Days/HH" + str(index) + "/HH" + str(index) + "_Day" + str(currentDay) + ".csv"), sep =";") for index in indexOfBuildingsOverall_BT7]



    #Rename column 'Demand Electricity [W]' to 'Electricity [W]' if it exists
    for i in range (0, len(list_df_buildingData_BT1_original)):
        if 'Demand Electricity [W]' in list_df_buildingData_BT1_original[i]:
            list_df_buildingData_BT1_original[i].rename(columns={'Demand Electricity [W]': 'Electricity [W]'}, inplace=True)
    for i in range (0, len(list_df_buildingData_BT2_original)):
        if 'Demand Electricity [W]' in list_df_buildingData_BT2_original[i]:
            list_df_buildingData_BT2_original[i].rename(columns={'Demand Electricity [W]': 'Electricity [W]'}, inplace=True)
    for i in range (0, len(list_df_buildingData_BT3_original)):
        if 'Demand Electricity [W]' in list_df_buildingData_BT3_original[i]:
            list_df_buildingData_BT3_original[i].rename(columns={'Demand Electricity [W]': 'Electricity [W]'}, inplace=True)
    for i in range (0, len(list_df_buildingData_BT4_original)):
        if 'Demand Electricity [W]' in list_df_buildingData_BT4_original[i]:
            list_df_buildingData_BT4_original[i].rename(columns={'Demand Electricity [W]': 'Electricity [W]'}, inplace=True)
    for i in range (0, len(list_df_buildingData_BT5_original)):
        if 'Demand Electricity [W]' in list_df_buildingData_BT5_original[i]:
            list_df_buildingData_BT5_original[i].rename(columns={'Demand Electricity [W]': 'Electricity [W]'}, inplace=True)
    for i in range (0, len(list_df_buildingData_BT6_original)):
        if 'Demand Electricity [W]' in list_df_buildingData_BT6_original[i]:
            list_df_buildingData_BT6_original[i].rename(columns={'Demand Electricity [W]': 'Electricity [W]'}, inplace=True)
    for i in range (0, len(list_df_buildingData_BT7_original)):
        if 'Demand Electricity [W]' in list_df_buildingData_BT7_original[i]:
            list_df_buildingData_BT7_original[i].rename(columns={'Demand Electricity [W]': 'Electricity [W]'}, inplace=True)

            

    list_df_buildingData_BT1 = list_df_buildingData_BT1_original.copy()
    list_df_buildingData_BT2 = list_df_buildingData_BT2_original.copy() 
    list_df_buildingData_BT3 = list_df_buildingData_BT3_original.copy()
    list_df_buildingData_BT4 = list_df_buildingData_BT4_original.copy()
    list_df_buildingData_BT5 = list_df_buildingData_BT5_original.copy()
    list_df_buildingData_BT6 = list_df_buildingData_BT6_original.copy()
    list_df_buildingData_BT7 = list_df_buildingData_BT7_original.copy()
    

    
    

    
    
    #######################################################################################################################
    
    #Building Type 1 (BT1): Buildings with modulating air-source heat pump (mHP) and electric vehicle (EV)



    #Adjust dataframes to the current time resolution and set new index "Timeslot"
    
    for i in range (0, len(list_df_buildingData_BT1_original)):
        list_df_buildingData_BT1_original[i]['Time'] = pd.to_datetime(list_df_buildingData_BT1_original[i]['Time'], format = '%d.%m.%Y %H:%M')
        list_df_buildingData_BT1 [i] = list_df_buildingData_BT1_original[i].set_index('Time').resample(str(SetUpScenarios.timeResolution_InMinutes) +'Min').mean()
        for j in range (0, len(list_df_buildingData_BT1[i]['Availability of the EV'])):
            if list_df_buildingData_BT1 [i]['Availability of the EV'] [j] > 0.1:
                list_df_buildingData_BT1 [i]['Availability of the EV'] [j] = 1.0
            if list_df_buildingData_BT1 [i]['Availability of the EV'] [j] < 0.1 and list_df_buildingData_BT1 [i]['Availability of the EV'] [j] >0.01:
                list_df_buildingData_BT1 [i]['Availability of the EV'] [j] = 0
        
        arrayTimeSlots = [k for k in range (1,SetUpScenarios.numberOfTimeSlotsPerDay + 1)]
        list_df_buildingData_BT1 [i]['Timeslot'] = arrayTimeSlots
        list_df_buildingData_BT1 [i] = list_df_buildingData_BT1 [i].set_index('Timeslot')
        
        


    if SetUpScenarios.numberOfBuildings_BT1 >=1:
        #Create dataframes by using pandas series 
   
        list_windProfileNominal_BT1 = [SetUpScenarios.calculateAssignedWindPowerNominalPerBuilding (currentDay,index_BT1) for index_BT1 in range(0, SetUpScenarios.numberOfBuildings_BT1)]
        list_df_windPowerAssignedNominalPerBuilding_BT1 = [pd.DataFrame({'Timeslot': list_df_buildingData_BT1 [i].index, 'Wind [nominal]':list_windProfileNominal_BT1[i] }) for i in range (0, SetUpScenarios.numberOfBuildings_BT1)]

        for i in range (0, len(list_df_windPowerAssignedNominalPerBuilding_BT1)):
            del list_df_windPowerAssignedNominalPerBuilding_BT1[i]['Timeslot']
            list_df_windPowerAssignedNominalPerBuilding_BT1[i].index +=1   
            

            
        #Create availability array for the EV
        availabilityOfTheEVCombined = np.zeros((SetUpScenarios.numberOfBuildings_WithEV, SetUpScenarios.numberOfTimeSlotsPerDay))
        for index_BT1 in range (0, SetUpScenarios.numberOfBuildings_BT1):
            for index_timeslot_for_Availability in range (0,  SetUpScenarios.numberOfTimeSlotsPerDay):
                availabilityOfTheEVCombined [index_BT1,index_timeslot_for_Availability] = list_df_buildingData_BT1 [index_BT1]['Availability of the EV'] [index_timeslot_for_Availability +1]
        

        list_energyConsumptionOfEVs_Joule_BT1 = np.zeros((SetUpScenarios.numberOfBuildings_BT1, SetUpScenarios.numberOfTimeSlotsPerDay))
        
        
        for indexEV in range (0, SetUpScenarios.numberOfBuildings_BT1):
            list_energyConsumptionOfEVs_Joule_BT1[indexEV] = SetUpScenarios.generateEVEnergyConsumptionPatterns(availabilityOfTheEVCombined [indexEV],indexEV)
        
        list_df_energyConsumptionEV_Joule = [pd.DataFrame({'Timeslot': list_df_buildingData_BT1 [i].index, 'Energy':list_energyConsumptionOfEVs_Joule_BT1 [i] }) for i in range (0, SetUpScenarios.numberOfBuildings_BT1)]
        


        for i in range (0, len(list_energyConsumptionOfEVs_Joule_BT1)): 
            del list_df_energyConsumptionEV_Joule [i]['Timeslot']
            list_df_energyConsumptionEV_Joule[i].index +=1
            

        combinedDataframe_heatDemand_BT1 = pd.DataFrame()
        combinedDataframe_DHWDemand_BT1 = pd.DataFrame()
        combinedDataframe_electricalDemand_BT1 = pd.DataFrame()
        combinedDataframe_pvGenerationNominal_BT1 = pd.DataFrame()
        combinedDataframe_windAssigned_BT1 = pd.DataFrame()
        combinedDataframe_availabilityPatternEV_BT1 = pd.DataFrame()
        combinedDataframe_energyConsumptionEV_Joule_BT1 = pd.DataFrame()
        

        
        
        for index in range (0, len(list_df_buildingData_BT1)):
            combinedDataframe_heatDemand_BT1[index] = list_df_buildingData_BT1[index] ["Space Heating [W]"]
            combinedDataframe_DHWDemand_BT1[index] = list_df_buildingData_BT1[index] ["DHW [W]"]
            combinedDataframe_electricalDemand_BT1[index] = list_df_buildingData_BT1[index] ["Electricity [W]"]
            combinedDataframe_pvGenerationNominal_BT1[index] = list_df_buildingData_BT1[index] ["PV [nominal]"]
            combinedDataframe_windAssigned_BT1 [index]= list_df_windPowerAssignedNominalPerBuilding_BT1[index] ["Wind [nominal]"]
            combinedDataframe_availabilityPatternEV_BT1[index] = list_df_buildingData_BT1 [index]  ['Availability of the EV']
            combinedDataframe_energyConsumptionEV_Joule_BT1[index] = list_df_energyConsumptionEV_Joule[index] ["Energy"]
        


    
        #Round the values
        for index in range (0,  SetUpScenarios.numberOfBuildings_BT1):
            decimalsForRounding = 2    
            list_df_buildingData_BT1 [index]['Space Heating [W]'] = list_df_buildingData_BT1 [index]['Space Heating [W]'].apply(lambda x: round(x, decimalsForRounding))
            list_df_buildingData_BT1 [index]['DHW [W]'] = list_df_buildingData_BT1 [index]['DHW [W]'].apply(lambda x: round(x, decimalsForRounding))
            list_df_buildingData_BT1 [index]['Electricity [W]'] = list_df_buildingData_BT1 [index]['Electricity [W]'].apply(lambda x: round(x, decimalsForRounding))
            decimalsForRounding = 4 
            list_df_buildingData_BT1 [index]['PV [nominal]'] = list_df_buildingData_BT1 [index]['PV [nominal]'].apply(lambda x: round(x, decimalsForRounding))
    
    

        #Calculate the COPs for the heat hump
        df_copHeatPump =  pd.DataFrame({'Timeslot': list_df_buildingData_BT1 [0].index, 'COP_SpaceHeating':cop_heatPump_SpaceHeating, 'COP_DHW':cop_heatPump_DHW})
        
        dictionaryCOPHeatPump_SpaceHeating= df_copHeatPump.set_index('Timeslot')['COP_SpaceHeating'].to_dict()
        dictionaryCOPHeatPump_DHW= df_copHeatPump.set_index('Timeslot')['COP_DHW'].to_dict()
    
    
    
        #Define the parameters of the model in pyomo
        def init_heatDemand (model, i,j):
            return combinedDataframe_heatDemand_BT1.iloc[j-1, i-1]
        
        model.param_heatDemand_In_W_BT1 = pyo.Param(model.set_buildings_BT1, model.set_timeslots, mutable = True, initialize=init_heatDemand)
        
        
        def init_DHWDemand (model, i,j):
            return combinedDataframe_DHWDemand_BT1.iloc[j-1, i-1]
            
        
        model.param_DHWDemand_In_W_BT1 = pyo.Param(model.set_buildings_BT1, model.set_timeslots,mutable = True, initialize=init_DHWDemand)
        
        
        def init_electricalDemand (model, i,j):
            return combinedDataframe_electricalDemand_BT1.iloc[j-1, i-1]
        
        model.param_electricalDemand_In_W_BT1 = pyo.Param(model.set_buildings_BT1, model.set_timeslots,mutable = True, initialize=init_electricalDemand)
        
        
        def init_pvGenerationNominal (model, i,j):
            return combinedDataframe_pvGenerationNominal_BT1.iloc[j-1, i-1]
        
        model.param_pvGenerationNominal_BT1  = pyo.Param(model.set_buildings_BT1, model.set_timeslots, mutable = True, initialize=init_pvGenerationNominal)
        
        
        model.param_outSideTemperature_In_C = pyo.Param(model.set_timeslots, initialize=dictionaryTemperature_In_C)
        
        
        def init_windAssignedNominal (model, i,j):
            return combinedDataframe_windAssigned_BT1.iloc[j-1, i-1]
        
        model.param_windAssignedNominal_BT1  = pyo.Param(model.set_buildings_BT1, model.set_timeslots, mutable = True, initialize=init_windAssignedNominal)
        
        
        def init_availabilityPatternEV (model, i,j):
            return combinedDataframe_availabilityPatternEV_BT1.iloc[j-1, i-1]
        
        model.param_availabilityPerTimeSlotOfEV_BT1  = pyo.Param(model.set_buildings_BT1, model.set_timeslots, mutable = True, initialize=init_availabilityPatternEV)
        

        def init_energyConsumptionEV_Joule (model, i,j):
            return combinedDataframe_energyConsumptionEV_Joule_BT1.iloc[j-1, i-1]
        
        model.param_energyConsumptionEV_Joule_BT1  = pyo.Param(model.set_buildings_BT1, model.set_timeslots, mutable = True, initialize=init_energyConsumptionEV_Joule)
        
        
        model.param_COPHeatPump_SpaceHeating_BT1 = pyo.Param(model.set_timeslots, initialize=dictionaryCOPHeatPump_SpaceHeating)
        model.param_COPHeatPump_DHW_BT1 = pyo.Param(model.set_timeslots, initialize=dictionaryCOPHeatPump_DHW)


            
        
        #Define the variables
        
        model.variable_heatGenerationCoefficient_SpaceHeating_BT1 = pyo.Var(model.set_buildings_BT1, model.set_timeslots,bounds=(0,1))
        model.variable_heatGenerationCoefficient_DHW_BT1 = pyo.Var(model.set_buildings_BT1, model.set_timeslots,bounds=(0,1))
        model.variable_help_OnlyOneStorage_BT1 = pyo.Var(model.set_buildings_BT1, model.set_timeslots,  within=pyo.Binary)
        model.variable_temperatureBufferStorage_BT1 = pyo.Var(model.set_buildings_BT1, model.set_timeslots,  bounds=(SetUpScenarios.minimalBufferStorageTemperature  , SetUpScenarios.maximalBufferStorageTemperature ))
        model.variable_usableVolumeDHWTank_BT1 = pyo.Var(model.set_buildings_BT1, model.set_timeslots, bounds=(SetUpScenarios.minimumCapacityDHWTankOptimization, SetUpScenarios.maximumCapacityDHWTankOptimization))
        
        model.variable_currentChargingPowerEV_BT1 = pyo.Var(model.set_buildings_BT1, model.set_timeslots, within=pyo.NonNegativeReals, bounds=(0,SetUpScenarios.chargingPowerMaximal_EV))
        model.variable_energyLevelEV_BT1 = pyo.Var(model.set_buildings_BT1, model.set_timeslots, within=pyo.NonNegativeReals, bounds=(0, SetUpScenarios.capacityMaximal_EV))
        model.variable_SOC_EV_BT1= pyo.Var(model.set_buildings_BT1, model.set_timeslots,  within=pyo.NonNegativeReals, bounds=(0,100))
        model.variable_electricalPowerTotal_BT1 = pyo.Var(model.set_buildings_BT1, model.set_timeslots)
        model.variable_pvGeneration_BT1 = pyo.Var(model.set_buildings_BT1, model.set_timeslots)
        model.variable_windPowerAssigned_BT1 = pyo.Var(model.set_buildings_BT1, model.set_timeslots)

        model.variable_deviation_temperature_positive_BT1 = pyo.Var(model.set_buildings_BT1, model.set_timeslots, within=pyo.NonNegativeReals)
        model.variable_deviation_temperature_negative_BT1 = pyo.Var(model.set_buildings_BT1, model.set_timeslots, within=pyo.NonNegativeReals)
        
        
        # Defining the constraints 
        
        
        #Temperature constraint for the buffer storage (space heating) with energetic difference equation
        
        def temperatureBufferStorageConstraintRule_BT1(model, i, t):
            if t == model.set_timeslots.first():
                return model.variable_temperatureBufferStorage_BT1[i, t] == SetUpScenarios.initialBufferStorageTemperature + ((model.variable_heatGenerationCoefficient_SpaceHeating_BT1[i, t] * model.param_COPHeatPump_SpaceHeating_BT1[t] *  SetUpScenarios.electricalPower_HP * SetUpScenarios.timeResolution_InMinutes * 60  - model.param_heatDemand_In_W_BT1 [i, t]  * SetUpScenarios.timeResolution_InMinutes * 60 - SetUpScenarios.standingLossesBufferStorage * SetUpScenarios.timeResolution_InMinutes * 60) / (SetUpScenarios.capacityOfBufferStorage * SetUpScenarios.densityOfCement * SetUpScenarios.specificHeatCapacityOfCement))
            return model.variable_temperatureBufferStorage_BT1[i, t] == model.variable_temperatureBufferStorage_BT1[i, t-1] + ((model.variable_heatGenerationCoefficient_SpaceHeating_BT1[i, t] * model.param_COPHeatPump_SpaceHeating_BT1[t] *  SetUpScenarios.electricalPower_HP * SetUpScenarios.timeResolution_InMinutes * 60  - model.param_heatDemand_In_W_BT1 [i, t]  * SetUpScenarios.timeResolution_InMinutes * 60 - SetUpScenarios.standingLossesBufferStorage * SetUpScenarios.timeResolution_InMinutes * 60) / (SetUpScenarios.capacityOfBufferStorage * SetUpScenarios.densityOfCement * SetUpScenarios.specificHeatCapacityOfCement))
        
        model.constraint_temperatureBufferStorage_BT1= pyo.Constraint (model.set_buildings_BT1, model.set_timeslots, rule=temperatureBufferStorageConstraintRule_BT1)
        
        #Constraint for allowing thermal discomfort
        def temperatureDeviationConstraintRule1_BT1 (model, i, t):
            if optParameters['optimizationGoal_minimizeThermalDiscomfort'] == True:
                return  SetUpScenarios.idealComfortTemperature - SetUpScenarios.allowedTemperatureDeviationForOptimalComfort <= model.variable_temperatureBufferStorage_BT1[i, t] - model.variable_deviation_temperature_positive_BT1 [i, t] +  model.variable_deviation_temperature_negative_BT1 [i, t]
            else:
                return pyo.Constraint.Skip

        model.constraint_temperatureDeviationConstraint1_BT1 = pyo.Constraint(model.set_buildings_BT1, model.set_timeslots, rule = temperatureDeviationConstraintRule1_BT1)


        def temperatureDeviationConstraintRule2_BT1 (model, i, t):
            if optParameters['optimizationGoal_minimizeThermalDiscomfort'] == True:
                return  model.variable_temperatureBufferStorage_BT1[i, t] - model.variable_deviation_temperature_positive_BT1 [i, t] +  model.variable_deviation_temperature_negative_BT1 [i, t]<= SetUpScenarios.idealComfortTemperature + SetUpScenarios.allowedTemperatureDeviationForOptimalComfort
            else:
                return pyo.Constraint.Skip

        model.constraint_temperatureDeviationConstraint2_BT1 = pyo.Constraint(model.set_buildings_BT1, model.set_timeslots, rule = temperatureDeviationConstraintRule2_BT1)


        #Constraints for the minimal and maximal temperature at the end of the optimization horizon
        def temperatureBufferStorage_lastLowerLimitRule_BT1 (model, i, t):
            return model.variable_temperatureBufferStorage_BT1[i, model.set_timeslots.last()] >= SetUpScenarios.initialBufferStorageTemperature - SetUpScenarios.endBufferStorageTemperatureAllowedDeviationFromInitalValue
        
        model.constraint_temperatureBufferStorage_lastLowerLimit_BT1 = pyo.Constraint (model.set_buildings_BT1, model.set_timeslots, rule=temperatureBufferStorage_lastLowerLimitRule_BT1) 
        
        
        
        def temperatureBufferStorage_lastUpperLimitRule_BT1 (model, i, t):
            return model.variable_temperatureBufferStorage_BT1[i, model.set_timeslots.last()] <= SetUpScenarios.initialBufferStorageTemperature + SetUpScenarios.endBufferStorageTemperatureAllowedDeviationFromInitalValue
        
        model.constraint_temperatureBufferStorage_lastUpperLimit_BT1 = pyo.Constraint (model.set_buildings_BT1, model.set_timeslots, rule=temperatureBufferStorage_lastUpperLimitRule_BT1) 
        
        

        
        #Volume constraint for the DHW tank with energetic difference equation
        def volumeDHWTankConstraintRule_BT1(model, i, t):
            if t == model.set_timeslots.first():
                return model.variable_usableVolumeDHWTank_BT1[i, t] == SetUpScenarios.initialUsableVolumeDHWTank  + ((model.variable_heatGenerationCoefficient_DHW_BT1[i, t] * model.param_COPHeatPump_DHW_BT1[t] *  SetUpScenarios.electricalPower_HP * SetUpScenarios.timeResolution_InMinutes * 60  - model.param_DHWDemand_In_W_BT1 [i, t]  * SetUpScenarios.timeResolution_InMinutes * 60 - SetUpScenarios.standingLossesDHWTank * SetUpScenarios.timeResolution_InMinutes * 60) / (SetUpScenarios.temperatureOfTheHotWaterInTheDHWTank * SetUpScenarios.densityOfWater * SetUpScenarios.specificHeatCapacityOfWater))
            return model.variable_usableVolumeDHWTank_BT1[i, t] == model.variable_usableVolumeDHWTank_BT1[i, t-1] + ((model.variable_heatGenerationCoefficient_DHW_BT1[i, t] * model.param_COPHeatPump_DHW_BT1[t] *  SetUpScenarios.electricalPower_HP * SetUpScenarios.timeResolution_InMinutes * 60  - model.param_DHWDemand_In_W_BT1 [i, t]  * SetUpScenarios.timeResolution_InMinutes * 60 - SetUpScenarios.standingLossesDHWTank * SetUpScenarios.timeResolution_InMinutes * 60) / (SetUpScenarios.temperatureOfTheHotWaterInTheDHWTank * SetUpScenarios.densityOfWater * SetUpScenarios.specificHeatCapacityOfWater))
        
        model.constraint_temperatureDHWTank_BT1= pyo.Constraint (model.set_buildings_BT1, model.set_timeslots, rule=volumeDHWTankConstraintRule_BT1)
        
        
        #Constraints for the minimal and maximal temperature at the end of the optimization horizon
        def volumeDHWTank_lastLowerLimitRule_BT1 (model, i, t):
            return model.variable_usableVolumeDHWTank_BT1[i, model.set_timeslots.last()] >= SetUpScenarios.initialUsableVolumeDHWTank - SetUpScenarios.endUsableVolumeDHWTankAllowedDeviationFromInitialValue
        
        model.constraint_volumeDHWTank_lastLowerLimit_BT1 = pyo.Constraint (model.set_buildings_BT1, model.set_timeslots, rule=volumeDHWTank_lastLowerLimitRule_BT1) 
        
        
        
        def volumeDHWTank_lastUpperLimitRule_BT1 (model, i, t):
            return model.variable_usableVolumeDHWTank_BT1[i, model.set_timeslots.last()] <= SetUpScenarios.initialUsableVolumeDHWTank + SetUpScenarios.endUsableVolumeDHWTankAllowedDeviationFromInitialValue
        
        model.constraint_volumeDHWTank_lastUpperLimit_BT1 = pyo.Constraint (model.set_buildings_BT1, model.set_timeslots, rule=volumeDHWTank_lastUpperLimitRule_BT1) 
        
        
        
        #Constraints ensure that only one storage is heated up in each time slot
        def onlyOneStorageRule_1_BT1 (model, i, t):
            return model.variable_heatGenerationCoefficient_SpaceHeating_BT1 [i,t] <= model.variable_help_OnlyOneStorage_BT1 [i,t]
        
        def onlyOneStorageRule_2_BT1 (model, i, t):
            return model.variable_heatGenerationCoefficient_DHW_BT1 [i,t] <= (1 - model.variable_help_OnlyOneStorage_BT1 [i, t])
        
        model.constraint_onlyOneStorage_1_BT1 = pyo.Constraint(model.set_buildings_BT1, model.set_timeslots, rule =onlyOneStorageRule_1_BT1)
        model.constraint_onlyOneStorage_2_BT1 = pyo.Constraint(model.set_buildings_BT1, model.set_timeslots, rule =onlyOneStorageRule_2_BT1)
        
        
        #Constraint for minimal modulation degree if the heat pump is always switched on
        def minimalModulationDegreeOfTheHeatPumpRule_BT1 (model,i, t):
            if Run_Simulations.isHPAlwaysSwitchedOn ==True:
                return model.variable_heatGenerationCoefficient_SpaceHeating_BT1[i,t] +  model.variable_heatGenerationCoefficient_DHW_BT1[i, t] >= (SetUpScenarios.minimalModulationdDegree_HP/100)
            if Run_Simulations.isHPAlwaysSwitchedOn ==False:
                return pyo.Constraint.Feasible
                
            
        model.constraint_minimalModulationDegreeOfTheHeatPump_BT1 = pyo.Constraint(model.set_buildings_BT1, model.set_timeslots, rule = minimalModulationDegreeOfTheHeatPumpRule_BT1)
        
        
        
        #EV Energy Level
        def energyLevelOfEVRule_BT1 (model, i, t):
            if t == model.set_timeslots.first():
                return model.variable_energyLevelEV_BT1 [i, t] ==  ((SetUpScenarios.initialSOC_EV/100) * SetUpScenarios.capacityMaximal_EV) + (model.variable_currentChargingPowerEV_BT1 [i, t] * (SetUpScenarios.chargingEfficiency_EV/100) * SetUpScenarios.timeResolution_InMinutes * 60 - model.param_energyConsumptionEV_Joule_BT1 [i, t]) 
            return model.variable_energyLevelEV_BT1[i, t]  == model.variable_energyLevelEV_BT1 [i, t-1] + ( model.variable_currentChargingPowerEV_BT1 [i, t] * (SetUpScenarios.chargingEfficiency_EV/100) * SetUpScenarios.timeResolution_InMinutes * 60 - model.param_energyConsumptionEV_Joule_BT1 [i, t]) 
        
        model.constraint_energyLevelOfEV_BT1 = pyo.Constraint(model.set_buildings_BT1, model.set_timeslots, rule=energyLevelOfEVRule_BT1)
        
        
        #Constraints for the minimal and maximal energy level of the EV at the end of the optimization horizon
        def constraint_energyLevelOfEV_lastLowerLimitRule_BT1 (model, i, t):
            return model.variable_energyLevelEV_BT1[i, model.set_timeslots.last()] >= ((SetUpScenarios.initialSOC_EV - SetUpScenarios.endSOC_EVAllowedDeviationFromInitalValue)/100) * SetUpScenarios.capacityMaximal_EV
        
        model.constraint_energyLevelOfEV_lastLowerLimit_BT1 = pyo.Constraint (model.set_buildings_BT1, model.set_timeslots, rule=constraint_energyLevelOfEV_lastLowerLimitRule_BT1) 
        
        
        def constraint_energyLevelOfEV_lastUpperLimitRule_BT1 (model, i, t):
            return model.variable_energyLevelEV_BT1[i, model.set_timeslots.last()] <= ((SetUpScenarios.initialSOC_EV + SetUpScenarios.endSOC_EVAllowedDeviationFromInitalValue)/100) * SetUpScenarios.capacityMaximal_EV
        model.constraint_energyLevelOfEV_lastUpperLimit_BT1 = pyo.Constraint (model.set_buildings_BT1, model.set_timeslots, rule=constraint_energyLevelOfEV_lastUpperLimitRule_BT1)
        
        
        
        #SOC of the EV
        def socOfEVRule_BT1 (model, i, t):
            return model.variable_SOC_EV_BT1[i, t] == (model.variable_energyLevelEV_BT1 [i, t] / SetUpScenarios.capacityMaximal_EV)*100
        
        model.constraint_SOCofEV_BT1 =  pyo.Constraint(model.set_buildings_BT1, model.set_timeslots, rule = socOfEVRule_BT1)



        #Constraint for the charging power: The EV can only be charged if it is at home (available)
        def chargingPowerOfTheEVRul_BT1 (model, i, t):
            return model.variable_currentChargingPowerEV_BT1 [i, t] <=  model.param_availabilityPerTimeSlotOfEV_BT1 [i, t] * SetUpScenarios.chargingPowerMaximal_EV 
        
        model.constraint_chargingPowerOfTheEV_BT1 = pyo.Constraint(model.set_buildings_BT1, model.set_timeslots, rule = chargingPowerOfTheEVRul_BT1)
        
        
        
        #Constraints for the electrical power of BT1
        def electricalPowerTotalRule_BT1 (model,i, t):
            return model.variable_electricalPowerTotal_BT1 [i, t] == (model.variable_heatGenerationCoefficient_SpaceHeating_BT1 [i, t] + model.variable_heatGenerationCoefficient_DHW_BT1 [i, t] ) * SetUpScenarios.electricalPower_HP  + model.variable_currentChargingPowerEV_BT1 [i, t] + model.param_electricalDemand_In_W_BT1 [i, t] 
        
        model.constraint_electricalPowerTotal_BT1 = pyo.Constraint(model.set_buildings_BT1,model.set_timeslots, rule = electricalPowerTotalRule_BT1)
        
        
        
        #Equation for calculating the PV generation of each BT1-building
        def PVgenerationTotalRule_BT1 (model,i, t):
            
            return model.variable_pvGeneration_BT1 [i, t] == model.param_pvGenerationNominal_BT1 [i, t] * SetUpScenarios.determinePVPeakOfBuildings (i - 1)
        model.constraint_PVgenerationTotal_BT1 = pyo.Constraint(model.set_buildings_BT1, model.set_timeslots, rule = PVgenerationTotalRule_BT1)
        
        
        #Equation for calculating the assigned wind power for each BT1-building
        def windPowerAssignedgRule_BT1 (model,i, t):
            return model.variable_windPowerAssigned_BT1 [i, t] == model.param_windAssignedNominal_BT1 [i, t] * SetUpScenarios.maximalPowerOfWindTurbine 
            
        model.constraint_windPowerAssignedg_BT1 = pyo.Constraint(model.set_buildings_BT1,model.set_timeslots, rule = windPowerAssignedgRule_BT1)
        
        
        
        
        
        #Constraint system for the maximum number of starts of the heat pump
        
        model.variable_HPswitchedOff_Individual_SpaceHeating_BT1 = pyo.Var(model.set_buildings_BT1, model.set_timeslots, within =pyo.Binary, initialize=0.0)
        model.variable_HPswitchedOff_Individual_DHW_BT1 = pyo.Var(model.set_buildings_BT1, model.set_timeslots, within =pyo.Binary, initialize=0.0)
        model.variable_HPswitchedOff_Combined_BT1 = pyo.Var(model.set_buildings_BT1, model.set_timeslots, within =pyo.Binary)
        
        model.variable_HP_running_HelpAssociatedBinary_SpaceHeating_BT1 = pyo.Var(model.set_buildings_BT1, model.set_timeslots, within =pyo.Binary, initialize=0.0)
        model.variable_HP_running_HelpAssociatedBinary_DHW_BT1 = pyo.Var(model.set_buildings_BT1, model.set_timeslots, within =pyo.Binary, initialize=0.0)
        
        model.variable_HPswitchedOff_HelpModulationBinary_SpaceHeating_BT1 = pyo.Var(model.set_buildings_BT1, model.set_timeslots, within =pyo.Binary)
        model.variable_HPswitchedOff_HelpModulationBinary_DHW_BT1 = pyo.Var(model.set_buildings_BT1, model.set_timeslots, within =pyo.Binary)
        
        
        
        #Constraints for maximum number of starts for the space heating
        
        def maximumNumberOfStarts_Individual_SpaceHeating_EQ1_Rule_BT1 (model, i, t):
            if Run_Simulations.considerMaxiumNumberOfStartsHP_Individual ==True:
                if t == model.set_timeslots.first():
                    return model.variable_HPswitchedOff_Individual_SpaceHeating_BT1 [i, t] == 0
                return model.variable_HPswitchedOff_Individual_SpaceHeating_BT1 [i, t] <= model.variable_HP_running_HelpAssociatedBinary_SpaceHeating_BT1 [i, t-1]
            if Run_Simulations.considerMaxiumNumberOfStartsHP_Individual ==False: 
                return pyo.Constraint.Skip
                
        model.constraint_maximumNumberOfStarts_Individual_SpaceHeating_EQ1_BT1 = pyo.Constraint(model.set_buildings_BT1, model.set_timeslots, rule =maximumNumberOfStarts_Individual_SpaceHeating_EQ1_Rule_BT1)
        
        
        def maximumNumberOfStarts_Individual_SpaceHeating_EQ2_Rule_BT1 (model, i,  t):
            if Run_Simulations.considerMaxiumNumberOfStartsHP_Individual ==True:
                return model.variable_HP_running_HelpAssociatedBinary_SpaceHeating_BT1 [i, t] + model.variable_HPswitchedOff_Individual_SpaceHeating_BT1 [i, t] <= 1
            if Run_Simulations.considerMaxiumNumberOfStartsHP_Individual ==False: 
                return pyo.Constraint.Skip       
                
        model.constraint_maximumNumberOfStarts_Individual_SpaceHeating_EQ2_BT1 = pyo.Constraint(model.set_buildings_BT1, model.set_timeslots, rule =maximumNumberOfStarts_Individual_SpaceHeating_EQ2_Rule_BT1)
        
        
        
        def maximumNumberOfStarts_Individual_SpaceHeating_EQ2_2_Rule_BT1 (model, i, t):
            if Run_Simulations.considerMaxiumNumberOfStartsHP_Individual ==True:
                if t == model.set_timeslots.first():
                    return model.variable_HPswitchedOff_Individual_SpaceHeating_BT1 [i, t] == 0
                return model.variable_HP_running_HelpAssociatedBinary_SpaceHeating_BT1 [i, t - 1] <= model.variable_HP_running_HelpAssociatedBinary_SpaceHeating_BT1 [i, t] + model.variable_HPswitchedOff_Individual_SpaceHeating_BT1 [i, t]
            if Run_Simulations.considerMaxiumNumberOfStartsHP_Individual ==False: 
                return pyo.Constraint.Skip       
                
        model.constraint_maximumNumberOfStarts_Individual_SpaceHeating_EQ2_2_BT1 = pyo.Constraint(model.set_buildings_BT1, model.set_timeslots, rule =maximumNumberOfStarts_Individual_SpaceHeating_EQ2_2_Rule_BT1)
        
        
        
        def maximumNumberOfStarts_Individual_SpaceHeating_EQ3_HelpAssociatedBinary_Rule_BT1 (model, i, t):
            if Run_Simulations.considerMaxiumNumberOfStartsHP_Individual ==True or Run_Simulations.considerMaximumNumberOfStartsHP_Combined ==True:
                return model.variable_HP_running_HelpAssociatedBinary_SpaceHeating_BT1 [i, t] >= model.variable_heatGenerationCoefficient_SpaceHeating_BT1[i, t]
            if Run_Simulations.considerMaxiumNumberOfStartsHP_Individual ==False: 
               return pyo.Constraint.Skip
        
        model.constraint_maximumNumberOfStarts_Individual_SpaceHeating_EQ3_HelpAssociatedBinary_BT1 = pyo.Constraint(model.set_buildings_BT1, model.set_timeslots, rule =maximumNumberOfStarts_Individual_SpaceHeating_EQ3_HelpAssociatedBinary_Rule_BT1)
        
        
        
        def maximumNumberOfStarts_Individual_SpaceHeating_EQ4_HelpAssociatedBinary_Rule_BT1 (model, i, t):
            if Run_Simulations.considerMaxiumNumberOfStartsHP_Individual ==True or Run_Simulations.considerMaximumNumberOfStartsHP_Combined ==True:
                return model.variable_heatGenerationCoefficient_SpaceHeating_BT1[i, t] * (1/(SetUpScenarios.minimalModulationdDegree_HP/100))  >= model.variable_HP_running_HelpAssociatedBinary_SpaceHeating_BT1 [i, t]
            if Run_Simulations.considerMaxiumNumberOfStartsHP_Individual ==False: 
               return pyo.Constraint.Skip
        
        model.constraint_maximumNumberOfStarts_Individual_SpaceHeating_EQ4_HelpAssociatedBinary_BT1 = pyo.Constraint(model.set_buildings_BT1, model.set_timeslots, rule =maximumNumberOfStarts_Individual_SpaceHeating_EQ4_HelpAssociatedBinary_Rule_BT1)
        
        
        
        def maximumNumberOfStarts_Individual_SpaceHeating_EQ5_NumberOfStarts_Rule_BT1 (model, i, t):
            if Run_Simulations.considerMaxiumNumberOfStartsHP_Individual ==True:
                return  sum (model.variable_HPswitchedOff_Individual_SpaceHeating_BT1 [i, t] for t in model.set_timeslots)<= Run_Simulations.maximumNumberOfStarts_Individual 
            if Run_Simulations.considerMaxiumNumberOfStartsHP_Individual ==False: 
                return pyo.Constraint.Skip
        
        model.constraint_maximumNumberOfStarts_Individual_SpaceHeating_EQ5_NumberOfStarts_BT1 = pyo.Constraint(model.set_buildings_BT1, model.set_timeslots, rule =maximumNumberOfStarts_Individual_SpaceHeating_EQ5_NumberOfStarts_Rule_BT1)
        
        
        
        
        #Constraints for maximum number of starts for the DHW
        
        def maximumNumberOfStarts_Individual_DHW_EQ1_Rule_BT1 (model, i, t):
            if Run_Simulations.considerMaxiumNumberOfStartsHP_Individual ==True:
                if t == model.set_timeslots.first():
                    return model.variable_HPswitchedOff_Individual_DHW_BT1 [i, t] == 0
                return model.variable_HPswitchedOff_Individual_DHW_BT1 [i, t] <= model.variable_HP_running_HelpAssociatedBinary_DHW_BT1 [i, t-1]
            if Run_Simulations.considerMaxiumNumberOfStartsHP_Individual ==False: 
                return pyo.Constraint.Skip
                
        model.constraint_maximumNumberOfStarts_Individual_DHW_EQ1_BT1 = pyo.Constraint(model.set_buildings_BT1, model.set_timeslots, rule =maximumNumberOfStarts_Individual_DHW_EQ1_Rule_BT1)
        
        
        def maximumNumberOfStarts_Individual_DHW_EQ2_Rule_BT1 (model, i,  t):
            if Run_Simulations.considerMaxiumNumberOfStartsHP_Individual ==True:
                return model.variable_HP_running_HelpAssociatedBinary_DHW_BT1 [i, t] + model.variable_HPswitchedOff_Individual_DHW_BT1 [i, t] <= 1
            if Run_Simulations.considerMaxiumNumberOfStartsHP_Individual ==False: 
                return pyo.Constraint.Skip       
                
        model.constraint_maximumNumberOfStarts_Individual_DHW_EQ2_BT1 = pyo.Constraint(model.set_buildings_BT1, model.set_timeslots, rule =maximumNumberOfStarts_Individual_DHW_EQ2_Rule_BT1)
        
        
        
        def maximumNumberOfStarts_Individual_DHW_EQ2_2_Rule_BT1 (model, i, t):
            if Run_Simulations.considerMaxiumNumberOfStartsHP_Individual ==True:
                if t == model.set_timeslots.first():
                    return model.variable_HPswitchedOff_Individual_DHW_BT1 [i, t] == 0
                return model.variable_HP_running_HelpAssociatedBinary_DHW_BT1 [i, t - 1] <= model.variable_HP_running_HelpAssociatedBinary_DHW_BT1 [i, t] + model.variable_HPswitchedOff_Individual_DHW_BT1 [i, t]
            if Run_Simulations.considerMaxiumNumberOfStartsHP_Individual ==False: 
                return pyo.Constraint.Skip       
                
        model.constraint_maximumNumberOfStarts_Individual_DHW_EQ2_2_BT1 = pyo.Constraint(model.set_buildings_BT1, model.set_timeslots, rule =maximumNumberOfStarts_Individual_DHW_EQ2_2_Rule_BT1)
        
        
        
        def maximumNumberOfStarts_Individual_DHW_EQ3_HelpAssociatedBinary_Rule_BT1 (model, i, t):
            if Run_Simulations.considerMaxiumNumberOfStartsHP_Individual ==True or Run_Simulations.considerMaximumNumberOfStartsHP_Combined ==True:
                return model.variable_HP_running_HelpAssociatedBinary_DHW_BT1 [i, t] >= model.variable_heatGenerationCoefficient_DHW_BT1[i, t]
            if Run_Simulations.considerMaxiumNumberOfStartsHP_Individual ==False: 
               return pyo.Constraint.Skip
        
        model.constraint_maximumNumberOfStarts_Individual_DHW_EQ3_HelpAssociatedBinary_BT1 = pyo.Constraint(model.set_buildings_BT1, model.set_timeslots, rule =maximumNumberOfStarts_Individual_DHW_EQ3_HelpAssociatedBinary_Rule_BT1)
        
        
        
        def maximumNumberOfStarts_Individual_DHW_EQ4_HelpAssociatedBinary_Rule_BT1 (model, i, t):
            if Run_Simulations.considerMaxiumNumberOfStartsHP_Individual ==True or Run_Simulations.considerMaximumNumberOfStartsHP_Combined ==True:
                return model.variable_heatGenerationCoefficient_DHW_BT1[i, t] * (1/(SetUpScenarios.minimalModulationdDegree_HP/100))  >= model.variable_HP_running_HelpAssociatedBinary_DHW_BT1 [i, t]
            if Run_Simulations.considerMaxiumNumberOfStartsHP_Individual ==False: 
               return pyo.Constraint.Skip
        
        model.constraint_maximumNumberOfStarts_Individual_DHW_EQ4_HelpAssociatedBinary_BT1 = pyo.Constraint(model.set_buildings_BT1, model.set_timeslots, rule =maximumNumberOfStarts_Individual_DHW_EQ4_HelpAssociatedBinary_Rule_BT1)
        
        
        
        def maximumNumberOfStarts_Individual_DHW_EQ5_NumberOfStarts_Rule_BT1 (model, i, t):
            if Run_Simulations.considerMaxiumNumberOfStartsHP_Individual ==True:
                return  sum (model.variable_HPswitchedOff_Individual_DHW_BT1 [i, t] for t in model.set_timeslots)<= Run_Simulations.maximumNumberOfStarts_Individual 
            if Run_Simulations.considerMaxiumNumberOfStartsHP_Individual ==False: 
                return pyo.Constraint.Skip
        
        model.constraint_maximumNumberOfStarts_Individual_DHW_EQ5_NumberOfStarts_BT1 = pyo.Constraint(model.set_buildings_BT1, model.set_timeslots, rule =maximumNumberOfStarts_Individual_DHW_EQ5_NumberOfStarts_Rule_BT1)
        
        
        
        # Constraints for the maximum number of starts combined
        
        def maximumNumberOfStarts_Combined_EQ1_Rule_BT1 (model, i, t):
            if Run_Simulations.considerMaximumNumberOfStartsHP_Combined ==True:
                if t == model.set_timeslots.first():
                    return model.variable_HPswitchedOff_Combined_BT1 [i, t]== 0
                return model.variable_HP_running_HelpAssociatedBinary_SpaceHeating_BT1 [i, t-1] + model.variable_HP_running_HelpAssociatedBinary_DHW_BT1 [i, t-1] <= model.variable_HP_running_HelpAssociatedBinary_SpaceHeating_BT1 [i, t] + model.variable_HP_running_HelpAssociatedBinary_DHW_BT1 [i, t] + model.variable_HPswitchedOff_Combined_BT1 [i, t]
            if Run_Simulations.considerMaximumNumberOfStartsHP_Combined ==False: 
                return pyo.Constraint.Skip
            
        model.constraint_maximumNumberOfStarts_Combined_EQ1_BT1 = pyo.Constraint(model.set_buildings_BT1, model.set_timeslots, rule =maximumNumberOfStarts_Combined_EQ1_Rule_BT1)
        
        
        
        def maximumNumberOfStarts_Combined_EQ2_Rule_BT1 (model, i, t):
            if Run_Simulations.considerMaximumNumberOfStartsHP_Combined ==True:
                if t == model.set_timeslots.first():
                    return model.variable_HPswitchedOff_Combined_BT1 [i, t]== 0
                return model.variable_HP_running_HelpAssociatedBinary_SpaceHeating_BT1 [i, t] + model.variable_HP_running_HelpAssociatedBinary_DHW_BT1 [i, t] <= 1
            if Run_Simulations.considerMaximumNumberOfStartsHP_Combined ==False: 
               return  pyo.Constraint.Skip
            
        model.constraint_maximumNumberOfStarts_Combined_EQ2_BT1 = pyo.Constraint(model.set_buildings_BT1, model.set_timeslots, rule =maximumNumberOfStarts_Combined_EQ2_Rule_BT1)
        
        
        
        def maximumNumberOfStarts_Combined_EQ3_NumberOfStarts_Rule_BT1 (model, i, t):
            if Run_Simulations.considerMaximumNumberOfStartsHP_Combined ==True:
                return  sum (model.variable_HPswitchedOff_Combined_BT1 [i, t] for t in model.set_timeslots)<= Run_Simulations.maximumNumberOfStarts_Combined
            if Run_Simulations.considerMaximumNumberOfStartsHP_Combined ==False: 
                return pyo.Constraint.Skip
        
        model.maximumNumberOfStarts_Combined_EQ3_NumberOfStarts_BT1 = pyo.Constraint(model.set_buildings_BT1, model.set_timeslots, rule =maximumNumberOfStarts_Combined_EQ3_NumberOfStarts_Rule_BT1)
        
    
    
    
    
    
    
    ###############################################################################################################################
    
    
    #Building Type 2 (BT2): Buildings with modulating air-source heat pump (mHP) 
    
    
    #Adjust dataframes to the current time resolution and set new index "Timeslot"
    
    for i in range (0, len(list_df_buildingData_BT2_original)):
        list_df_buildingData_BT2_original[i]['Time'] = pd.to_datetime(list_df_buildingData_BT2_original[i]['Time'], format = '%d.%m.%Y %H:%M')
        list_df_buildingData_BT2 [i] = list_df_buildingData_BT2_original[i].set_index('Time').resample(str(SetUpScenarios.timeResolution_InMinutes) +'Min').mean()

        arrayTimeSlots = [k for k in range (1,SetUpScenarios.numberOfTimeSlotsPerDay + 1)]
        list_df_buildingData_BT2 [i]['Timeslot'] = arrayTimeSlots
        list_df_buildingData_BT2 [i] = list_df_buildingData_BT2 [i].set_index('Timeslot')
        
        

    if SetUpScenarios.numberOfBuildings_BT2 >=1:
        #Create dataframes by using pandas series 
   
        list_windProfileNominal_BT2 = [SetUpScenarios.calculateAssignedWindPowerNominalPerBuilding (currentDay,SetUpScenarios.numberOfBuildings_BT1 + index_BT2) for index_BT2 in range(0, SetUpScenarios.numberOfBuildings_BT2)]
        list_df_windPowerAssignedNominalPerBuilding_BT2 = [pd.DataFrame({'Timeslot': list_df_buildingData_BT2 [i].index, 'Wind [nominal]':list_windProfileNominal_BT2[i] }) for i in range (0, SetUpScenarios.numberOfBuildings_BT2)]

        for i in range (0, len(list_df_windPowerAssignedNominalPerBuilding_BT2)):
            del list_df_windPowerAssignedNominalPerBuilding_BT2[i]['Timeslot']
            list_df_windPowerAssignedNominalPerBuilding_BT2[i].index +=1   
       
        combinedDataframe_heatDemand_BT2 = pd.DataFrame()
        combinedDataframe_DHWDemand_BT2 = pd.DataFrame()
        combinedDataframe_electricalDemand_BT2 = pd.DataFrame()
        combinedDataframe_pvGenerationNominal_BT2 = pd.DataFrame()
        combinedDataframe_windAssigned_BT2 = pd.DataFrame()
        

        
        for index in range (0, len(list_df_buildingData_BT2)):
            combinedDataframe_heatDemand_BT2[index] = list_df_buildingData_BT2[index] ["Space Heating [W]"]
            combinedDataframe_DHWDemand_BT2[index] = list_df_buildingData_BT2[index] ["DHW [W]"]
            combinedDataframe_electricalDemand_BT2[index] = list_df_buildingData_BT2[index] ["Electricity [W]"]
            combinedDataframe_pvGenerationNominal_BT2[index] = list_df_buildingData_BT2[index] ["PV [nominal]"]
            combinedDataframe_windAssigned_BT2 [index]= list_df_windPowerAssignedNominalPerBuilding_BT2[index] ["Wind [nominal]"]


    
        #Round the values
        for index in range (0,  SetUpScenarios.numberOfBuildings_BT2):
            decimalsForRounding = 2    
            list_df_buildingData_BT2 [index]['Space Heating [W]'] = list_df_buildingData_BT2 [index]['Space Heating [W]'].apply(lambda x: round(x, decimalsForRounding))
            list_df_buildingData_BT2 [index]['DHW [W]'] = list_df_buildingData_BT2 [index]['DHW [W]'].apply(lambda x: round(x, decimalsForRounding))
            list_df_buildingData_BT2 [index]['Electricity [W]'] = list_df_buildingData_BT2 [index]['Electricity [W]'].apply(lambda x: round(x, decimalsForRounding))
            decimalsForRounding = 4 
            list_df_buildingData_BT2 [index]['PV [nominal]'] = list_df_buildingData_BT2 [index]['PV [nominal]'].apply(lambda x: round(x, decimalsForRounding))
    
    

        #Calculate the COPs for the heat hump
        df_copHeatPump =  pd.DataFrame({'Timeslot': list_df_buildingData_BT2 [0].index, 'COP_SpaceHeating':cop_heatPump_SpaceHeating, 'COP_DHW':cop_heatPump_DHW})
        
        dictionaryCOPHeatPump_SpaceHeating= df_copHeatPump.set_index('Timeslot')['COP_SpaceHeating'].to_dict()
        dictionaryCOPHeatPump_DHW= df_copHeatPump.set_index('Timeslot')['COP_DHW'].to_dict()
        
        
        #Define the parameters of the model in pyomo
        def init_heatDemand (model, i,j):
            return combinedDataframe_heatDemand_BT2.iloc[j-1, i-1]
        
        model.param_heatDemand_In_W_BT2 = pyo.Param(model.set_buildings_BT2, model.set_timeslots, mutable = True, initialize=init_heatDemand)
        
        
        def init_DHWDemand (model, i,j):
            return combinedDataframe_DHWDemand_BT2.iloc[j-1, i-1]
        
        model.param_DHWDemand_In_W_BT2 = pyo.Param(model.set_buildings_BT2, model.set_timeslots,mutable = True, initialize=init_DHWDemand)
        
        
        def init_electricalDemand (model, i,j):
            return combinedDataframe_electricalDemand_BT2.iloc[j-1, i-1]
        
        model.param_electricalDemand_In_W_BT2 = pyo.Param(model.set_buildings_BT2, model.set_timeslots,mutable = True, initialize=init_electricalDemand)
        
        
        def init_pvGenerationNominal (model, i,j):
            return combinedDataframe_pvGenerationNominal_BT2.iloc[j-1, i-1]
        
        model.param_pvGenerationNominal_BT2  = pyo.Param(model.set_buildings_BT2, model.set_timeslots, mutable = True, initialize=init_pvGenerationNominal)
        
        
        model.param_outSideTemperature_In_C = pyo.Param(model.set_timeslots, initialize=dictionaryTemperature_In_C)
        
        
        def init_windAssignedNominal (model, i,j):
            return combinedDataframe_windAssigned_BT2.iloc[j-1, i-1]
        
        model.param_windAssignedNominal_BT2  = pyo.Param(model.set_buildings_BT2, model.set_timeslots, mutable = True, initialize=init_windAssignedNominal)
        
        
        
        model.param_COPHeatPump_SpaceHeating_BT2 = pyo.Param(model.set_timeslots, initialize=dictionaryCOPHeatPump_SpaceHeating)
        model.param_COPHeatPump_DHW_BT2 = pyo.Param(model.set_timeslots, initialize=dictionaryCOPHeatPump_DHW)
        
        
        
        
        #Define the variables
        
        model.variable_heatGenerationCoefficient_SpaceHeating_BT2 = pyo.Var(model.set_buildings_BT2, model.set_timeslots,bounds=(0,1))
        model.variable_heatGenerationCoefficient_DHW_BT2 = pyo.Var(model.set_buildings_BT2, model.set_timeslots,bounds=(0,1))
        model.variable_help_OnlyOneStorage_BT2 = pyo.Var(model.set_buildings_BT2, model.set_timeslots,  within=pyo.Binary)
        model.variable_temperatureBufferStorage_BT2 = pyo.Var(model.set_buildings_BT2, model.set_timeslots,  bounds=(SetUpScenarios.minimalBufferStorageTemperature , SetUpScenarios.maximalBufferStorageTemperature))
        model.variable_usableVolumeDHWTank_BT2 = pyo.Var(model.set_buildings_BT2, model.set_timeslots, bounds=(SetUpScenarios.minimumCapacityDHWTankOptimization, SetUpScenarios.maximumCapacityDHWTankOptimization))
        
        model.variable_electricalPowerTotal_BT2 = pyo.Var(model.set_buildings_BT2, model.set_timeslots)
        model.variable_pvGeneration_BT2 = pyo.Var(model.set_buildings_BT2, model.set_timeslots)
        model.variable_windPowerAssigned_BT2 = pyo.Var(model.set_buildings_BT2, model.set_timeslots)
        
        model.variable_deviation_temperature_positive_BT2 = pyo.Var(model.set_buildings_BT2, model.set_timeslots, within=pyo.NonNegativeReals)
        model.variable_deviation_temperature_negative_BT2 = pyo.Var(model.set_buildings_BT2, model.set_timeslots, within=pyo.NonNegativeReals)
        
        
        # Defining the constraints 
        
        
        #Temperature constraint for the buffer storage (space heating) with energetic difference equation
        def temperatureBufferStorageConstraintRule_BT2(model, i, t):
            if t == model.set_timeslots.first():
                return model.variable_temperatureBufferStorage_BT2[i, t] == SetUpScenarios.initialBufferStorageTemperature + ((model.variable_heatGenerationCoefficient_SpaceHeating_BT2[i, t] * model.param_COPHeatPump_SpaceHeating_BT2[t] *  SetUpScenarios.electricalPower_HP * SetUpScenarios.timeResolution_InMinutes * 60  - model.param_heatDemand_In_W_BT2 [i, t]  * SetUpScenarios.timeResolution_InMinutes * 60 - SetUpScenarios.standingLossesBufferStorage * SetUpScenarios.timeResolution_InMinutes * 60) / (SetUpScenarios.capacityOfBufferStorage * SetUpScenarios.densityOfCement * SetUpScenarios.specificHeatCapacityOfCement))
            return model.variable_temperatureBufferStorage_BT2[i, t] == model.variable_temperatureBufferStorage_BT2[i, t-1] + ((model.variable_heatGenerationCoefficient_SpaceHeating_BT2[i, t] * model.param_COPHeatPump_SpaceHeating_BT2[t] *  SetUpScenarios.electricalPower_HP * SetUpScenarios.timeResolution_InMinutes * 60  - model.param_heatDemand_In_W_BT2 [i, t]  * SetUpScenarios.timeResolution_InMinutes * 60 - SetUpScenarios.standingLossesBufferStorage * SetUpScenarios.timeResolution_InMinutes * 60) / (SetUpScenarios.capacityOfBufferStorage * SetUpScenarios.densityOfCement * SetUpScenarios.specificHeatCapacityOfCement))
        
        model.constraint_temperatureBufferStorage_BT2= pyo.Constraint (model.set_buildings_BT2, model.set_timeslots, rule=temperatureBufferStorageConstraintRule_BT2)
        
        
        #Constraints for the minimal and maximal temperature at the end of the optimization horizon
        def temperatureBufferStorage_lastLowerLimitRule_BT2 (model, i, t):
            return model.variable_temperatureBufferStorage_BT2[i, model.set_timeslots.last()] >= SetUpScenarios.initialBufferStorageTemperature - SetUpScenarios.endBufferStorageTemperatureAllowedDeviationFromInitalValue
        
        model.constraint_temperatureBufferStorage_lastLowerLimit_BT2 = pyo.Constraint (model.set_buildings_BT2, model.set_timeslots, rule=temperatureBufferStorage_lastLowerLimitRule_BT2) 
        
        
        
        def temperatureBufferStorage_lastUpperLimitRule_BT2 (model, i, t):
            return model.variable_temperatureBufferStorage_BT2[i, model.set_timeslots.last()] <= SetUpScenarios.initialBufferStorageTemperature + SetUpScenarios.endBufferStorageTemperatureAllowedDeviationFromInitalValue
        
        model.constraint_temperatureBufferStorage_lastUpperLimit_BT2 = pyo.Constraint (model.set_buildings_BT2, model.set_timeslots, rule=temperatureBufferStorage_lastUpperLimitRule_BT2) 
        
        
        #Constraint for allowing thermal discomfort
        def temperatureDeviationConstraintRule1_BT2 (model, i, t):
            if optParameters['optimizationGoal_minimizeThermalDiscomfort'] == True:
                return  SetUpScenarios.idealComfortTemperature - SetUpScenarios.allowedTemperatureDeviationForOptimalComfort <= model.variable_temperatureBufferStorage_BT2[i, t] - model.variable_deviation_temperature_positive_BT2 [i, t] +  model.variable_deviation_temperature_negative_BT2 [i, t]
            else:
                return pyo.Constraint.Skip

        model.constraint_temperatureDeviationConstraint1_BT2 = pyo.Constraint(model.set_buildings_BT2, model.set_timeslots, rule = temperatureDeviationConstraintRule1_BT2)


        def temperatureDeviationConstraintRule2_BT2 (model, i, t):
            if optParameters['optimizationGoal_minimizeThermalDiscomfort'] == True:
                return  model.variable_temperatureBufferStorage_BT2[i, t] - model.variable_deviation_temperature_positive_BT2 [i, t] +  model.variable_deviation_temperature_negative_BT2 [i, t]<= SetUpScenarios.idealComfortTemperature + SetUpScenarios.allowedTemperatureDeviationForOptimalComfort
            else:
                return pyo.Constraint.Skip

        model.constraint_temperatureDeviationConstraint2_BT2 = pyo.Constraint(model.set_buildings_BT2, model.set_timeslots, rule = temperatureDeviationConstraintRule2_BT2)
        
        
        #Volume constraint for the DHW tank with energetic difference equation
        def volumeDHWTankConstraintRule_BT2(model, i, t):
            if t == model.set_timeslots.first():
                return model.variable_usableVolumeDHWTank_BT2[i, t] == SetUpScenarios.initialUsableVolumeDHWTank + ((model.variable_heatGenerationCoefficient_DHW_BT2[i, t] * model.param_COPHeatPump_DHW_BT2[t] *  SetUpScenarios.electricalPower_HP * SetUpScenarios.timeResolution_InMinutes * 60  - model.param_DHWDemand_In_W_BT2 [i, t]  * SetUpScenarios.timeResolution_InMinutes * 60 - SetUpScenarios.standingLossesDHWTank * SetUpScenarios.timeResolution_InMinutes * 60) / (SetUpScenarios.temperatureOfTheHotWaterInTheDHWTank * SetUpScenarios.densityOfWater * SetUpScenarios.specificHeatCapacityOfWater))
            return model.variable_usableVolumeDHWTank_BT2[i, t] == model.variable_usableVolumeDHWTank_BT2[i, t-1] + ((model.variable_heatGenerationCoefficient_DHW_BT2[i, t] * model.param_COPHeatPump_DHW_BT2[t] *  SetUpScenarios.electricalPower_HP * SetUpScenarios.timeResolution_InMinutes * 60  - model.param_DHWDemand_In_W_BT2 [i, t]  * SetUpScenarios.timeResolution_InMinutes * 60 - SetUpScenarios.standingLossesDHWTank * SetUpScenarios.timeResolution_InMinutes * 60) / (SetUpScenarios.temperatureOfTheHotWaterInTheDHWTank * SetUpScenarios.densityOfWater * SetUpScenarios.specificHeatCapacityOfWater))
        
        model.constraint_temperatureDHWTank_BT2= pyo.Constraint (model.set_buildings_BT2, model.set_timeslots, rule=volumeDHWTankConstraintRule_BT2)
        
        
        #Constraints for the minimal and maximal temperature at the end of the optimization horizon
        def volumeDHWTank_lastLowerLimitRule_BT2 (model, i, t):
            return model.variable_usableVolumeDHWTank_BT2[i, model.set_timeslots.last()] >= SetUpScenarios.initialUsableVolumeDHWTank - SetUpScenarios.endUsableVolumeDHWTankAllowedDeviationFromInitialValue
        
        model.constraint_volumeDHWTank_lastLowerLimit_BT2 = pyo.Constraint (model.set_buildings_BT2, model.set_timeslots, rule=volumeDHWTank_lastLowerLimitRule_BT2) 
        
        
        
        def volumeDHWTank_lastUpperLimitRule_BT2 (model, i, t):
            return model.variable_usableVolumeDHWTank_BT2[i, model.set_timeslots.last()] <= SetUpScenarios.initialUsableVolumeDHWTank + SetUpScenarios.endUsableVolumeDHWTankAllowedDeviationFromInitialValue
        
        model.constraint_volumeDHWTank_lastUpperLimit_BT2 = pyo.Constraint (model.set_buildings_BT2, model.set_timeslots, rule=volumeDHWTank_lastUpperLimitRule_BT2) 
        
        
        
        #Constraints ensure that only one storage is heated up in each time slot
        def onlyOneStorageRule_1_BT2 (model, i, t):
            return model.variable_heatGenerationCoefficient_SpaceHeating_BT2 [i,t] <= model.variable_help_OnlyOneStorage_BT2 [i,t]
        
        def onlyOneStorageRule_2_BT2 (model, i, t):
            return model.variable_heatGenerationCoefficient_DHW_BT2 [i,t] <= (1 - model.variable_help_OnlyOneStorage_BT2 [i, t])
        
        model.constraint_onlyOneStorage_1_BT2 = pyo.Constraint(model.set_buildings_BT2, model.set_timeslots, rule =onlyOneStorageRule_1_BT2)
        model.constraint_onlyOneStorage_2_BT2 = pyo.Constraint(model.set_buildings_BT2, model.set_timeslots, rule =onlyOneStorageRule_2_BT2)
        
        
        #Constraint for minimal modulation degree if the heat pump is always switched on
        def minimalModulationDegreeOfTheHeatPumpRule_BT2 (model,i, t):
            if Run_Simulations.isHPAlwaysSwitchedOn ==True:
                return model.variable_heatGenerationCoefficient_SpaceHeating_BT2[i,t] +  model.variable_heatGenerationCoefficient_DHW_BT2[i, t] >= (SetUpScenarios.minimalModulationdDegree_HP/100)
            if Run_Simulations.isHPAlwaysSwitchedOn ==False:
                return pyo.Constraint.Feasible
                
            
        model.constraint_minimalModulationDegreeOfTheHeatPump_BT2 = pyo.Constraint(model.set_buildings_BT2, model.set_timeslots, rule = minimalModulationDegreeOfTheHeatPumpRule_BT2)
        
        
        #Constraints for the electrical power of BT2
        def electricalPowerTotalRule_BT2 (model,i, t):
            return model.variable_electricalPowerTotal_BT2 [i, t] == (model.variable_heatGenerationCoefficient_SpaceHeating_BT2 [i, t] + model.variable_heatGenerationCoefficient_DHW_BT2 [i, t] ) * SetUpScenarios.electricalPower_HP  + model.param_electricalDemand_In_W_BT2 [i, t] 
        
        model.constraint_electricalPowerTotal_BT2 = pyo.Constraint(model.set_buildings_BT2,model.set_timeslots, rule = electricalPowerTotalRule_BT2)
        
        
        
        #Equation for calculating the PV generation of each BT2-building
        def PVgenerationTotalRule_BT2 (model,i, t):
            return model.variable_pvGeneration_BT2 [i, t] == model.param_pvGenerationNominal_BT2 [i, t] * SetUpScenarios.determinePVPeakOfBuildings(SetUpScenarios.numberOfBuildings_BT1 + i - 1) 
        model.constraint_PVgenerationTotal_BT2 = pyo.Constraint(model.set_buildings_BT2, model.set_timeslots, rule = PVgenerationTotalRule_BT2)
        
        
        #Equation for the assigned wind power of each BT2-building
        def windPowerAssignedgRule_BT2 (model,i, t):
            return model.variable_windPowerAssigned_BT2 [i, t] == model.param_windAssignedNominal_BT2 [i, t] * SetUpScenarios.maximalPowerOfWindTurbine 
            
        model.constraint_windPowerAssignedg_BT2 = pyo.Constraint(model.set_buildings_BT2,model.set_timeslots, rule = windPowerAssignedgRule_BT2)
        
        
        
        
        
        
        #Constraint system for the maximum number of starts of the heat pump
        
        model.variable_HPswitchedOff_Individual_SpaceHeating_BT2 = pyo.Var(model.set_buildings_BT2, model.set_timeslots, within =pyo.Binary, initialize=0.0)
        model.variable_HPswitchedOff_Individual_DHW_BT2 = pyo.Var(model.set_buildings_BT2, model.set_timeslots, within =pyo.Binary, initialize=0.0)
        model.variable_HPswitchedOff_Combined_BT2 = pyo.Var(model.set_buildings_BT2, model.set_timeslots, within =pyo.Binary)
        
        model.variable_HP_running_HelpAssociatedBinary_SpaceHeating_BT2 = pyo.Var(model.set_buildings_BT2, model.set_timeslots, within =pyo.Binary, initialize=0.0)
        model.variable_HP_running_HelpAssociatedBinary_DHW_BT2 = pyo.Var(model.set_buildings_BT2, model.set_timeslots, within =pyo.Binary, initialize=0.0)
        
        model.variable_HPswitchedOff_HelpModulationBinary_SpaceHeating_BT2 = pyo.Var(model.set_buildings_BT2, model.set_timeslots, within =pyo.Binary)
        model.variable_HPswitchedOff_HelpModulationBinary_DHW_BT2 = pyo.Var(model.set_buildings_BT2, model.set_timeslots, within =pyo.Binary)
        
        
        
        #Constraints for maximum number of starts for the space heating
        
        def maximumNumberOfStarts_Individual_SpaceHeating_EQ1_Rule_BT2 (model, i, t):
            if Run_Simulations.considerMaxiumNumberOfStartsHP_Individual ==True:
                if t == model.set_timeslots.first():
                    return model.variable_HPswitchedOff_Individual_SpaceHeating_BT2 [i, t] == 0
                return model.variable_HPswitchedOff_Individual_SpaceHeating_BT2 [i, t] <= model.variable_HP_running_HelpAssociatedBinary_SpaceHeating_BT2 [i, t-1]
            if Run_Simulations.considerMaxiumNumberOfStartsHP_Individual ==False: 
                return pyo.Constraint.Skip
                
        model.constraint_maximumNumberOfStarts_Individual_SpaceHeating_EQ1_BT2 = pyo.Constraint(model.set_buildings_BT2, model.set_timeslots, rule =maximumNumberOfStarts_Individual_SpaceHeating_EQ1_Rule_BT2)
        
        
        def maximumNumberOfStarts_Individual_SpaceHeating_EQ2_Rule_BT2 (model, i,  t):
            if Run_Simulations.considerMaxiumNumberOfStartsHP_Individual ==True:
                return model.variable_HP_running_HelpAssociatedBinary_SpaceHeating_BT2 [i, t] + model.variable_HPswitchedOff_Individual_SpaceHeating_BT2 [i, t] <= 1
            if Run_Simulations.considerMaxiumNumberOfStartsHP_Individual ==False: 
                return pyo.Constraint.Skip       
                
        model.constraint_maximumNumberOfStarts_Individual_SpaceHeating_EQ2_BT2 = pyo.Constraint(model.set_buildings_BT2, model.set_timeslots, rule =maximumNumberOfStarts_Individual_SpaceHeating_EQ2_Rule_BT2)
        
        
        
        def maximumNumberOfStarts_Individual_SpaceHeating_EQ2_2_Rule_BT2 (model, i, t):
            if Run_Simulations.considerMaxiumNumberOfStartsHP_Individual ==True:
                if t == model.set_timeslots.first():
                    return model.variable_HPswitchedOff_Individual_SpaceHeating_BT2 [i, t] == 0
                return model.variable_HP_running_HelpAssociatedBinary_SpaceHeating_BT2 [i, t - 1] <= model.variable_HP_running_HelpAssociatedBinary_SpaceHeating_BT2 [i, t] + model.variable_HPswitchedOff_Individual_SpaceHeating_BT2 [i, t]
            if Run_Simulations.considerMaxiumNumberOfStartsHP_Individual ==False: 
                return pyo.Constraint.Skip       
                
        model.constraint_maximumNumberOfStarts_Individual_SpaceHeating_EQ2_2_BT2 = pyo.Constraint(model.set_buildings_BT2, model.set_timeslots, rule =maximumNumberOfStarts_Individual_SpaceHeating_EQ2_2_Rule_BT2)
        
        
        
        def maximumNumberOfStarts_Individual_SpaceHeating_EQ3_HelpAssociatedBinary_Rule_BT2 (model, i, t):
            if Run_Simulations.considerMaxiumNumberOfStartsHP_Individual ==True or Run_Simulations.considerMaximumNumberOfStartsHP_Combined ==True:
                return model.variable_HP_running_HelpAssociatedBinary_SpaceHeating_BT2 [i, t] >= model.variable_heatGenerationCoefficient_SpaceHeating_BT2[i, t]
            if Run_Simulations.considerMaxiumNumberOfStartsHP_Individual ==False: 
               return pyo.Constraint.Skip
        
        model.constraint_maximumNumberOfStarts_Individual_SpaceHeating_EQ3_HelpAssociatedBinary_BT2 = pyo.Constraint(model.set_buildings_BT2, model.set_timeslots, rule =maximumNumberOfStarts_Individual_SpaceHeating_EQ3_HelpAssociatedBinary_Rule_BT2)
        
        
        
        def maximumNumberOfStarts_Individual_SpaceHeating_EQ4_HelpAssociatedBinary_Rule_BT2 (model, i, t):
            if Run_Simulations.considerMaxiumNumberOfStartsHP_Individual ==True or Run_Simulations.considerMaximumNumberOfStartsHP_Combined ==True:
                return model.variable_heatGenerationCoefficient_SpaceHeating_BT2[i, t] * (1/(SetUpScenarios.minimalModulationdDegree_HP/100))  >= model.variable_HP_running_HelpAssociatedBinary_SpaceHeating_BT2 [i, t]
            if Run_Simulations.considerMaxiumNumberOfStartsHP_Individual ==False: 
               return pyo.Constraint.Skip
        
        model.constraint_maximumNumberOfStarts_Individual_SpaceHeating_EQ4_HelpAssociatedBinary_BT2 = pyo.Constraint(model.set_buildings_BT2, model.set_timeslots, rule =maximumNumberOfStarts_Individual_SpaceHeating_EQ4_HelpAssociatedBinary_Rule_BT2)
        
        
        
        def maximumNumberOfStarts_Individual_SpaceHeating_EQ5_NumberOfStarts_Rule_BT2 (model, i, t):
            if Run_Simulations.considerMaxiumNumberOfStartsHP_Individual ==True:
                return  sum (model.variable_HPswitchedOff_Individual_SpaceHeating_BT2 [i, t] for t in model.set_timeslots)<= Run_Simulations.maximumNumberOfStarts_Individual 
            if Run_Simulations.considerMaxiumNumberOfStartsHP_Individual ==False: 
                return pyo.Constraint.Skip
        
        model.constraint_maximumNumberOfStarts_Individual_SpaceHeating_EQ5_NumberOfStarts_BT2 = pyo.Constraint(model.set_buildings_BT2, model.set_timeslots, rule =maximumNumberOfStarts_Individual_SpaceHeating_EQ5_NumberOfStarts_Rule_BT2)
        
        
        
        
        #Constraints for maximum number of starts for the DHW
        
        def maximumNumberOfStarts_Individual_DHW_EQ1_Rule_BT2 (model, i, t):
            if Run_Simulations.considerMaxiumNumberOfStartsHP_Individual ==True:
                if t == model.set_timeslots.first():
                    return model.variable_HPswitchedOff_Individual_DHW_BT2 [i, t] == 0
                return model.variable_HPswitchedOff_Individual_DHW_BT2 [i, t] <= model.variable_HP_running_HelpAssociatedBinary_DHW_BT2 [i, t-1]
            if Run_Simulations.considerMaxiumNumberOfStartsHP_Individual ==False: 
                return pyo.Constraint.Skip
                
        model.constraint_maximumNumberOfStarts_Individual_DHW_EQ1_BT2 = pyo.Constraint(model.set_buildings_BT2, model.set_timeslots, rule =maximumNumberOfStarts_Individual_DHW_EQ1_Rule_BT2)
        
        
        def maximumNumberOfStarts_Individual_DHW_EQ2_Rule_BT2 (model, i,  t):
            if Run_Simulations.considerMaxiumNumberOfStartsHP_Individual ==True:
                return model.variable_HP_running_HelpAssociatedBinary_DHW_BT2 [i, t] + model.variable_HPswitchedOff_Individual_DHW_BT2 [i, t] <= 1
            if Run_Simulations.considerMaxiumNumberOfStartsHP_Individual ==False: 
                return pyo.Constraint.Skip       
                
        model.constraint_maximumNumberOfStarts_Individual_DHW_EQ2_BT2 = pyo.Constraint(model.set_buildings_BT2, model.set_timeslots, rule =maximumNumberOfStarts_Individual_DHW_EQ2_Rule_BT2)
        
        
        
        def maximumNumberOfStarts_Individual_DHW_EQ2_2_Rule_BT2 (model, i, t):
            if Run_Simulations.considerMaxiumNumberOfStartsHP_Individual ==True:
                if t == model.set_timeslots.first():
                    return model.variable_HPswitchedOff_Individual_DHW_BT2 [i, t] == 0
                return model.variable_HP_running_HelpAssociatedBinary_DHW_BT2 [i, t - 1] <= model.variable_HP_running_HelpAssociatedBinary_DHW_BT2 [i, t] + model.variable_HPswitchedOff_Individual_DHW_BT2 [i, t]
            if Run_Simulations.considerMaxiumNumberOfStartsHP_Individual ==False: 
                return pyo.Constraint.Skip       
                
        model.constraint_maximumNumberOfStarts_Individual_DHW_EQ2_2_BT2 = pyo.Constraint(model.set_buildings_BT2, model.set_timeslots, rule =maximumNumberOfStarts_Individual_DHW_EQ2_2_Rule_BT2)
        
        
        
        def maximumNumberOfStarts_Individual_DHW_EQ3_HelpAssociatedBinary_Rule_BT2 (model, i, t):
            if Run_Simulations.considerMaxiumNumberOfStartsHP_Individual ==True or Run_Simulations.considerMaximumNumberOfStartsHP_Combined ==True:
                return model.variable_HP_running_HelpAssociatedBinary_DHW_BT2 [i, t] >= model.variable_heatGenerationCoefficient_DHW_BT2[i, t]
            if Run_Simulations.considerMaxiumNumberOfStartsHP_Individual ==False: 
               return pyo.Constraint.Skip
        
        model.constraint_maximumNumberOfStarts_Individual_DHW_EQ3_HelpAssociatedBinary_BT2 = pyo.Constraint(model.set_buildings_BT2, model.set_timeslots, rule =maximumNumberOfStarts_Individual_DHW_EQ3_HelpAssociatedBinary_Rule_BT2)
        
        
        
        def maximumNumberOfStarts_Individual_DHW_EQ4_HelpAssociatedBinary_Rule_BT2 (model, i, t):
            if Run_Simulations.considerMaxiumNumberOfStartsHP_Individual ==True or Run_Simulations.considerMaximumNumberOfStartsHP_Combined ==True:
                return model.variable_heatGenerationCoefficient_DHW_BT2[i, t] * (1/(SetUpScenarios.minimalModulationdDegree_HP/100))  >= model.variable_HP_running_HelpAssociatedBinary_DHW_BT2 [i, t]
            if Run_Simulations.considerMaxiumNumberOfStartsHP_Individual ==False: 
               return pyo.Constraint.Skip
        
        model.constraint_maximumNumberOfStarts_Individual_DHW_EQ4_HelpAssociatedBinary_BT2 = pyo.Constraint(model.set_buildings_BT2, model.set_timeslots, rule =maximumNumberOfStarts_Individual_DHW_EQ4_HelpAssociatedBinary_Rule_BT2)
        
        
        
        def maximumNumberOfStarts_Individual_DHW_EQ5_NumberOfStarts_Rule_BT2 (model, i, t):
            if Run_Simulations.considerMaxiumNumberOfStartsHP_Individual ==True:
                return  sum (model.variable_HPswitchedOff_Individual_DHW_BT2 [i, t] for t in model.set_timeslots)<= Run_Simulations.maximumNumberOfStarts_Individual 
            if Run_Simulations.considerMaxiumNumberOfStartsHP_Individual ==False: 
                return pyo.Constraint.Skip
        
        model.constraint_maximumNumberOfStarts_Individual_DHW_EQ5_NumberOfStarts_BT2 = pyo.Constraint(model.set_buildings_BT2, model.set_timeslots, rule =maximumNumberOfStarts_Individual_DHW_EQ5_NumberOfStarts_Rule_BT2)
        
        
        
        # Constraints for the maximum number of starts combined
        
        def maximumNumberOfStarts_Combined_EQ1_Rule_BT2 (model, i, t):
            if Run_Simulations.considerMaximumNumberOfStartsHP_Combined ==True:
                if t == model.set_timeslots.first():
                    return model.variable_HPswitchedOff_Combined_BT2 [i, t]== 0
                return model.variable_HP_running_HelpAssociatedBinary_SpaceHeating_BT2 [i, t-1] + model.variable_HP_running_HelpAssociatedBinary_DHW_BT2 [i, t-1] <= model.variable_HP_running_HelpAssociatedBinary_SpaceHeating_BT2 [i, t] + model.variable_HP_running_HelpAssociatedBinary_DHW_BT2 [i, t] + model.variable_HPswitchedOff_Combined_BT2 [i, t]
            if Run_Simulations.considerMaximumNumberOfStartsHP_Combined ==False: 
                return pyo.Constraint.Skip
            
        model.constraint_maximumNumberOfStarts_Combined_EQ1_BT2 = pyo.Constraint(model.set_buildings_BT2, model.set_timeslots, rule =maximumNumberOfStarts_Combined_EQ1_Rule_BT2)
        
        
        
        def maximumNumberOfStarts_Combined_EQ2_Rule_BT2 (model, i, t):
            if Run_Simulations.considerMaximumNumberOfStartsHP_Combined ==True:
                if t == model.set_timeslots.first():
                    return model.variable_HPswitchedOff_Combined_BT2 [i, t]== 0
                return model.variable_HP_running_HelpAssociatedBinary_SpaceHeating_BT2 [i, t] + model.variable_HP_running_HelpAssociatedBinary_DHW_BT2 [i, t] <= 1
            if Run_Simulations.considerMaximumNumberOfStartsHP_Combined ==False: 
               return  pyo.Constraint.Skip
            
        model.constraint_maximumNumberOfStarts_Combined_EQ2_BT2 = pyo.Constraint(model.set_buildings_BT2, model.set_timeslots, rule =maximumNumberOfStarts_Combined_EQ2_Rule_BT2)
        
        
        
        def maximumNumberOfStarts_Combined_EQ3_NumberOfStarts_Rule_BT2 (model, i, t):
            if Run_Simulations.considerMaximumNumberOfStartsHP_Combined ==True:
                return  sum (model.variable_HPswitchedOff_Combined_BT2 [i, t] for t in model.set_timeslots)<= Run_Simulations.maximumNumberOfStarts_Combined
            if Run_Simulations.considerMaximumNumberOfStartsHP_Combined ==False: 
                return pyo.Constraint.Skip
        
        model.maximumNumberOfStarts_Combined_EQ3_NumberOfStarts_BT2 = pyo.Constraint(model.set_buildings_BT2, model.set_timeslots, rule =maximumNumberOfStarts_Combined_EQ3_NumberOfStarts_Rule_BT2)
        
        
    
    
    
    
    
    
    
    ##########################################################################################################################
    
    
    #Building Type 3 (BT3): Buildings with an electric vehicle (EV)
    
    #Adjust dataframes to the current time resolution and set new index "Timeslot"
    
    for i in range (0, len(list_df_buildingData_BT3_original)):
        list_df_buildingData_BT3_original[i]['Time'] = pd.to_datetime(list_df_buildingData_BT3_original[i]['Time'], format = '%d.%m.%Y %H:%M')
        list_df_buildingData_BT3 [i] = list_df_buildingData_BT3_original[i].set_index('Time').resample(str(SetUpScenarios.timeResolution_InMinutes) +'Min').mean()
        for j in range (0, len(list_df_buildingData_BT3[i]['Availability of the EV'])):
            if list_df_buildingData_BT3 [i]['Availability of the EV'] [j] > 0.1:
                list_df_buildingData_BT3 [i]['Availability of the EV'] [j] = 1.0
            if list_df_buildingData_BT3 [i]['Availability of the EV'] [j] < 0.1 and list_df_buildingData_BT3 [i]['Availability of the EV'] [j] >0.01:
                list_df_buildingData_BT3 [i]['Availability of the EV'] [j] = 0
        
        
        arrayTimeSlots = [k for k in range (1,SetUpScenarios.numberOfTimeSlotsPerDay + 1)]
        list_df_buildingData_BT3 [i]['Timeslot'] = arrayTimeSlots
        list_df_buildingData_BT3 [i] = list_df_buildingData_BT3 [i].set_index('Timeslot')


    if SetUpScenarios.numberOfBuildings_BT3 >=1:
        #Create dataframes by using pandas series 
   
        list_windProfileNominal_BT3 = [SetUpScenarios.calculateAssignedWindPowerNominalPerBuilding (currentDay,SetUpScenarios.numberOfBuildings_BT1 + SetUpScenarios.numberOfBuildings_BT2 +index_BT3) for index_BT3 in range(0, SetUpScenarios.numberOfBuildings_BT3)]
        list_df_windPowerAssignedNominalPerBuilding_BT3 = [pd.DataFrame({'Timeslot': list_df_buildingData_BT3 [i].index, 'Wind [nominal]':list_windProfileNominal_BT3[i] }) for i in range (0, SetUpScenarios.numberOfBuildings_BT3)]

        for i in range (0, len(list_df_windPowerAssignedNominalPerBuilding_BT3)):
            del list_df_windPowerAssignedNominalPerBuilding_BT3[i]['Timeslot']
            list_df_windPowerAssignedNominalPerBuilding_BT3[i].index +=1   
            

            
        #Create availability array for the EV
        availabilityOfTheEVCombined = np.zeros((SetUpScenarios.numberOfBuildings_WithEV, SetUpScenarios.numberOfTimeSlotsPerDay))
        for index_BT3 in range (0, SetUpScenarios.numberOfBuildings_BT3):
            for index_timeslot_for_Availability in range (0,  SetUpScenarios.numberOfTimeSlotsPerDay):
                availabilityOfTheEVCombined [SetUpScenarios.numberOfBuildings_BT1 + index_BT3,index_timeslot_for_Availability] = list_df_buildingData_BT3 [index_BT3]['Availability of the EV'] [index_timeslot_for_Availability +1]
        

        list_energyConsumptionOfEVs_Joule_BT3 = np.zeros((SetUpScenarios.numberOfBuildings_BT3, SetUpScenarios.numberOfTimeSlotsPerDay))
        
        
        for indexBT3 in range (0, SetUpScenarios.numberOfBuildings_BT3):
            list_energyConsumptionOfEVs_Joule_BT3[indexBT3] = SetUpScenarios.generateEVEnergyConsumptionPatterns(availabilityOfTheEVCombined [SetUpScenarios.numberOfBuildings_BT1 + indexBT3],SetUpScenarios.numberOfBuildings_BT1 + indexBT3)
        
        list_df_energyConsumptionEV_Joule = [pd.DataFrame({'Timeslot': list_df_buildingData_BT3 [i].index, 'Energy':list_energyConsumptionOfEVs_Joule_BT3 [i] }) for i in range (0, SetUpScenarios.numberOfBuildings_BT3)]
        


        for i in range (0, len(list_energyConsumptionOfEVs_Joule_BT3)): 
            del list_df_energyConsumptionEV_Joule [i]['Timeslot']
            list_df_energyConsumptionEV_Joule[i].index +=1
            

        combinedDataframe_electricalDemand_BT3 = pd.DataFrame()
        combinedDataframe_pvGenerationNominal_BT3 = pd.DataFrame()
        combinedDataframe_windAssigned_BT3 = pd.DataFrame()
        combinedDataframe_availabilityPatternEV_BT3 = pd.DataFrame()
        combinedDataframe_energyConsumptionEV_Joule_BT3 = pd.DataFrame()
        

        
        
        for index in range (0, len(list_df_buildingData_BT3)):
            combinedDataframe_electricalDemand_BT3[index] = list_df_buildingData_BT3[index] ["Electricity [W]"]
            combinedDataframe_pvGenerationNominal_BT3[index] = list_df_buildingData_BT3[index] ["PV [nominal]"]
            combinedDataframe_windAssigned_BT3 [index]= list_df_windPowerAssignedNominalPerBuilding_BT3[index] ["Wind [nominal]"]
            combinedDataframe_availabilityPatternEV_BT3[index] = list_df_buildingData_BT3 [index]  ['Availability of the EV']
            combinedDataframe_energyConsumptionEV_Joule_BT3[index] = list_df_energyConsumptionEV_Joule[index] ["Energy"]
        


    
        #Round the values
        for index in range (0,  SetUpScenarios.numberOfBuildings_BT3):
            decimalsForRounding = 2    
            list_df_buildingData_BT3 [index]['Electricity [W]'] = list_df_buildingData_BT3 [index]['Electricity [W]'].apply(lambda x: round(x, decimalsForRounding))
            decimalsForRounding = 4 
            list_df_buildingData_BT3 [index]['PV [nominal]'] = list_df_buildingData_BT3 [index]['PV [nominal]'].apply(lambda x: round(x, decimalsForRounding))
    

        
        def init_electricalDemand (model, i,j):
            return combinedDataframe_electricalDemand_BT3.iloc[j-1, i-1]
        
        model.param_electricalDemand_In_W_BT3 = pyo.Param(model.set_buildings_BT3, model.set_timeslots,mutable = True, initialize=init_electricalDemand)
        
        
        def init_pvGenerationNominal (model, i,j):
            return combinedDataframe_pvGenerationNominal_BT3.iloc[j-1, i-1]
        
        model.param_pvGenerationNominal_BT3  = pyo.Param(model.set_buildings_BT3, model.set_timeslots, mutable = True, initialize=init_pvGenerationNominal)
        
        
        model.param_outSideTemperature_In_C = pyo.Param(model.set_timeslots, initialize=dictionaryTemperature_In_C)
        
        
        def init_windAssignedNominal (model, i,j):
            return combinedDataframe_windAssigned_BT3.iloc[j-1, i-1]
        
        model.param_windAssignedNominal_BT3  = pyo.Param(model.set_buildings_BT3, model.set_timeslots, mutable = True, initialize=init_windAssignedNominal)
        
        
        def init_availabilityPatternEV (model, i,j):
            return combinedDataframe_availabilityPatternEV_BT3.iloc[j-1, i-1]
        
        model.param_availabilityPerTimeSlotOfEV_BT3  = pyo.Param(model.set_buildings_BT3, model.set_timeslots, mutable = True, initialize=init_availabilityPatternEV)
        
        
        def init_energyConsumptionEV_Joule (model, i,j):
            return combinedDataframe_energyConsumptionEV_Joule_BT3.iloc[j-1, i-1]
        
        model.param_energyConsumptionEV_Joule_BT3  = pyo.Param(model.set_buildings_BT3, model.set_timeslots, mutable = True, initialize=init_energyConsumptionEV_Joule)
        
        
        #Define the variables
        
        
        model.variable_currentChargingPowerEV_BT3 = pyo.Var(model.set_buildings_BT3, model.set_timeslots, within=pyo.NonNegativeReals, bounds=(0,SetUpScenarios.chargingPowerMaximal_EV))
        model.variable_energyLevelEV_BT3 = pyo.Var(model.set_buildings_BT3, model.set_timeslots, within=pyo.NonNegativeReals, bounds=(0, SetUpScenarios.capacityMaximal_EV))
        model.variable_SOC_EV_BT3= pyo.Var(model.set_buildings_BT3, model.set_timeslots,  within=pyo.NonNegativeReals, bounds=(0,100))
        model.variable_electricalPowerTotal_BT3 = pyo.Var(model.set_buildings_BT3, model.set_timeslots)
        
        model.variable_pvGeneration_BT3 = pyo.Var(model.set_buildings_BT3, model.set_timeslots)
        model.variable_windPowerAssigned_BT3 = pyo.Var(model.set_buildings_BT3, model.set_timeslots)
        
        
        # Defining the constraints 
        
        
        
        #EV Energy Level
        def energyLevelOfEVRule_BT3 (model, i, t):
            if t == model.set_timeslots.first():
                return model.variable_energyLevelEV_BT3 [i, t] ==  ((SetUpScenarios.initialSOC_EV/100) * SetUpScenarios.capacityMaximal_EV) + (model.variable_currentChargingPowerEV_BT3 [i, t] * (SetUpScenarios.chargingEfficiency_EV/100) * SetUpScenarios.timeResolution_InMinutes * 60 - model.param_energyConsumptionEV_Joule_BT3 [i, t])
            return model.variable_energyLevelEV_BT3[i, t]  == model.variable_energyLevelEV_BT3 [i, t-1] + ( model.variable_currentChargingPowerEV_BT3 [i, t] * (SetUpScenarios.chargingEfficiency_EV/100) * SetUpScenarios.timeResolution_InMinutes * 60 - model.param_energyConsumptionEV_Joule_BT3 [i, t]) 
        
        model.constraint_energyLevelOfEV_BT3 = pyo.Constraint(model.set_buildings_BT3, model.set_timeslots, rule=energyLevelOfEVRule_BT3)
        
        
        #Constraints for the minimal and maximal energy level of the EV at the end of the optimization horizon
        def constraint_energyLevelOfEV_lastLowerLimitRule_BT3 (model, i, t):
            return model.variable_energyLevelEV_BT3[i, model.set_timeslots.last()] >= ((SetUpScenarios.initialSOC_EV - SetUpScenarios.endSOC_EVAllowedDeviationFromInitalValue)/100) * SetUpScenarios.capacityMaximal_EV
        
        model.constraint_energyLevelOfEV_lastLowerLimit_BT3 = pyo.Constraint (model.set_buildings_BT3, model.set_timeslots, rule=constraint_energyLevelOfEV_lastLowerLimitRule_BT3) 
        
        
        def constraint_energyLevelOfEV_lastUpperLimitRule_BT3 (model, i, t):
            return model.variable_energyLevelEV_BT3[i, model.set_timeslots.last()] <= ((SetUpScenarios.initialSOC_EV + SetUpScenarios.endSOC_EVAllowedDeviationFromInitalValue)/100) * SetUpScenarios.capacityMaximal_EV
        model.constraint_energyLevelOfEV_lastUpperLimit_BT3 = pyo.Constraint (model.set_buildings_BT3, model.set_timeslots, rule=constraint_energyLevelOfEV_lastUpperLimitRule_BT3)
        
        
        
        #SOC of the EV
        def socOfEVRule_BT3 (model, i, t):
            return model.variable_SOC_EV_BT3[i, t] == (model.variable_energyLevelEV_BT3 [i, t] / SetUpScenarios.capacityMaximal_EV)*100
        
        model.constraint_SOCofEV_BT3 =  pyo.Constraint(model.set_buildings_BT3, model.set_timeslots, rule = socOfEVRule_BT3)
        
        
        #Constraint for the charging power: The EV can only be charged if it is at home (available)
        def chargingPowerOfTheEVRul_BT3 (model, i, t):
            return model.variable_currentChargingPowerEV_BT3 [i, t] <=  model.param_availabilityPerTimeSlotOfEV_BT3 [i, t] * SetUpScenarios.chargingPowerMaximal_EV 
        
        model.constraint_chargingPowerOfTheEV_BT3 = pyo.Constraint(model.set_buildings_BT3, model.set_timeslots, rule = chargingPowerOfTheEVRul_BT3)
        
        
        
        
        #Constraints for the electrical power of BT3
        def electricalPowerTotalRule_BT3 (model,i, t):
            return model.variable_electricalPowerTotal_BT3 [i, t] ==  model.variable_currentChargingPowerEV_BT3 [i, t] + model.param_electricalDemand_In_W_BT3 [i, t] 
        
        model.constraint_electricalPowerTotal_BT3 = pyo.Constraint(model.set_buildings_BT3,model.set_timeslots, rule = electricalPowerTotalRule_BT3)
        
        
        
        #Equation for calculating the PV generation of each BT3-building
        def PVgenerationTotalRule_BT3 (model,i, t):
            return model.variable_pvGeneration_BT3 [i, t] == model.param_pvGenerationNominal_BT3 [i, t] * SetUpScenarios.determinePVPeakOfBuildings(SetUpScenarios.numberOfBuildings_BT1 + SetUpScenarios.numberOfBuildings_BT2 + i - 1) 
        model.constraint_PVgenerationTotal_BT3 = pyo.Constraint(model.set_buildings_BT3, model.set_timeslots, rule = PVgenerationTotalRule_BT3)
 
        
        #Equation for the assigned wind power of each BT3-building
        def windPowerAssignedgRule_BT3 (model,i, t):
            return model.variable_windPowerAssigned_BT3 [i, t] == model.param_windAssignedNominal_BT3 [i, t] * SetUpScenarios.maximalPowerOfWindTurbine 
            
        model.constraint_windPowerAssignedg_BT3 = pyo.Constraint(model.set_buildings_BT3,model.set_timeslots, rule = windPowerAssignedgRule_BT3)
    
    
    
    ###########################################################################################################################
    

    #Building Type 4 (BT4): Buildings with modulating air-source heat pump (mHP) for a mulit family house (MFH)
    
   
    
    #Adjust dataframes to the current time resolution and set new index "Timeslot"
    
    for i in range (0, len(list_df_buildingData_BT4_original)):
        list_df_buildingData_BT4_original[i]['Time'] = pd.to_datetime(list_df_buildingData_BT4_original[i]['Time'], format = '%d.%m.%Y %H:%M')
        list_df_buildingData_BT4 [i] = list_df_buildingData_BT4_original[i].set_index('Time').resample(str(SetUpScenarios.timeResolution_InMinutes) +'Min').mean()

        
        arrayTimeSlots = [k for k in range (1,SetUpScenarios.numberOfTimeSlotsPerDay + 1)]
        list_df_buildingData_BT4 [i]['Timeslot'] = arrayTimeSlots
        list_df_buildingData_BT4 [i] = list_df_buildingData_BT4 [i].set_index('Timeslot')
        
        

    if SetUpScenarios.numberOfBuildings_BT4 >=1:
        #Create dataframes by using pandas series 
   
        list_windProfileNominal_BT4 = [SetUpScenarios.calculateAssignedWindPowerNominalPerBuilding (currentDay,SetUpScenarios.numberOfBuildings_BT1 + SetUpScenarios.numberOfBuildings_BT2 + SetUpScenarios.numberOfBuildings_BT3 + index_BT4) for index_BT4 in range(0, SetUpScenarios.numberOfBuildings_BT4)]
        list_df_windPowerAssignedNominalPerBuilding_BT4 = [pd.DataFrame({'Timeslot': list_df_buildingData_BT4 [i].index, 'Wind [nominal]':list_windProfileNominal_BT4[i] }) for i in range (0, SetUpScenarios.numberOfBuildings_BT4)]

        for i in range (0, len(list_df_windPowerAssignedNominalPerBuilding_BT4)):
            del list_df_windPowerAssignedNominalPerBuilding_BT4[i]['Timeslot']
            list_df_windPowerAssignedNominalPerBuilding_BT4[i].index +=1   
       
        combinedDataframe_heatDemand_BT4 = pd.DataFrame()
        combinedDataframe_electricalDemand_BT4 = pd.DataFrame()
        combinedDataframe_pvGenerationNominal_BT4 = pd.DataFrame()
        combinedDataframe_windAssigned_BT4 = pd.DataFrame()
        

        for index in range (0, len(list_df_buildingData_BT4)):
            combinedDataframe_heatDemand_BT4[index] = list_df_buildingData_BT4[index] ["Space Heating [W]"]
            combinedDataframe_electricalDemand_BT4[index] = list_df_buildingData_BT4[index] ["Electricity [W]"]
            combinedDataframe_pvGenerationNominal_BT4[index] = list_df_buildingData_BT4[index] ["PV [nominal]"]
            combinedDataframe_windAssigned_BT4 [index]= list_df_windPowerAssignedNominalPerBuilding_BT4[index] ["Wind [nominal]"]


    
        #Round the values
        for index in range (0,  SetUpScenarios.numberOfBuildings_BT4):
            decimalsForRounding = 2    
            list_df_buildingData_BT4 [index]['Space Heating [W]'] = list_df_buildingData_BT4 [index]['Space Heating [W]'].apply(lambda x: round(x, decimalsForRounding))
            list_df_buildingData_BT4 [index]['Electricity [W]'] = list_df_buildingData_BT4 [index]['Electricity [W]'].apply(lambda x: round(x, decimalsForRounding))
            decimalsForRounding = 4 
            list_df_buildingData_BT4 [index]['PV [nominal]'] = list_df_buildingData_BT4 [index]['PV [nominal]'].apply(lambda x: round(x, decimalsForRounding))
    
    
        #Calculate the COPs for the heat hump
        df_copHeatPump =  pd.DataFrame({'Timeslot': list_df_buildingData_BT4 [0].index, 'COP_SpaceHeating':cop_heatPump_SpaceHeating, 'COP_DHW':cop_heatPump_DHW})
        dictionaryCOPHeatPump_SpaceHeating= df_copHeatPump.set_index('Timeslot')['COP_SpaceHeating'].to_dict()


        #Define the parameters of the model in pyomo
        def init_heatDemand (model, i,j):
            return combinedDataframe_heatDemand_BT4.iloc[j-1, i-1]
        
        model.param_heatDemand_In_W_BT4 = pyo.Param(model.set_buildings_BT4, model.set_timeslots, mutable = True, initialize=init_heatDemand)
        
        
        def init_electricalDemand (model, i,j):
            return combinedDataframe_electricalDemand_BT4.iloc[j-1, i-1]
        
        model.param_electricalDemand_In_W_BT4 = pyo.Param(model.set_buildings_BT4, model.set_timeslots,mutable = True, initialize=init_electricalDemand)
        
        
        def init_pvGenerationNominal (model, i,j):
            return combinedDataframe_pvGenerationNominal_BT4.iloc[j-1, i-1]
        
        model.param_pvGenerationNominal_BT4  = pyo.Param(model.set_buildings_BT4, model.set_timeslots, mutable = True, initialize=init_pvGenerationNominal)
        
        
        model.param_outSideTemperature_In_C = pyo.Param(model.set_timeslots, initialize=dictionaryTemperature_In_C)
        
        
        def init_windAssignedNominal (model, i,j):
            return combinedDataframe_windAssigned_BT4.iloc[j-1, i-1]
        
        model.param_windAssignedNominal_BT4  = pyo.Param(model.set_buildings_BT4, model.set_timeslots, mutable = True, initialize=init_windAssignedNominal)
        
        
        model.param_COPHeatPump_SpaceHeating_BT4 = pyo.Param(model.set_timeslots, initialize=dictionaryCOPHeatPump_SpaceHeating)
        

        
        
        #Define the variables
        
        model.variable_heatGenerationCoefficient_SpaceHeating_BT4 = pyo.Var(model.set_buildings_BT4, model.set_timeslots,bounds=(0,1))
        model.variable_temperatureBufferStorage_BT4 = pyo.Var(model.set_buildings_BT4, model.set_timeslots,  bounds=(SetUpScenarios.minimalBufferStorageTemperature  , SetUpScenarios.maximalBufferStorageTemperature ))
        
        model.variable_electricalPowerTotal_BT4 = pyo.Var(model.set_buildings_BT4, model.set_timeslots)
        model.variable_pvGeneration_BT4 = pyo.Var(model.set_buildings_BT4, model.set_timeslots)
        model.variable_windPowerAssigned_BT4 = pyo.Var(model.set_buildings_BT4, model.set_timeslots)

        model.variable_deviation_temperature_positive_BT4 = pyo.Var(model.set_buildings_BT4, model.set_timeslots, within=pyo.NonNegativeReals)
        model.variable_deviation_temperature_negative_BT4 = pyo.Var(model.set_buildings_BT4, model.set_timeslots, within=pyo.NonNegativeReals)
        

        # Defining the constraints 
        
        
        #Temperature constraint for the buffer storage (space heating) with energetic difference equation
        
        def temperatureBufferStorageConstraintRule_BT4(model, i, t):
            if t == model.set_timeslots.first():
                return model.variable_temperatureBufferStorage_BT4[i, t] == SetUpScenarios.initialBufferStorageTemperature + ((model.variable_heatGenerationCoefficient_SpaceHeating_BT4[i, t] * model.param_COPHeatPump_SpaceHeating_BT4[t] *  SetUpScenarios.electricalPower_HP_BT4_MFH * SetUpScenarios.timeResolution_InMinutes * 60  - model.param_heatDemand_In_W_BT4 [i, t]  * SetUpScenarios.timeResolution_InMinutes * 60 - SetUpScenarios.standingLossesBufferStorage_BT4_MFH * SetUpScenarios.timeResolution_InMinutes * 60) / (SetUpScenarios.capacityOfBufferStorage_BT4_MFH * SetUpScenarios.densityOfCement * SetUpScenarios.specificHeatCapacityOfCement))
            return model.variable_temperatureBufferStorage_BT4[i, t] == model.variable_temperatureBufferStorage_BT4[i, t-1] + ((model.variable_heatGenerationCoefficient_SpaceHeating_BT4[i, t] * model.param_COPHeatPump_SpaceHeating_BT4[t] *  SetUpScenarios.electricalPower_HP_BT4_MFH * SetUpScenarios.timeResolution_InMinutes * 60  - model.param_heatDemand_In_W_BT4 [i, t]  * SetUpScenarios.timeResolution_InMinutes * 60 - SetUpScenarios.standingLossesBufferStorage_BT4_MFH * SetUpScenarios.timeResolution_InMinutes * 60) / (SetUpScenarios.capacityOfBufferStorage_BT4_MFH * SetUpScenarios.densityOfCement * SetUpScenarios.specificHeatCapacityOfCement))
        
        model.constraint_temperatureBufferStorage_BT4= pyo.Constraint (model.set_buildings_BT4, model.set_timeslots, rule=temperatureBufferStorageConstraintRule_BT4)
        

        #Constraint for allowing thermal discomfort
        def temperatureDeviationConstraintRule1_BT4 (model, i, t):
            if optParameters['optimizationGoal_minimizeThermalDiscomfort'] == True:
                return  SetUpScenarios.idealComfortTemperature - SetUpScenarios.allowedTemperatureDeviationForOptimalComfort <= model.variable_temperatureBufferStorage_BT4[i, t] - model.variable_deviation_temperature_positive_BT4 [i, t] +  model.variable_deviation_temperature_negative_BT4 [i, t]
            else:
                return pyo.Constraint.Skip

        model.constraint_temperatureDeviationConstraint1_BT4 = pyo.Constraint(model.set_buildings_BT4, model.set_timeslots, rule = temperatureDeviationConstraintRule1_BT4)


        def temperatureDeviationConstraintRule2_BT4 (model, i, t):
            if optParameters['optimizationGoal_minimizeThermalDiscomfort'] == True:
                return  model.variable_temperatureBufferStorage_BT4[i, t] - model.variable_deviation_temperature_positive_BT4 [i, t] +  model.variable_deviation_temperature_negative_BT4 [i, t]<= SetUpScenarios.idealComfortTemperature + SetUpScenarios.allowedTemperatureDeviationForOptimalComfort
            else:
                return pyo.Constraint.Skip

        model.constraint_temperatureDeviationConstraint2_BT4 = pyo.Constraint(model.set_buildings_BT4, model.set_timeslots, rule = temperatureDeviationConstraintRule2_BT4)


                
        
        
        #Constraints for the minimal and maximal temperature at the end of the optimization horizon
        def temperatureBufferStorage_lastLowerLimitRule_BT4 (model, i, t):
            return model.variable_temperatureBufferStorage_BT4[i, model.set_timeslots.last()] >= SetUpScenarios.initialBufferStorageTemperature - SetUpScenarios.endBufferStorageTemperatureAllowedDeviationFromInitalValue
        
        model.constraint_temperatureBufferStorage_lastLowerLimit_BT4 = pyo.Constraint (model.set_buildings_BT4, model.set_timeslots, rule=temperatureBufferStorage_lastLowerLimitRule_BT4) 
        
        
        
        def temperatureBufferStorage_lastUpperLimitRule_BT4 (model, i, t):
            return model.variable_temperatureBufferStorage_BT4[i, model.set_timeslots.last()] <= SetUpScenarios.initialBufferStorageTemperature + SetUpScenarios.endBufferStorageTemperatureAllowedDeviationFromInitalValue
        
        model.constraint_temperatureBufferStorage_lastUpperLimit_BT4 = pyo.Constraint (model.set_buildings_BT4, model.set_timeslots, rule=temperatureBufferStorage_lastUpperLimitRule_BT4) 
        
        
        
        #Constraint for minimal modulation degree if the heat pump is always switched on
        def minimalModulationDegreeOfTheHeatPumpRule_BT4 (model,i, t):
            if Run_Simulations.isHPAlwaysSwitchedOn ==True:
                return model.variable_heatGenerationCoefficient_SpaceHeating_BT4[i,t]  >= (SetUpScenarios.minimalModulationdDegree_HP/100)
            if Run_Simulations.isHPAlwaysSwitchedOn ==False:
                return pyo.Constraint.Feasible
                
            
        model.constraint_minimalModulationDegreeOfTheHeatPump_BT4 = pyo.Constraint(model.set_buildings_BT4, model.set_timeslots, rule = minimalModulationDegreeOfTheHeatPumpRule_BT4)
        
      
        
        #Constraints for the electrical power of BT4
        def electricalPowerTotalRule_BT4 (model,i, t):
            return model.variable_electricalPowerTotal_BT4 [i, t] == (model.variable_heatGenerationCoefficient_SpaceHeating_BT4 [i, t] ) * SetUpScenarios.electricalPower_HP_BT4_MFH  + model.param_electricalDemand_In_W_BT4 [i, t] 
        
        model.constraint_electricalPowerTotal_BT4 = pyo.Constraint(model.set_buildings_BT4,model.set_timeslots, rule = electricalPowerTotalRule_BT4)
        
        
        
        #Equation for calculating the PV generation of each BT4-building
        def PVgenerationTotalRule_BT4 (model,i, t):
            return model.variable_pvGeneration_BT4 [i, t] == model.param_pvGenerationNominal_BT4 [i, t] * SetUpScenarios.determinePVPeakOfBuildings(SetUpScenarios.numberOfBuildings_BT1 + SetUpScenarios.numberOfBuildings_BT2 + SetUpScenarios.numberOfBuildings_BT3 + i - 1)
        model.constraint_PVgenerationTotal_BT4 = pyo.Constraint(model.set_buildings_BT4, model.set_timeslots, rule = PVgenerationTotalRule_BT4)
        
        
        #Equation for calculating the assigned wind power for each BT4-building
        def windPowerAssignedgRule_BT4 (model,i, t):
            return model.variable_windPowerAssigned_BT4 [i, t] == model.param_windAssignedNominal_BT4 [i, t] * SetUpScenarios.maximalPowerOfWindTurbine 
            
        model.constraint_windPowerAssignedg_BT4 = pyo.Constraint(model.set_buildings_BT4,model.set_timeslots, rule = windPowerAssignedgRule_BT4)
        
        
        
       
        
        #Constraint system for the maximum number of starts of the heat pump
        
        model.variable_HPswitchedOff_Individual_SpaceHeating_BT4 = pyo.Var(model.set_buildings_BT4, model.set_timeslots, within =pyo.Binary, initialize=0.0)
        model.variable_HP_running_HelpAssociatedBinary_SpaceHeating_BT4 = pyo.Var(model.set_buildings_BT4, model.set_timeslots, within =pyo.Binary, initialize=0.0)
        model.variable_HPswitchedOff_HelpModulationBinary_SpaceHeating_BT4 = pyo.Var(model.set_buildings_BT4, model.set_timeslots, within =pyo.Binary)
        
        
        #Constraints for maximum number of starts for the space heating
        
        def maximumNumberOfStarts_Individual_SpaceHeating_EQ1_Rule_BT4 (model, i, t):
            if Run_Simulations.considerMaxiumNumberOfStartsHP_MFH_Individual ==True:
                if t == model.set_timeslots.first():
                    return model.variable_HPswitchedOff_Individual_SpaceHeating_BT4 [i, t] == 0
                return model.variable_HPswitchedOff_Individual_SpaceHeating_BT4 [i, t] <= model.variable_HP_running_HelpAssociatedBinary_SpaceHeating_BT4 [i, t-1]
            if Run_Simulations.considerMaxiumNumberOfStartsHP_MFH_Individual ==False: 
                return pyo.Constraint.Skip
                
        model.constraint_maximumNumberOfStarts_Individual_SpaceHeating_EQ1_BT4 = pyo.Constraint(model.set_buildings_BT4, model.set_timeslots, rule =maximumNumberOfStarts_Individual_SpaceHeating_EQ1_Rule_BT4)
        
        
        def maximumNumberOfStarts_Individual_SpaceHeating_EQ2_Rule_BT4 (model, i,  t):
            if Run_Simulations.considerMaxiumNumberOfStartsHP_MFH_Individual ==True:
                return model.variable_HP_running_HelpAssociatedBinary_SpaceHeating_BT4 [i, t] + model.variable_HPswitchedOff_Individual_SpaceHeating_BT4 [i, t] <= 1
            if Run_Simulations.considerMaxiumNumberOfStartsHP_MFH_Individual ==False: 
                return pyo.Constraint.Skip       
                
        model.constraint_maximumNumberOfStarts_Individual_SpaceHeating_EQ2_BT4 = pyo.Constraint(model.set_buildings_BT4, model.set_timeslots, rule =maximumNumberOfStarts_Individual_SpaceHeating_EQ2_Rule_BT4)
        
        
        
        def maximumNumberOfStarts_Individual_SpaceHeating_EQ2_2_Rule_BT4 (model, i, t):
            if Run_Simulations.considerMaxiumNumberOfStartsHP_MFH_Individual ==True:
                if t == model.set_timeslots.first():
                    return model.variable_HPswitchedOff_Individual_SpaceHeating_BT4 [i, t] == 0
                return model.variable_HP_running_HelpAssociatedBinary_SpaceHeating_BT4 [i, t - 1] <= model.variable_HP_running_HelpAssociatedBinary_SpaceHeating_BT4 [i, t] + model.variable_HPswitchedOff_Individual_SpaceHeating_BT4 [i, t]
            if Run_Simulations.considerMaxiumNumberOfStartsHP_MFH_Individual ==False: 
                return pyo.Constraint.Skip       
                
        model.constraint_maximumNumberOfStarts_Individual_SpaceHeating_EQ2_2_BT4 = pyo.Constraint(model.set_buildings_BT4, model.set_timeslots, rule =maximumNumberOfStarts_Individual_SpaceHeating_EQ2_2_Rule_BT4)
        
        
        
        def maximumNumberOfStarts_Individual_SpaceHeating_EQ3_HelpAssociatedBinary_Rule_BT4 (model, i, t):
            if Run_Simulations.considerMaxiumNumberOfStartsHP_MFH_Individual ==True or Run_Simulations.considerMaximumNumberOfStartsHP_Combined ==True:
                return model.variable_HP_running_HelpAssociatedBinary_SpaceHeating_BT4 [i, t] >= model.variable_heatGenerationCoefficient_SpaceHeating_BT4[i, t]
            if Run_Simulations.considerMaxiumNumberOfStartsHP_MFH_Individual ==False: 
               return pyo.Constraint.Skip
        
        model.constraint_maximumNumberOfStarts_Individual_SpaceHeating_EQ3_HelpAssociatedBinary_BT4 = pyo.Constraint(model.set_buildings_BT4, model.set_timeslots, rule =maximumNumberOfStarts_Individual_SpaceHeating_EQ3_HelpAssociatedBinary_Rule_BT4)
        
        
        
        def maximumNumberOfStarts_Individual_SpaceHeating_EQ4_HelpAssociatedBinary_Rule_BT4 (model, i, t):
            if Run_Simulations.considerMaxiumNumberOfStartsHP_MFH_Individual ==True or Run_Simulations.considerMaximumNumberOfStartsHP_Combined ==True:
                return model.variable_heatGenerationCoefficient_SpaceHeating_BT4[i, t] * (1/(SetUpScenarios.minimalModulationdDegree_HP/100))  >= model.variable_HP_running_HelpAssociatedBinary_SpaceHeating_BT4 [i, t]
            if Run_Simulations.considerMaxiumNumberOfStartsHP_MFH_Individual ==False: 
               return pyo.Constraint.Skip
        
        model.constraint_maximumNumberOfStarts_Individual_SpaceHeating_EQ4_HelpAssociatedBinary_BT4 = pyo.Constraint(model.set_buildings_BT4, model.set_timeslots, rule =maximumNumberOfStarts_Individual_SpaceHeating_EQ4_HelpAssociatedBinary_Rule_BT4)
        
        
        
        def maximumNumberOfStarts_Individual_SpaceHeating_EQ5_NumberOfStarts_Rule_BT4 (model, i, t):
            if Run_Simulations.considerMaxiumNumberOfStartsHP_MFH_Individual ==True:
                return  sum (model.variable_HPswitchedOff_Individual_SpaceHeating_BT4 [i, t] for t in model.set_timeslots)<= Run_Simulations.maximumNumberOfStarts_Individual 
            if Run_Simulations.considerMaxiumNumberOfStartsHP_MFH_Individual ==False: 
                return pyo.Constraint.Skip
        
        model.constraint_maximumNumberOfStarts_Individual_SpaceHeating_EQ5_NumberOfStarts_BT4 = pyo.Constraint(model.set_buildings_BT4, model.set_timeslots, rule =maximumNumberOfStarts_Individual_SpaceHeating_EQ5_NumberOfStarts_Rule_BT4)
    
    
    ##########################################################################################################################
    
    
    #Building Type 5 (BT5): Buildings with a battery storage system (BAT)
    
    #Adjust dataframes to the current time resolution and set new index "Timeslot"
    
    for i in range (0, len(list_df_buildingData_BT5_original)):
        list_df_buildingData_BT5_original[i]['Time'] = pd.to_datetime(list_df_buildingData_BT5_original[i]['Time'], format = '%d.%m.%Y %H:%M')
        list_df_buildingData_BT5 [i] = list_df_buildingData_BT5_original[i].set_index('Time').resample(str(SetUpScenarios.timeResolution_InMinutes) +'Min').mean()

        arrayTimeSlots = [k for k in range (1,SetUpScenarios.numberOfTimeSlotsPerDay + 1)]
        list_df_buildingData_BT5 [i]['Timeslot'] = arrayTimeSlots
        list_df_buildingData_BT5 [i] = list_df_buildingData_BT5 [i].set_index('Timeslot')


    if SetUpScenarios.numberOfBuildings_BT5 >=1:
        #Create dataframes by using pandas series 
   
        list_windProfileNominal_BT5 = [SetUpScenarios.calculateAssignedWindPowerNominalPerBuilding (currentDay,SetUpScenarios.numberOfBuildings_BT1 + SetUpScenarios.numberOfBuildings_BT2 +index_BT5) for index_BT5 in range(0, SetUpScenarios.numberOfBuildings_BT5)]
        list_df_windPowerAssignedNominalPerBuilding_BT5 = [pd.DataFrame({'Timeslot': list_df_buildingData_BT5 [i].index, 'Wind [nominal]':list_windProfileNominal_BT5[i] }) for i in range (0, SetUpScenarios.numberOfBuildings_BT5)]

        for i in range (0, len(list_df_windPowerAssignedNominalPerBuilding_BT5)):
            del list_df_windPowerAssignedNominalPerBuilding_BT5[i]['Timeslot']
            list_df_windPowerAssignedNominalPerBuilding_BT5[i].index +=1   
            

        combinedDataframe_electricalDemand_BT5 = pd.DataFrame()
        combinedDataframe_pvGenerationNominal_BT5 = pd.DataFrame()
        combinedDataframe_windAssigned_BT5 = pd.DataFrame()
        combinedDataframe_availabilityPatternEV_BT5 = pd.DataFrame()
        

        
        for index in range (0, len(list_df_buildingData_BT5)):
            combinedDataframe_electricalDemand_BT5[index] = list_df_buildingData_BT5[index] ["Electricity [W]"]
            combinedDataframe_pvGenerationNominal_BT5[index] = list_df_buildingData_BT5[index] ["PV [nominal]"]
            combinedDataframe_windAssigned_BT5 [index]= list_df_windPowerAssignedNominalPerBuilding_BT5[index] ["Wind [nominal]"]


    
        #Round the values
        for index in range (0,  SetUpScenarios.numberOfBuildings_BT5):
            decimalsForRounding = 2    
            list_df_buildingData_BT5 [index]['Electricity [W]'] = list_df_buildingData_BT5 [index]['Electricity [W]'].apply(lambda x: round(x, decimalsForRounding))
            decimalsForRounding = 4 
            list_df_buildingData_BT5 [index]['PV [nominal]'] = list_df_buildingData_BT5 [index]['PV [nominal]'].apply(lambda x: round(x, decimalsForRounding))
    

        
        def init_electricalDemand (model, i,j):
            return combinedDataframe_electricalDemand_BT5.iloc[j-1, i-1]
        
        model.param_electricalDemand_In_W_BT5 = pyo.Param(model.set_buildings_BT5, model.set_timeslots,mutable = True, initialize=init_electricalDemand)
        
        
        def init_pvGenerationNominal (model, i,j):
            return combinedDataframe_pvGenerationNominal_BT5.iloc[j-1, i-1]
        
        model.param_pvGenerationNominal_BT5  = pyo.Param(model.set_buildings_BT5, model.set_timeslots, mutable = True, initialize=init_pvGenerationNominal)
        
        
        model.param_outSideTemperature_In_C = pyo.Param(model.set_timeslots, initialize=dictionaryTemperature_In_C)
        
        
        def init_windAssignedNominal (model, i,j):
            return combinedDataframe_windAssigned_BT5.iloc[j-1, i-1]
        
        model.param_windAssignedNominal_BT5  = pyo.Param(model.set_buildings_BT5, model.set_timeslots, mutable = True, initialize=init_windAssignedNominal)


        #Define the variables
        model.variable_currentChargingPowerBAT_BT5 = pyo.Var(model.set_buildings_BT5, model.set_timeslots, within=pyo.NonNegativeReals, bounds=(0,SetUpScenarios.chargingPowerMaximal_BAT))
        model.variable_currentDisChargingPowerBAT_BT5 = pyo.Var(model.set_buildings_BT5, model.set_timeslots, within=pyo.NonNegativeReals, bounds=(0,SetUpScenarios.chargingPowerMaximal_BAT))
        model.variable_helpBinaryChargingPositive_BT5 = pyo.Var(model.set_buildings_BT5, model.set_timeslots, within= pyo.Binary)
        model.variable_energyLevelBAT_BT5 = pyo.Var(model.set_buildings_BT5, model.set_timeslots, within=pyo.NonNegativeReals, bounds=(0, SetUpScenarios.capacityMaximal_BAT))
        model.variable_SOC_BAT_BT5= pyo.Var(model.set_buildings_BT5, model.set_timeslots,  within=pyo.NonNegativeReals, bounds=(0,100))
        
        
        model.variable_electricalPowerTotal_BT5 = pyo.Var(model.set_buildings_BT5, model.set_timeslots)
        
        model.variable_pvGeneration_BT5 = pyo.Var(model.set_buildings_BT5, model.set_timeslots)
        model.variable_windPowerAssigned_BT5 = pyo.Var(model.set_buildings_BT5, model.set_timeslots)
        
        
        # Defining the constraints
        
        
        #BAT Energy Level
        def energyLevelOfBATRule_BT5 (model, i, t):
            if t == model.set_timeslots.first():
                return model.variable_energyLevelBAT_BT5 [i, t] ==  ((SetUpScenarios.initialSOC_BAT/100) * SetUpScenarios.capacityMaximal_BAT) + ((model.variable_currentChargingPowerBAT_BT5 [i, t] * (SetUpScenarios.chargingEfficiency_BAT) - model.variable_currentDisChargingPowerBAT_BT5 [i, t] * (1 /(SetUpScenarios.chargingEfficiency_BAT))) * SetUpScenarios.timeResolution_InMinutes * 60 )
            return model.variable_energyLevelBAT_BT5[i, t]  == model.variable_energyLevelBAT_BT5 [i, t-1] + ((model.variable_currentChargingPowerBAT_BT5 [i, t] * (SetUpScenarios.chargingEfficiency_BAT) - model.variable_currentDisChargingPowerBAT_BT5 [i, t] *(1 /SetUpScenarios.dischargingEfficiency_BAT)) * SetUpScenarios.timeResolution_InMinutes * 60 )
        
        model.constraint_energyLevelOfBAT_BT5 = pyo.Constraint(model.set_buildings_BT5, model.set_timeslots, rule=energyLevelOfBATRule_BT5)
        
        
        #Constraints for the minimal and maximal energy level of the battery at the end of the optimization horizon
        def constraint_energyLevelOfBAT_lastLowerLimitRule_BT5 (model, i, t):
            return model.variable_energyLevelBAT_BT5[i, model.set_timeslots.last()] >= ((SetUpScenarios.initialSOC_BAT - SetUpScenarios.endSOC_BATAllowedDeviationFromInitalValueLowerLimit)/100) * SetUpScenarios.capacityMaximal_BAT
        
        model.constraint_energyLevelOfBAT_lastLowerLimit_BT5 = pyo.Constraint (model.set_buildings_BT5, model.set_timeslots, rule=constraint_energyLevelOfBAT_lastLowerLimitRule_BT5) 
        
        
        def constraint_energyLevelOfBAT_lastUpperLimitRule_BT5 (model, i, t):
            return model.variable_energyLevelBAT_BT5[i, model.set_timeslots.last()] <= ((SetUpScenarios.initialSOC_BAT + SetUpScenarios.endSOC_BATAllowedDeviationFromInitalValueUpperLimit)/100) * SetUpScenarios.capacityMaximal_BAT
        model.constraint_energyLevelOfBAT_lastUpperLimit_BT5 = pyo.Constraint (model.set_buildings_BT5, model.set_timeslots, rule=constraint_energyLevelOfBAT_lastUpperLimitRule_BT5)
        
        
        
        #SOC of the BAT
        def socOfBATRule_BT5 (model, i, t):
            return model.variable_SOC_BAT_BT5[i, t] == (model.variable_energyLevelBAT_BT5 [i, t] / SetUpScenarios.capacityMaximal_BAT)*100
        
        model.constraint_SOCofBAT_BT5 =  pyo.Constraint(model.set_buildings_BT5, model.set_timeslots, rule = socOfBATRule_BT5)
        
        
        #Constraint ensuring that the battery can't be charged and discharged at the same time (EQ1)
        def chargingDischargingPowerOfTheBATRule1_BT5 (model, i, t):
            return model.variable_currentChargingPowerBAT_BT5 [i, t]   <= SetUpScenarios.chargingPowerMaximal_BAT   * model.variable_helpBinaryChargingPositive_BT5 [i,t]
        
        model.constraint_chargingDischargingPowerOfTheBATRule1_BT5 = pyo.Constraint(model.set_buildings_BT5, model.set_timeslots, rule = chargingDischargingPowerOfTheBATRule1_BT5)


        #Constraint ensuring that the battery can't be charged and discharged at the same time (EQ2)
        def chargingDischargingPowerOfTheBATRule2_BT5 (model, i, t):
            return model.variable_currentDisChargingPowerBAT_BT5 [i, t]   <= SetUpScenarios.chargingPowerMaximal_BAT * (1- model.variable_helpBinaryChargingPositive_BT5 [i,t])

        model.constraint_chargingDischargingPowerOfTheBATRule2_BT5 = pyo.Constraint(model.set_buildings_BT5, model.set_timeslots, rule = chargingDischargingPowerOfTheBATRule2_BT5)


        #Constraint for limiting the disChargingPower to the current electricity demand
        def limitDisChargingPowerOfTheBAT_BT5 (model, i ,t ):
            return model.variable_currentDisChargingPowerBAT_BT5 [i, t] <=  model.param_electricalDemand_In_W_BT5 [i, t]

        model.constraint_limitDisChargingPowerOfTheBAT_BT5 = pyo.Constraint(model.set_buildings_BT5, model.set_timeslots, rule = limitDisChargingPowerOfTheBAT_BT5)

        #Constraints for the electrical power of BT5
        def electricalPowerTotalRule_BT5 (model,i, t):
            return model.variable_electricalPowerTotal_BT5 [i, t] ==  model.variable_currentChargingPowerBAT_BT5 [i, t] - model.variable_currentDisChargingPowerBAT_BT5 [i, t] + model.param_electricalDemand_In_W_BT5 [i, t] 
        
        model.constraint_electricalPowerTotal_BT5 = pyo.Constraint(model.set_buildings_BT5,model.set_timeslots, rule = electricalPowerTotalRule_BT5)
        
        
        
        #Equation for calculating the PV generation of each BT5-building
        def PVgenerationTotalRule_BT5 (model,i, t):
            return model.variable_pvGeneration_BT5 [i, t] == model.param_pvGenerationNominal_BT5 [i, t] * SetUpScenarios.determinePVPeakOfBuildings(SetUpScenarios.numberOfBuildings_BT1 + SetUpScenarios.numberOfBuildings_BT2 + i - 1) 
        model.constraint_PVgenerationTotal_BT5 = pyo.Constraint(model.set_buildings_BT5, model.set_timeslots, rule = PVgenerationTotalRule_BT5)
 
        
        #Equation for the assigned wind power of each BT5-building
        def windPowerAssignedgRule_BT5 (model,i, t):
            return model.variable_windPowerAssigned_BT5 [i, t] == model.param_windAssignedNominal_BT5 [i, t] * SetUpScenarios.maximalPowerOfWindTurbine 
            
        model.constraint_windPowerAssignedg_BT5 = pyo.Constraint(model.set_buildings_BT5,model.set_timeslots, rule = windPowerAssignedgRule_BT5)
    
    ###########################################################################################################################

    #Building Type 6 (BT6): Buildings with modulating gas boiler, an electrical heating element in a combined storage, and the building mass


    #Adjust dataframes to the current time resolution and set new index "Timeslot"

    for i in range (0, len(list_df_buildingData_BT6_original)):
        list_df_buildingData_BT6_original[i]['Time'] = pd.to_datetime(list_df_buildingData_BT6_original[i]['Time'], format = '%d.%m.%Y %H:%M')
        list_df_buildingData_BT6 [i] = list_df_buildingData_BT6_original[i].set_index('Time').resample(str(SetUpScenarios.timeResolution_InMinutes) +'Min').mean()

        arrayTimeSlots = [k for k in range (1,SetUpScenarios.numberOfTimeSlotsPerDay + 1)]
        list_df_buildingData_BT6 [i]['Timeslot'] = arrayTimeSlots
        list_df_buildingData_BT6 [i] = list_df_buildingData_BT6 [i].set_index('Timeslot')



    if SetUpScenarios.numberOfBuildings_BT6 >=1:
        #Create dataframes by using pandas series

        list_windProfileNominal_BT6 = [SetUpScenarios.calculateAssignedWindPowerNominalPerBuilding (currentDay,SetUpScenarios.numberOfBuildings_BT1 + index_BT6) for index_BT6 in range(0, SetUpScenarios.numberOfBuildings_BT6)]
        list_df_windPowerAssignedNominalPerBuilding_BT6 = [pd.DataFrame({'Timeslot': list_df_buildingData_BT6 [i].index, 'Wind [nominal]':list_windProfileNominal_BT6[i] }) for i in range (0, SetUpScenarios.numberOfBuildings_BT6)]

        for i in range (0, len(list_df_windPowerAssignedNominalPerBuilding_BT6)):
            del list_df_windPowerAssignedNominalPerBuilding_BT6[i]['Timeslot']
            list_df_windPowerAssignedNominalPerBuilding_BT6[i].index +=1

        combinedDataframe_heatDemand_BT6 = pd.DataFrame()
        combinedDataframe_DHWDemand_BT6 = pd.DataFrame()
        combinedDataframe_electricalDemand_BT6 = pd.DataFrame()
        combinedDataframe_pvGenerationNominal_BT6 = pd.DataFrame()
        combinedDataframe_windAssigned_BT6 = pd.DataFrame()



        for index in range (0, len(list_df_buildingData_BT6)):
            combinedDataframe_heatDemand_BT6[index] = list_df_buildingData_BT6[index] ["Space Heating [W]"]
            combinedDataframe_DHWDemand_BT6[index] = list_df_buildingData_BT6[index] ["DHW [W]"]
            combinedDataframe_electricalDemand_BT6[index] = list_df_buildingData_BT6[index] ["Electricity [W]"]
            combinedDataframe_pvGenerationNominal_BT6[index] = list_df_buildingData_BT6[index] ["PV [nominal]"]
            combinedDataframe_windAssigned_BT6 [index]= list_df_windPowerAssignedNominalPerBuilding_BT6[index] ["Wind [nominal]"]



        #Round the values
        for index in range (0,  SetUpScenarios.numberOfBuildings_BT6):
            decimalsForRounding = 2
            list_df_buildingData_BT6 [index]['Space Heating [W]'] = list_df_buildingData_BT6 [index]['Space Heating [W]'].apply(lambda x: round(x, decimalsForRounding))
            list_df_buildingData_BT6 [index]['DHW [W]'] = list_df_buildingData_BT6 [index]['DHW [W]'].apply(lambda x: round(x, decimalsForRounding))
            list_df_buildingData_BT6 [index]['Electricity [W]'] = list_df_buildingData_BT6 [index]['Electricity [W]'].apply(lambda x: round(x, decimalsForRounding))
            decimalsForRounding = 4
            list_df_buildingData_BT6 [index]['PV [nominal]'] = list_df_buildingData_BT6 [index]['PV [nominal]'].apply(lambda x: round(x, decimalsForRounding))


        #Define the parameters of the model in pyomo
        def init_heatDemand (model, i,j):
            return combinedDataframe_heatDemand_BT6.iloc[j-1, i-1]

        model.param_heatDemand_In_W_BT6 = pyo.Param(model.set_buildings_BT6, model.set_timeslots, mutable = True, initialize=init_heatDemand)


        def init_DHWDemand (model, i,j):
            return combinedDataframe_DHWDemand_BT6.iloc[j-1, i-1]

        model.param_DHWDemand_In_W_BT6 = pyo.Param(model.set_buildings_BT6, model.set_timeslots,mutable = True, initialize=init_DHWDemand)


        def init_electricalDemand (model, i,j):
            return combinedDataframe_electricalDemand_BT6.iloc[j-1, i-1]

        model.param_electricalDemand_In_W_BT6 = pyo.Param(model.set_buildings_BT6, model.set_timeslots,mutable = True, initialize=init_electricalDemand)


        def init_pvGenerationNominal (model, i,j):
            return combinedDataframe_pvGenerationNominal_BT6.iloc[j-1, i-1]

        model.param_pvGenerationNominal_BT6  = pyo.Param(model.set_buildings_BT6, model.set_timeslots, mutable = True, initialize=init_pvGenerationNominal)


        model.param_outSideTemperature_In_C = pyo.Param(model.set_timeslots, initialize=dictionaryTemperature_In_C)


        def init_windAssignedNominal (model, i,j):
            return combinedDataframe_windAssigned_BT6.iloc[j-1, i-1]

        model.param_windAssignedNominal_BT6  = pyo.Param(model.set_buildings_BT6, model.set_timeslots, mutable = True, initialize=init_windAssignedNominal)



        #Define the variables

        model.variable_heatGenerationCoefficient_GasBoiler_BT6 = pyo.Var(model.set_buildings_BT6, model.set_timeslots,bounds=(0,1))
        model.variable_heatGenerationCoefficient_ElectricalHeatingElement_BT6 = pyo.Var(model.set_buildings_BT6, model.set_timeslots,bounds=(0,1))
        model.variable_heatTransferCoefficient_StorageToRoom_BT6 = pyo.Var(model.set_buildings_BT6, model.set_timeslots,bounds=(0, 1))
        model.variable_temperatureBuilding_BT6 = pyo.Var(model.set_buildings_BT6, model.set_timeslots, bounds=(SetUpScenarios.minimumTemperatureBuilding, SetUpScenarios.maximumTemperatureBuilding))

        model.variable_energyLevelCombinedStorage_BT6 = pyo.Var(model.set_buildings_BT6, model.set_timeslots,  bounds=(0 , SetUpScenarios.maximumEnergyContentCombinedStorage))

        model.variable_electricalPowerTotal_BT6 = pyo.Var(model.set_buildings_BT6, model.set_timeslots)
        model.variable_pvGeneration_BT6 = pyo.Var(model.set_buildings_BT6, model.set_timeslots)
        model.variable_windPowerAssigned_BT6 = pyo.Var(model.set_buildings_BT6, model.set_timeslots)
        
        
        
        model.variable_deviation_temperature_positive_BT6 = pyo.Var(model.set_buildings_BT6, model.set_timeslots, within=pyo.NonNegativeReals)
        model.variable_deviation_temperature_negative_BT6 = pyo.Var(model.set_buildings_BT6, model.set_timeslots, within=pyo.NonNegativeReals)
        

        # Defining the constraints

        #Energy constraint for the combined storage
        def energyCombinedStorageConstraintRule_BT6(model, i, t):
            if t == model.set_timeslots.first():
                return model.variable_energyLevelCombinedStorage_BT6[i, t] == SetUpScenarios.initialEnergyContentCombinedStorage + (model.variable_heatGenerationCoefficient_GasBoiler_BT6[i, t] * SetUpScenarios.maximalPowerGasBoiler * SetUpScenarios.efficiency_GasBoiler + model.variable_heatGenerationCoefficient_ElectricalHeatingElement_BT6[i, t] * SetUpScenarios.maximalPowerElectricalHeatingElement * SetUpScenarios.efficiency_ElectricalHeatingElement - model.variable_heatTransferCoefficient_StorageToRoom_BT6[i, t] * SetUpScenarios.maximalPowerHeatingSystem  - SetUpScenarios.standingLossesCombinedStorage )  * (SetUpScenarios.timeResolution_InMinutes * 60)
            return model.variable_energyLevelCombinedStorage_BT6[i, t] == model.variable_energyLevelCombinedStorage_BT6[i, t-1] + (model.variable_heatGenerationCoefficient_GasBoiler_BT6[i, t] * SetUpScenarios.maximalPowerGasBoiler * SetUpScenarios.efficiency_GasBoiler + model.variable_heatGenerationCoefficient_ElectricalHeatingElement_BT6[i, t] * SetUpScenarios.maximalPowerElectricalHeatingElement * SetUpScenarios.efficiency_ElectricalHeatingElement - model.variable_heatTransferCoefficient_StorageToRoom_BT6[i, t] * SetUpScenarios.maximalPowerHeatingSystem   - SetUpScenarios.standingLossesCombinedStorage)  * (SetUpScenarios.timeResolution_InMinutes * 60)

        model.constraint_energyCombinedStorage_BT6= pyo.Constraint (model.set_buildings_BT6, model.set_timeslots, rule=energyCombinedStorageConstraintRule_BT6)



        #Temperature constraint for the buffer storage (space heating) with energetic difference equation
        def temperatureBuildingConstraintRule_BT6(model, i, t):
            if t == model.set_timeslots.first():
                return model.variable_temperatureBuilding_BT6[i, t] == SetUpScenarios.initialTemperatureBuilding + ((model.variable_heatTransferCoefficient_StorageToRoom_BT6[i, t] * SetUpScenarios.maximalPowerHeatingSystem  - model.param_heatDemand_In_W_BT6 [i, t])*SetUpScenarios.timeResolution_InMinutes * 60)/(SetUpScenarios.totalHeatCapacityOfTheBuilding)
            return model.variable_temperatureBuilding_BT6[i, t] == model.variable_temperatureBuilding_BT6[i, t-1] + ((model.variable_heatTransferCoefficient_StorageToRoom_BT6[i, t] * SetUpScenarios.maximalPowerHeatingSystem  - model.param_heatDemand_In_W_BT6 [i, t])*SetUpScenarios.timeResolution_InMinutes * 60)/(SetUpScenarios.totalHeatCapacityOfTheBuilding)

        model.constraint_temperatureBuilding_BT6= pyo.Constraint (model.set_buildings_BT6, model.set_timeslots, rule=temperatureBuildingConstraintRule_BT6)

        #Constraint for allowing thermal discomfort
        def temperatureDeviationConstraintRule1_BT6 (model, i, t):
            if optParameters['optimizationGoal_minimizeThermalDiscomfort'] == True:
                return  SetUpScenarios.idealComfortTemperature - SetUpScenarios.allowedTemperatureDeviationForOptimalComfort <= model.variable_temperatureBuilding_BT6[i, t] - model.variable_deviation_temperature_positive_BT6 [i, t] +  model.variable_deviation_temperature_negative_BT6 [i, t]
            else:
                return pyo.Constraint.Skip

        model.constraint_temperatureDeviationConstraint1_BT6 = pyo.Constraint(model.set_buildings_BT6, model.set_timeslots, rule = temperatureDeviationConstraintRule1_BT6)


        def temperatureDeviationConstraintRule2_BT6 (model, i, t):
            if optParameters['optimizationGoal_minimizeThermalDiscomfort'] == True:
                return  model.variable_temperatureBuilding_BT6[i, t] - model.variable_deviation_temperature_positive_BT6 [i, t] +  model.variable_deviation_temperature_negative_BT6 [i, t]<= SetUpScenarios.idealComfortTemperature + SetUpScenarios.allowedTemperatureDeviationForOptimalComfort
            else:
                return pyo.Constraint.Skip

        model.constraint_temperatureDeviationConstraint2_BT6 = pyo.Constraint(model.set_buildings_BT6, model.set_timeslots, rule = temperatureDeviationConstraintRule2_BT6)




        #Constraint for ensuring that the energy taken from the combined storage to the rooms is smaller than the energy content of the combined storage
        def energyTransferLimitFromCombinedStorageRule (model, i, t):
            return model.variable_heatTransferCoefficient_StorageToRoom_BT6[i, t] * SetUpScenarios.maximalPowerHeatingSystem <= model.variable_energyLevelCombinedStorage_BT6[i, t]

        model.constraint_energyTransferLimitFromCombinedStorage = pyo.Constraint(model.set_buildings_BT6, model.set_timeslots, rule= energyTransferLimitFromCombinedStorageRule)



        #Constraints for the minimal and maximal temperature at the end of the optimization horizon
        def temperatureBuilding_lastLowerLimitRule_BT6 (model, i, t):
            return model.variable_temperatureBuilding_BT6[i, model.set_timeslots.last()] >= SetUpScenarios.initialTemperatureBuilding - SetUpScenarios.endTemperatureBuildingAllowedDeviationFromInitalValue

        model.constraint_temperatureBuilding_lastLowerLimit_BT6 = pyo.Constraint (model.set_buildings_BT6, model.set_timeslots, rule=temperatureBuilding_lastLowerLimitRule_BT6)


        def temperatureBuilding_lastUpperLimitRule_BT6 (model, i, t):
            return model.variable_temperatureBuilding_BT6[i, model.set_timeslots.last()] <= SetUpScenarios.initialTemperatureBuilding + SetUpScenarios.endTemperatureBuildingAllowedDeviationFromInitalValue

        model.constraint_temperatureBuilding_lastUpperLimit_BT6 = pyo.Constraint (model.set_buildings_BT6, model.set_timeslots, rule=temperatureBuilding_lastUpperLimitRule_BT6)


        #Constraints for the electrical power of BT6
        def electricalPowerTotalRule_BT6 (model,i, t):
            return model.variable_electricalPowerTotal_BT6 [i, t] == model.variable_heatGenerationCoefficient_ElectricalHeatingElement_BT6[i, t] * SetUpScenarios.maximalPowerElectricalHeatingElement   + model.param_electricalDemand_In_W_BT6 [i, t]

        model.constraint_electricalPowerTotal_BT6 = pyo.Constraint(model.set_buildings_BT6,model.set_timeslots, rule = electricalPowerTotalRule_BT6)



        #Equation for calculating the PV generation of each BT6-building
        def PVgenerationTotalRule_BT6 (model,i, t):
            return model.variable_pvGeneration_BT6 [i, t] == model.param_pvGenerationNominal_BT6 [i, t] * SetUpScenarios.determinePVPeakOfBuildings(SetUpScenarios.numberOfBuildings_BT1 + SetUpScenarios.numberOfBuildings_BT2 + SetUpScenarios.numberOfBuildings_BT3 + SetUpScenarios.numberOfBuildings_BT4 +SetUpScenarios.numberOfBuildings_BT5 + i - 1)
        model.constraint_PVgenerationTotal_BT6 = pyo.Constraint(model.set_buildings_BT6, model.set_timeslots, rule = PVgenerationTotalRule_BT6)


        #Equation for the assigned wind power of each BT6-building
        def windPowerAssignedgRule_BT6 (model,i, t):
            return model.variable_windPowerAssigned_BT6 [i, t] == model.param_windAssignedNominal_BT6 [i, t] * SetUpScenarios.maximalPowerOfWindTurbine

        model.constraint_windPowerAssignedg_BT6 = pyo.Constraint(model.set_buildings_BT6,model.set_timeslots, rule = windPowerAssignedgRule_BT6)




        #Constraint system for the maximum number of starts of the heat pump

        model.variable_HPswitchedOff_Individual_GasHeating_BT6 = pyo.Var(model.set_buildings_BT6, model.set_timeslots, within =pyo.Binary, initialize=0.0)
        model.variable_HPswitchedOff_HelpAssociatedBinary_GasHeating_BT6 = pyo.Var(model.set_buildings_BT6, model.set_timeslots, within =pyo.Binary, initialize=0.0)
        model.variable_HPswitchedOff_HelpModulationBinary_GasHeating_BT6 = pyo.Var(model.set_buildings_BT6, model.set_timeslots, within =pyo.Binary)

        #Constraints for maximum number of starts for the space heating

        def maximumNumberOfStarts_Individual_GasHeating_EQ1_Rule_BT6 (model, i, t):
            if Run_Simulations.considerMaxiumNumberOfStartsGas_Individual ==True:
                if t == model.set_timeslots.first():
                    return model.variable_HPswitchedOff_Individual_GasHeating_BT6 [i, t] == 0
                return model.variable_HPswitchedOff_Individual_GasHeating_BT6 [i, t] <= model.variable_HPswitchedOff_HelpAssociatedBinary_GasHeating_BT6 [i, t-1]
            if Run_Simulations.considerMaxiumNumberOfStartsGas_Individual ==False:
                return pyo.Constraint.Skip

        model.constraint_maximumNumberOfStarts_Individual_GasHeating_EQ1_BT4 = pyo.Constraint(model.set_buildings_BT6, model.set_timeslots, rule =maximumNumberOfStarts_Individual_GasHeating_EQ1_Rule_BT6)


        def maximumNumberOfStarts_Individual_GasHeating_EQ2_Rule_BT6 (model, i,  t):
            if Run_Simulations.considerMaxiumNumberOfStartsGas_Individual ==True:
                return model.variable_HPswitchedOff_HelpAssociatedBinary_GasHeating_BT6 [i, t] + model.variable_HPswitchedOff_Individual_GasHeating_BT6 [i, t] <= 1
            if Run_Simulations.considerMaxiumNumberOfStartsGas_Individual ==False:
                return pyo.Constraint.Skip

        model.constraint_maximumNumberOfStarts_Individual_GasHeating_EQ2_BT4 = pyo.Constraint(model.set_buildings_BT6, model.set_timeslots, rule =maximumNumberOfStarts_Individual_GasHeating_EQ2_Rule_BT6)


        def maximumNumberOfStarts_Individual_GasHeating_EQ2_2_Rule_BT6 (model, i, t):
            if Run_Simulations.considerMaxiumNumberOfStartsGas_Individual ==True:
                if t == model.set_timeslots.first():
                    return model.variable_HPswitchedOff_Individual_GasHeating_BT6 [i, t] == 0
                return model.variable_HPswitchedOff_HelpAssociatedBinary_GasHeating_BT6 [i, t - 1] <= model.variable_HPswitchedOff_HelpAssociatedBinary_GasHeating_BT6 [i, t] + model.variable_HPswitchedOff_Individual_GasHeating_BT6 [i, t]
            if Run_Simulations.considerMaxiumNumberOfStartsGas_Individual ==False:
                return pyo.Constraint.Skip

        model.constraint_maximumNumberOfStarts_Individual_GasHeating_EQ2_2_BT6 = pyo.Constraint(model.set_buildings_BT6, model.set_timeslots, rule =maximumNumberOfStarts_Individual_GasHeating_EQ2_2_Rule_BT6)



        def maximumNumberOfStarts_Individual_GasHeating_EQ3_HelpAssociatedBinary_Rule_BT6 (model, i, t):
            if Run_Simulations.considerMaxiumNumberOfStartsGas_Individual ==True :
                return model.variable_HPswitchedOff_HelpAssociatedBinary_GasHeating_BT6 [i, t] >= model.variable_heatGenerationCoefficient_GasBoiler_BT6[i, t]
            if Run_Simulations.considerMaxiumNumberOfStartsGas_Individual ==False:
               return pyo.Constraint.Skip

        model.constraint_maximumNumberOfStarts_Individual_GasHeating_EQ3_HelpAssociatedBinary_BT6 = pyo.Constraint(model.set_buildings_BT6, model.set_timeslots, rule =maximumNumberOfStarts_Individual_GasHeating_EQ3_HelpAssociatedBinary_Rule_BT6)



        def maximumNumberOfStarts_Individual_Gaseating_EQ4_HelpAssociatedBinary_Rule_BT6 (model, i, t):
            if Run_Simulations.considerMaxiumNumberOfStartsGas_Individual ==True :
                return model.variable_heatGenerationCoefficient_GasBoiler_BT6[i, t] * (1/(SetUpScenarios.minimalModulationdDegree_GasBoiler/100))  >= model.variable_HPswitchedOff_HelpAssociatedBinary_GasHeating_BT6 [i, t]
            if Run_Simulations.considerMaxiumNumberOfStartsGas_Individual ==False:
               return pyo.Constraint.Skip

        model.constraint_maximumNumberOfStarts_Individual_GasHeating_EQ4_HelpAssociatedBinary_BT6 = pyo.Constraint(model.set_buildings_BT6, model.set_timeslots, rule =maximumNumberOfStarts_Individual_Gaseating_EQ4_HelpAssociatedBinary_Rule_BT6)



        def maximumNumberOfStarts_Individual_GasHeating_EQ5_NumberOfStarts_Rule_BT6 (model, i, t):
            if Run_Simulations.considerMaxiumNumberOfStartsGas_Individual ==True:
                return  sum (model.variable_HPswitchedOff_Individual_GasHeating_BT6 [i, t] for t in model.set_timeslots)<= Run_Simulations.maximumNumberOfStarts_Individual
            if Run_Simulations.considerMaxiumNumberOfStartsGas_Individual ==False:
                return pyo.Constraint.Skip

        model.constraint_maximumNumberOfStarts_Individual_GasHeating_EQ5_NumberOfStarts_BT6 = pyo.Constraint(model.set_buildings_BT6, model.set_timeslots, rule =maximumNumberOfStarts_Individual_GasHeating_EQ5_NumberOfStarts_Rule_BT6)

   ###########################################################################################################################

    #Building Type 7 (BT7): Buildings with modulating gas boiler, an electrical fan heater and no hot water tank (only building mass servers as storage)


    #Adjust dataframes to the current time resolution and set new index "Timeslot"

    for i in range (0, len(list_df_buildingData_BT7_original)):
        list_df_buildingData_BT7_original[i]['Time'] = pd.to_datetime(list_df_buildingData_BT7_original[i]['Time'], format = '%d.%m.%Y %H:%M')
        list_df_buildingData_BT7 [i] = list_df_buildingData_BT7_original[i].set_index('Time').resample(str(SetUpScenarios.timeResolution_InMinutes) +'Min').mean()

        arrayTimeSlots = [k for k in range (1,SetUpScenarios.numberOfTimeSlotsPerDay + 1)]
        list_df_buildingData_BT7 [i]['Timeslot'] = arrayTimeSlots
        list_df_buildingData_BT7 [i] = list_df_buildingData_BT7 [i].set_index('Timeslot')



    if SetUpScenarios.numberOfBuildings_BT7 >=1:
        #Create dataframes by using pandas series

        list_windProfileNominal_BT7 = [SetUpScenarios.calculateAssignedWindPowerNominalPerBuilding (currentDay,SetUpScenarios.numberOfBuildings_BT1 + index_BT7) for index_BT7 in range(0, SetUpScenarios.numberOfBuildings_BT7)]
        list_df_windPowerAssignedNominalPerBuilding_BT7 = [pd.DataFrame({'Timeslot': list_df_buildingData_BT7 [i].index, 'Wind [nominal]':list_windProfileNominal_BT7[i] }) for i in range (0, SetUpScenarios.numberOfBuildings_BT7)]

        for i in range (0, len(list_df_windPowerAssignedNominalPerBuilding_BT7)):
            del list_df_windPowerAssignedNominalPerBuilding_BT7[i]['Timeslot']
            list_df_windPowerAssignedNominalPerBuilding_BT7[i].index +=1

        combinedDataframe_heatDemand_BT7 = pd.DataFrame()
        combinedDataframe_electricalDemand_BT7 = pd.DataFrame()
        combinedDataframe_pvGenerationNominal_BT7 = pd.DataFrame()
        combinedDataframe_windAssigned_BT7 = pd.DataFrame()



        for index in range (0, len(list_df_buildingData_BT7)):
            combinedDataframe_heatDemand_BT7[index] = list_df_buildingData_BT7[index] ["Space Heating [W]"]
            combinedDataframe_electricalDemand_BT7[index] = list_df_buildingData_BT7[index] ["Electricity [W]"]
            combinedDataframe_pvGenerationNominal_BT7[index] = list_df_buildingData_BT7[index] ["PV [nominal]"]
            combinedDataframe_windAssigned_BT7 [index]= list_df_windPowerAssignedNominalPerBuilding_BT7[index] ["Wind [nominal]"]



        #Round the values
        for index in range (0,  SetUpScenarios.numberOfBuildings_BT7):
            decimalsForRounding = 2
            list_df_buildingData_BT7 [index]['Space Heating [W]'] = list_df_buildingData_BT7 [index]['Space Heating [W]'].apply(lambda x: round(x, decimalsForRounding))
            list_df_buildingData_BT7 [index]['Electricity [W]'] = list_df_buildingData_BT7 [index]['Electricity [W]'].apply(lambda x: round(x, decimalsForRounding))
            decimalsForRounding = 4
            list_df_buildingData_BT7 [index]['PV [nominal]'] = list_df_buildingData_BT7 [index]['PV [nominal]'].apply(lambda x: round(x, decimalsForRounding))


        #Define the parameters of the model in pyomo
        def init_heatDemand (model, i,j):
            return combinedDataframe_heatDemand_BT7.iloc[j-1, i-1]

        model.param_heatDemand_In_W_BT7 = pyo.Param(model.set_buildings_BT7, model.set_timeslots, mutable = True, initialize=init_heatDemand)


        def init_electricalDemand (model, i,j):
            return combinedDataframe_electricalDemand_BT7.iloc[j-1, i-1]

        model.param_electricalDemand_In_W_BT7 = pyo.Param(model.set_buildings_BT7, model.set_timeslots,mutable = True, initialize=init_electricalDemand)


        def init_pvGenerationNominal (model, i,j):
            return combinedDataframe_pvGenerationNominal_BT7.iloc[j-1, i-1]

        model.param_pvGenerationNominal_BT7  = pyo.Param(model.set_buildings_BT7, model.set_timeslots, mutable = True, initialize=init_pvGenerationNominal)


        model.param_outSideTemperature_In_C = pyo.Param(model.set_timeslots, initialize=dictionaryTemperature_In_C)


        def init_windAssignedNominal (model, i,j):
            return combinedDataframe_windAssigned_BT7.iloc[j-1, i-1]

        model.param_windAssignedNominal_BT7  = pyo.Param(model.set_buildings_BT7, model.set_timeslots, mutable = True, initialize=init_windAssignedNominal)



        #Define the variables

        model.variable_heatGenerationCoefficient_GasBoiler_BT7 = pyo.Var(model.set_buildings_BT7, model.set_timeslots,bounds=(0,1))
        model.variable_heatGenerationCoefficient_FanHeater_Stage1_BT7 = pyo.Var(model.set_buildings_BT7, model.set_timeslots,within =pyo.Binary, initialize=0.0)
        model.variable_heatGenerationCoefficient_FanHeater_Stage2_BT7 = pyo.Var(model.set_buildings_BT7, model.set_timeslots,within =pyo.Binary, initialize=0.0)
        model.variable_heatGenerationCoefficient_FanHeater_Stage3_BT7 = pyo.Var(model.set_buildings_BT7, model.set_timeslots,within =pyo.Binary, initialize=0.0)

        model.variable_temperatureBuilding_BT7 = pyo.Var(model.set_buildings_BT7, model.set_timeslots, bounds=(SetUpScenarios.minimumTemperatureBuilding, SetUpScenarios.maximumTemperatureBuilding))

        model.variable_electricalPowerTotal_BT7 = pyo.Var(model.set_buildings_BT7, model.set_timeslots)
        model.variable_electricalPowerFanHeater_BT7 = pyo.Var(model.set_buildings_BT7, model.set_timeslots)
        model.variable_pvGeneration_BT7 = pyo.Var(model.set_buildings_BT7, model.set_timeslots)
        model.variable_windPowerAssigned_BT7 = pyo.Var(model.set_buildings_BT7, model.set_timeslots)
        
                
        model.variable_deviation_temperature_positive_BT7 = pyo.Var(model.set_buildings_BT7, model.set_timeslots, within=pyo.NonNegativeReals)
        model.variable_deviation_temperature_negative_BT7 = pyo.Var(model.set_buildings_BT7, model.set_timeslots, within=pyo.NonNegativeReals)




        # Defining the constraints


        #Temperature constraint for the buffer storage (space heating) with energetic difference equation
        def temperatureBuildingConstraintRule_BT7(model, i, t):
            if t == model.set_timeslots.first():
                return model.variable_temperatureBuilding_BT7[i, t] == SetUpScenarios.initialTemperatureBuilding + ((model.variable_heatGenerationCoefficient_GasBoiler_BT7[i, t] * SetUpScenarios.maximalPowerGasBoiler * SetUpScenarios.efficiency_GasBoiler + model.variable_electricalPowerFanHeater_BT7[i, t] * SetUpScenarios.efficiency_ElectricalFanHeater  - model.param_heatDemand_In_W_BT7 [i, t])*SetUpScenarios.timeResolution_InMinutes * 60)/(SetUpScenarios.totalHeatCapacityOfTheBuilding)
            return model.variable_temperatureBuilding_BT7[i, t] == model.variable_temperatureBuilding_BT7[i, t-1] + ((model.variable_heatGenerationCoefficient_GasBoiler_BT7[i, t] * SetUpScenarios.maximalPowerGasBoiler * SetUpScenarios.efficiency_GasBoiler + model.variable_electricalPowerFanHeater_BT7[i, t] * SetUpScenarios.efficiency_ElectricalFanHeater  - model.param_heatDemand_In_W_BT7 [i, t])*SetUpScenarios.timeResolution_InMinutes * 60)/(SetUpScenarios.totalHeatCapacityOfTheBuilding)

        model.constraint_temperatureBuilding_BT7= pyo.Constraint (model.set_buildings_BT7, model.set_timeslots, rule=temperatureBuildingConstraintRule_BT7)


        #Constraint for allowing thermal discomfort
        def temperatureDeviationConstraintRule1_BT7 (model, i, t):
            if optParameters['optimizationGoal_minimizeThermalDiscomfort'] == True:
                return  SetUpScenarios.idealComfortTemperature - SetUpScenarios.allowedTemperatureDeviationForOptimalComfort <= model.variable_temperatureBuilding_BT7[i, t] - model.variable_deviation_temperature_positive_BT7 [i, t] +  model.variable_deviation_temperature_negative_BT7 [i, t]
            else:
                return pyo.Constraint.Skip

        model.constraint_temperatureDeviationConstraint1_BT7 = pyo.Constraint(model.set_buildings_BT7, model.set_timeslots, rule = temperatureDeviationConstraintRule1_BT7)


        def temperatureDeviationConstraintRule2_BT7 (model, i, t):
            if optParameters['optimizationGoal_minimizeThermalDiscomfort'] == True:
                return  model.variable_temperatureBuilding_BT7[i, t] - model.variable_deviation_temperature_positive_BT7 [i, t] +  model.variable_deviation_temperature_negative_BT7 [i, t]<= SetUpScenarios.idealComfortTemperature + SetUpScenarios.allowedTemperatureDeviationForOptimalComfort
            else:
                return pyo.Constraint.Skip

        model.constraint_temperatureDeviationConstraint2_BT7 = pyo.Constraint(model.set_buildings_BT7, model.set_timeslots, rule = temperatureDeviationConstraintRule2_BT7)



        #Constraints for the minimal and maximal temperature at the end of the optimization horizon
        def temperatureBuilding_lastLowerLimitRule_BT7 (model, i, t):
            return model.variable_temperatureBuilding_BT7[i, model.set_timeslots.last()] >= SetUpScenarios.initialTemperatureBuilding - SetUpScenarios.endTemperatureBuildingAllowedDeviationFromInitalValue

        model.constraint_temperatureBuilding_lastLowerLimit_BT7 = pyo.Constraint (model.set_buildings_BT7, model.set_timeslots, rule=temperatureBuilding_lastLowerLimitRule_BT7)


        def temperatureBuilding_lastUpperLimitRule_BT7 (model, i, t):
            return model.variable_temperatureBuilding_BT7[i, model.set_timeslots.last()] <= SetUpScenarios.initialTemperatureBuilding + SetUpScenarios.endTemperatureBuildingAllowedDeviationFromInitalValue

        model.constraint_temperatureBuilding_lastUpperLimit_BT7 = pyo.Constraint (model.set_buildings_BT7, model.set_timeslots, rule=temperatureBuilding_lastUpperLimitRule_BT7)

        #Constraint for the electrical heating power of the fan heater
        def electricalHeatingPowerTotalRule_BT7 (model, i, t):
            return model.variable_electricalPowerFanHeater_BT7 [i,t] == model.variable_heatGenerationCoefficient_FanHeater_Stage1_BT7 [i,t] * SetUpScenarios.electricalPowerFanHeater_Stage1 + model.variable_heatGenerationCoefficient_FanHeater_Stage2_BT7 [i,t] * SetUpScenarios.electricalPowerFanHeater_Stage2 + model.variable_heatGenerationCoefficient_FanHeater_Stage3_BT7 [i,t] * SetUpScenarios.electricalPowerFanHeater_Stage3

        model.constraint_electricalHeatingPowerTotal_BT7 = pyo.Constraint(model.set_buildings_BT7, model.set_timeslots, rule =electricalHeatingPowerTotalRule_BT7 )

        def electricalHeatingPowerOnlyOneStageRule_BT7  (model, i, t):
            return model.variable_heatGenerationCoefficient_FanHeater_Stage1_BT7 [i,t] + model.variable_heatGenerationCoefficient_FanHeater_Stage2_BT7 [i,t] + model.variable_heatGenerationCoefficient_FanHeater_Stage3_BT7 [i,t] <= 1

        model.constraint_electricalHeatingPowerOnlyOneStage = pyo.Constraint(model.set_buildings_BT7, model.set_timeslots, rule = electricalHeatingPowerOnlyOneStageRule_BT7)


        #Constraints for the electrical power of BT7
        def electricalPowerTotalRule_BT7 (model,i, t):
            return model.variable_electricalPowerTotal_BT7 [i, t] == model.variable_electricalPowerFanHeater_BT7 [i,t]  + model.param_electricalDemand_In_W_BT7 [i, t]

        model.constraint_electricalPowerTotal_BT7 = pyo.Constraint(model.set_buildings_BT7,model.set_timeslots, rule = electricalPowerTotalRule_BT7)



        #Equation for calculating the PV generation of each BT7-building
        def PVgenerationTotalRule_BT7 (model,i, t):
            return model.variable_pvGeneration_BT7 [i, t] == model.param_pvGenerationNominal_BT7 [i, t] * SetUpScenarios.determinePVPeakOfBuildings(SetUpScenarios.numberOfBuildings_BT1 + SetUpScenarios.numberOfBuildings_BT2 + SetUpScenarios.numberOfBuildings_BT3 + SetUpScenarios.numberOfBuildings_BT4 +SetUpScenarios.numberOfBuildings_BT5 +SetUpScenarios.numberOfBuildings_BT6 + i - 1)
        model.constraint_PVgenerationTotal_BT7 = pyo.Constraint(model.set_buildings_BT7, model.set_timeslots, rule = PVgenerationTotalRule_BT7)


        #Equation for the assigned wind power of each BT7-building
        def windPowerAssignedgRule_BT7 (model,i, t):
            return model.variable_windPowerAssigned_BT7 [i, t] == model.param_windAssignedNominal_BT7 [i, t] * SetUpScenarios.maximalPowerOfWindTurbine

        model.constraint_windPowerAssignedg_BT7 = pyo.Constraint(model.set_buildings_BT7,model.set_timeslots, rule = windPowerAssignedgRule_BT7)




        #Constraint system for the maximum number of starts of the heat pump

        model.variable_HPswitchedOff_Individual_GasHeating_BT7 = pyo.Var(model.set_buildings_BT7, model.set_timeslots, within =pyo.Binary, initialize=0.0)
        model.variable_HPswitchedOff_HelpAssociatedBinary_GasHeating_BT7 = pyo.Var(model.set_buildings_BT7, model.set_timeslots, within =pyo.Binary, initialize=0.0)
        model.variable_HPswitchedOff_HelpModulationBinary_GasHeating_BT7 = pyo.Var(model.set_buildings_BT7, model.set_timeslots, within =pyo.Binary)


        #Constraints for maximum number of starts for the space heating

        def maximumNumberOfStarts_Individual_GasHeating_EQ1_Rule_BT7 (model, i, t):
            if Run_Simulations.considerMaxiumNumberOfStartsGas_Individual ==True:
                if t == model.set_timeslots.first():
                    return model.variable_HPswitchedOff_Individual_GasHeating_BT7 [i, t] == 0
                return model.variable_HPswitchedOff_Individual_GasHeating_BT7 [i, t] <= model.variable_HPswitchedOff_HelpAssociatedBinary_GasHeating_BT7 [i, t-1]
            if Run_Simulations.considerMaxiumNumberOfStartsGas_Individual ==False:
                return pyo.Constraint.Skip

        model.constraint_maximumNumberOfStarts_Individual_GasHeating_EQ1_BT7 = pyo.Constraint(model.set_buildings_BT7, model.set_timeslots, rule =maximumNumberOfStarts_Individual_GasHeating_EQ1_Rule_BT7)


        def maximumNumberOfStarts_Individual_GasHeating_EQ2_Rule_BT7 (model, i,  t):
            if Run_Simulations.considerMaxiumNumberOfStartsGas_Individual ==True:
                return model.variable_HPswitchedOff_HelpAssociatedBinary_GasHeating_BT7 [i, t] + model.variable_HPswitchedOff_Individual_GasHeating_BT7 [i, t] <= 1
            if Run_Simulations.considerMaxiumNumberOfStartsGas_Individual ==False:
                return pyo.Constraint.Skip

        model.constraint_maximumNumberOfStarts_Individual_GasHeating_EQ2_BT7 = pyo.Constraint(model.set_buildings_BT7, model.set_timeslots, rule =maximumNumberOfStarts_Individual_GasHeating_EQ2_Rule_BT7)


        def maximumNumberOfStarts_Individual_GasHeating_EQ2_2_Rule_BT7 (model, i, t):
            if Run_Simulations.considerMaxiumNumberOfStartsGas_Individual ==True:
                if t == model.set_timeslots.first():
                    return model.variable_HPswitchedOff_Individual_GasHeating_BT7 [i, t] == 0
                return model.variable_HPswitchedOff_HelpAssociatedBinary_GasHeating_BT7 [i, t - 1] <= model.variable_HPswitchedOff_HelpAssociatedBinary_GasHeating_BT7 [i, t] + model.variable_HPswitchedOff_Individual_GasHeating_BT7 [i, t]
            if Run_Simulations.considerMaxiumNumberOfStartsGas_Individual ==False:
                return pyo.Constraint.Skip

        model.constraint_maximumNumberOfStarts_Individual_GasHeating_EQ2_2_BT7 = pyo.Constraint(model.set_buildings_BT7, model.set_timeslots, rule =maximumNumberOfStarts_Individual_GasHeating_EQ2_2_Rule_BT7)



        def maximumNumberOfStarts_Individual_GasHeating_EQ3_HelpAssociatedBinary_Rule_BT7 (model, i, t):
            if Run_Simulations.considerMaxiumNumberOfStartsGas_Individual ==True :
                return model.variable_HPswitchedOff_HelpAssociatedBinary_GasHeating_BT7 [i, t] >= model.variable_heatGenerationCoefficient_GasBoiler_BT7[i, t]
            if Run_Simulations.considerMaxiumNumberOfStartsGas_Individual ==False:
               return pyo.Constraint.Skip

        model.constraint_maximumNumberOfStarts_Individual_GasHeating_EQ3_HelpAssociatedBinary_BT7 = pyo.Constraint(model.set_buildings_BT7, model.set_timeslots, rule =maximumNumberOfStarts_Individual_GasHeating_EQ3_HelpAssociatedBinary_Rule_BT7)



        def maximumNumberOfStarts_Individual_Gaseating_EQ4_HelpAssociatedBinary_Rule_BT7 (model, i, t):
            if Run_Simulations.considerMaxiumNumberOfStartsGas_Individual ==True :
                return model.variable_heatGenerationCoefficient_GasBoiler_BT7[i, t] * (1/(SetUpScenarios.minimalModulationdDegree_GasBoiler/100))  >= model.variable_HPswitchedOff_HelpAssociatedBinary_GasHeating_BT7 [i, t]
            if Run_Simulations.considerMaxiumNumberOfStartsGas_Individual ==False:
               return pyo.Constraint.Skip

        model.constraint_maximumNumberOfStarts_Individual_GasHeating_EQ4_HelpAssociatedBinary_BT7 = pyo.Constraint(model.set_buildings_BT7, model.set_timeslots, rule =maximumNumberOfStarts_Individual_Gaseating_EQ4_HelpAssociatedBinary_Rule_BT7)



        def maximumNumberOfStarts_Individual_GasHeating_EQ5_NumberOfStarts_Rule_BT7 (model, i, t):
            if Run_Simulations.considerMaxiumNumberOfStartsGas_Individual ==True:
                return  sum (model.variable_HPswitchedOff_Individual_GasHeating_BT7 [i, t] for t in model.set_timeslots)<= Run_Simulations.maximumNumberOfStarts_Individual
            if Run_Simulations.considerMaxiumNumberOfStartsGas_Individual ==False:
                return pyo.Constraint.Skip

        model.constraint_maximumNumberOfStarts_Individual_GasHeating_EQ5_NumberOfStarts_BT7 = pyo.Constraint(model.set_buildings_BT7, model.set_timeslots, rule =maximumNumberOfStarts_Individual_GasHeating_EQ5_NumberOfStarts_Rule_BT7)



    ###########################################################################################################################
    
    #Combined equations for all 7 building types
    
    
    
    model.variable_RESGenerationTotal = pyo.Var(model.set_timeslots, within=pyo.NonNegativeReals)
    model.variable_PVGenerationTotal = pyo.Var(model.set_timeslots, within=pyo.NonNegativeReals)
    model.variable_electricalPowerTotal = pyo.Var(model.set_timeslots, within=pyo.NonNegativeReals)
    model.variable_surplusPowerTotal = pyo.Var(model.set_timeslots)
    model.variable_surplusPowerPositivePart = pyo.Var(model.set_timeslots, within=pyo.NonNegativeReals)
    model.variable_surplusPowerNegativePart = pyo.Var(model.set_timeslots, within=pyo.NonNegativeReals)
    model.variable_help_isSurplusPowerPositive = pyo.Var(model.set_timeslots, within=pyo.Binary)
    model.variable_costsPerTimeSlot = pyo.Var(model.set_timeslots)
    model.variable_gasConsumptionPerTimeSlot_kWh = pyo.Var(model.set_timeslots)
    model.variable_revenuePerTimeSlot = pyo.Var(model.set_timeslots)
    model.variable_thermalDiscomfortCombined = pyo.Var(model.set_timeslots)
    
    model.variable_objectiveMaximumLoad = pyo.Var(within=pyo.NonNegativeReals)
    model.variable_objectiveSurplusEnergy = pyo.Var()
    model.variable_objectiveCosts = pyo.Var()
    model.variable_objectiveThermalDiscomfort = pyo.Var()
    model.variable_objectiveGasConsumption = pyo.Var()
    
    
    #Define help parameters for the printed files
    def init_param_helpTimeSlots_BT1 (model, i,j):
        return j
    
    model.param_helpTimeSlots_BT1 = pyo.Param(model.set_buildings_BT1, model.set_timeslots, mutable = True, initialize=init_param_helpTimeSlots_BT1)
    
    def init_param_helpTimeSlots_BT2 (model, i,j):
        return j
    
    model.param_helpTimeSlots_BT2 = pyo.Param(model.set_buildings_BT2, model.set_timeslots, mutable = True, initialize=init_param_helpTimeSlots_BT2)
    
    def init_param_helpTimeSlots_BT3 (model, i,j):
        return j
    
    model.param_helpTimeSlots_BT3 = pyo.Param(model.set_buildings_BT3, model.set_timeslots, mutable = True, initialize=init_param_helpTimeSlots_BT3)

    def init_param_helpTimeSlots_BT4 (model, i,j):
        return j
    
    model.param_helpTimeSlots_BT4 = pyo.Param(model.set_buildings_BT4, model.set_timeslots, mutable = True, initialize=init_param_helpTimeSlots_BT4)

    def init_param_helpTimeSlots_BT5 (model, i,j):
        return j
    
    model.param_helpTimeSlots_BT5 = pyo.Param(model.set_buildings_BT5, model.set_timeslots, mutable = True, initialize=init_param_helpTimeSlots_BT5)

    def init_param_helpTimeSlots_BT6 (model, i,j):
        return j

    model.param_helpTimeSlots_BT6 = pyo.Param(model.set_buildings_BT6, model.set_timeslots, mutable = True, initialize=init_param_helpTimeSlots_BT6)

    def init_param_helpTimeSlots_BT7 (model, i,j):
        return j

    model.param_helpTimeSlots_BT7 = pyo.Param(model.set_buildings_BT7, model.set_timeslots, mutable = True, initialize=init_param_helpTimeSlots_BT7)



    #Initializer functions for the Big-M parameters
    
    def BigM_Surplus_PositiveRule_Init (model, t):
        return  sum (model.param_pvGenerationNominal_BT1 [setIndex_BT1, t] * SetUpScenarios.determinePVPeakOfBuildings(setIndex_BT1 - 1) + model.param_windAssignedNominal_BT1 [setIndex_BT1, t] * SetUpScenarios.maximalPowerOfWindTurbine for setIndex_BT1 in model.set_buildings_BT1) + sum (model.param_pvGenerationNominal_BT2 [setIndex_BT2, t] * SetUpScenarios.determinePVPeakOfBuildings(SetUpScenarios.numberOfBuildings_BT1 + setIndex_BT2 - 1) + model.param_windAssignedNominal_BT2 [setIndex_BT2, t] * SetUpScenarios.maximalPowerOfWindTurbine for setIndex_BT2 in model.set_buildings_BT2) + sum (model.param_pvGenerationNominal_BT3 [setIndex_BT3, t] * SetUpScenarios.determinePVPeakOfBuildings(SetUpScenarios.numberOfBuildings_BT1 + SetUpScenarios.numberOfBuildings_BT2 + setIndex_BT3 - 1) + model.param_windAssignedNominal_BT3 [setIndex_BT3, t] * SetUpScenarios.maximalPowerOfWindTurbine for setIndex_BT3 in model.set_buildings_BT3) + sum (model.param_pvGenerationNominal_BT4 [setIndex_BT4, t] * SetUpScenarios.determinePVPeakOfBuildings(SetUpScenarios.numberOfBuildings_BT1 + SetUpScenarios.numberOfBuildings_BT2 + SetUpScenarios.numberOfBuildings_BT3 + setIndex_BT4 - 1) + model.param_windAssignedNominal_BT4 [setIndex_BT4, t] * SetUpScenarios.maximalPowerOfWindTurbine for setIndex_BT4 in model.set_buildings_BT4) +  sum (model.param_pvGenerationNominal_BT5 [setIndex_BT5, t] * SetUpScenarios.determinePVPeakOfBuildings(SetUpScenarios.numberOfBuildings_BT1 + SetUpScenarios.numberOfBuildings_BT2 + SetUpScenarios.numberOfBuildings_BT3 + SetUpScenarios.numberOfBuildings_BT4 + setIndex_BT5 - 1) + model.param_windAssignedNominal_BT5 [setIndex_BT5, t] * SetUpScenarios.maximalPowerOfWindTurbine for setIndex_BT5 in model.set_buildings_BT5)  +  sum (model.param_pvGenerationNominal_BT6 [setIndex_BT6, t] * SetUpScenarios.determinePVPeakOfBuildings(SetUpScenarios.numberOfBuildings_BT1 + SetUpScenarios.numberOfBuildings_BT2 + SetUpScenarios.numberOfBuildings_BT3 + SetUpScenarios.numberOfBuildings_BT4 + SetUpScenarios.numberOfBuildings_BT5 + setIndex_BT6 - 1) + model.param_windAssignedNominal_BT6 [setIndex_BT6, t] * SetUpScenarios.maximalPowerOfWindTurbine for setIndex_BT6 in model.set_buildings_BT6)  +  sum (model.param_pvGenerationNominal_BT7 [setIndex_BT7, t] * SetUpScenarios.determinePVPeakOfBuildings(SetUpScenarios.numberOfBuildings_BT1 + SetUpScenarios.numberOfBuildings_BT2 + SetUpScenarios.numberOfBuildings_BT3 + SetUpScenarios.numberOfBuildings_BT4 + SetUpScenarios.numberOfBuildings_BT5 + SetUpScenarios.numberOfBuildings_BT6 + setIndex_BT7 - 1) + model.param_windAssignedNominal_BT7 [setIndex_BT7, t] * SetUpScenarios.maximalPowerOfWindTurbine for setIndex_BT7 in model.set_buildings_BT7) + 1000
    
    
    def BigM_Surplus_NegativeRule_Init (model, t):
        return  sum (SetUpScenarios.electricalPower_HP  +  SetUpScenarios.chargingPowerMaximal_EV + model.param_electricalDemand_In_W_BT1 [setIndex_BT1, t] for setIndex_BT1 in model.set_buildings_BT1) + sum (SetUpScenarios.electricalPower_HP  +   model.param_electricalDemand_In_W_BT2 [setIndex_BT2, t] for setIndex_BT2 in model.set_buildings_BT2) + sum ( SetUpScenarios.chargingPowerMaximal_EV + model.param_electricalDemand_In_W_BT3 [setIndex_BT3, t] for setIndex_BT3 in model.set_buildings_BT3)  + sum (SetUpScenarios.electricalPower_HP_BT4_MFH  + model.param_electricalDemand_In_W_BT4 [setIndex_BT4, t] for setIndex_BT4 in model.set_buildings_BT4) + sum ( SetUpScenarios.chargingPowerMaximal_BAT + model.param_electricalDemand_In_W_BT5 [setIndex_BT5, t] for setIndex_BT5 in model.set_buildings_BT5) + sum (SetUpScenarios.maximalPowerElectricalHeatingElement  + model.param_electricalDemand_In_W_BT6 [setIndex_BT6, t] for setIndex_BT6 in model.set_buildings_BT6) + sum (SetUpScenarios.electricalPowerFanHeater_Stage3  + model.param_electricalDemand_In_W_BT7 [setIndex_BT7, t] for setIndex_BT7 in model.set_buildings_BT7) + 1000

    
    model.param_BigM_Surplus_Positive = pyo.Param(model.set_timeslots, mutable=True, initialize =BigM_Surplus_PositiveRule_Init)
    model.param_BigM_Surplus_Negative = pyo.Param(model.set_timeslots, mutable=True, initialize =BigM_Surplus_NegativeRule_Init)


    
    #Equations for calculating the total generation from renewable energy sources (RES): 
    def RESgenerationTotalRule (model, t):
        return model.variable_RESGenerationTotal [t] == sum (model.param_pvGenerationNominal_BT1 [setIndex_BT1, t] * SetUpScenarios.determinePVPeakOfBuildings(setIndex_BT1 - 1) + model.param_windAssignedNominal_BT1 [setIndex_BT1, t] * SetUpScenarios.maximalPowerOfWindTurbine for setIndex_BT1 in model.set_buildings_BT1) + sum (model.param_pvGenerationNominal_BT2 [setIndex_BT2, t] * SetUpScenarios.determinePVPeakOfBuildings(SetUpScenarios.numberOfBuildings_BT1 + setIndex_BT2 - 1) + model.param_windAssignedNominal_BT2 [setIndex_BT2, t] * SetUpScenarios.maximalPowerOfWindTurbine for setIndex_BT2 in model.set_buildings_BT2) + sum (model.param_pvGenerationNominal_BT3 [setIndex_BT3, t] * SetUpScenarios.determinePVPeakOfBuildings(SetUpScenarios.numberOfBuildings_BT1 + SetUpScenarios.numberOfBuildings_BT2 + setIndex_BT3 - 1) + model.param_windAssignedNominal_BT3 [setIndex_BT3, t] * SetUpScenarios.maximalPowerOfWindTurbine for setIndex_BT3 in model.set_buildings_BT3) + sum (model.param_pvGenerationNominal_BT4 [setIndex_BT4, t] * SetUpScenarios.determinePVPeakOfBuildings(SetUpScenarios.numberOfBuildings_BT1 + SetUpScenarios.numberOfBuildings_BT2  + SetUpScenarios.numberOfBuildings_BT3  + setIndex_BT4 - 1) + model.param_windAssignedNominal_BT4 [setIndex_BT4, t] * SetUpScenarios.maximalPowerOfWindTurbine for setIndex_BT4 in model.set_buildings_BT4) + sum (model.param_pvGenerationNominal_BT5 [setIndex_BT5, t] * SetUpScenarios.determinePVPeakOfBuildings(SetUpScenarios.numberOfBuildings_BT1 + SetUpScenarios.numberOfBuildings_BT2  + SetUpScenarios.numberOfBuildings_BT3 + SetUpScenarios.numberOfBuildings_BT4  + setIndex_BT5 - 1) + model.param_windAssignedNominal_BT5 [setIndex_BT5, t] * SetUpScenarios.maximalPowerOfWindTurbine for setIndex_BT5 in model.set_buildings_BT5) +  sum (model.param_pvGenerationNominal_BT6 [setIndex_BT6, t] * SetUpScenarios.determinePVPeakOfBuildings(SetUpScenarios.numberOfBuildings_BT1 + SetUpScenarios.numberOfBuildings_BT2  + SetUpScenarios.numberOfBuildings_BT3 + SetUpScenarios.numberOfBuildings_BT4 + SetUpScenarios.numberOfBuildings_BT5  + setIndex_BT6 - 1) + model.param_windAssignedNominal_BT6 [setIndex_BT6, t] * SetUpScenarios.maximalPowerOfWindTurbine for setIndex_BT6 in model.set_buildings_BT6) +  sum (model.param_pvGenerationNominal_BT7 [setIndex_BT7, t] * SetUpScenarios.determinePVPeakOfBuildings(SetUpScenarios.numberOfBuildings_BT1 + SetUpScenarios.numberOfBuildings_BT2  + SetUpScenarios.numberOfBuildings_BT3 + SetUpScenarios.numberOfBuildings_BT4 + SetUpScenarios.numberOfBuildings_BT5  + SetUpScenarios.numberOfBuildings_BT6  + setIndex_BT7 - 1) + model.param_windAssignedNominal_BT7 [setIndex_BT7, t] * SetUpScenarios.maximalPowerOfWindTurbine for setIndex_BT7 in model.set_buildings_BT7)
        
    model.constraint_RESgenerationTotal = pyo.Constraint(model.set_timeslots, rule = RESgenerationTotalRule)


    #Equations for calculating the total thermal discomfort
    def totalDiscomfortRule (model, t):
        return model.variable_thermalDiscomfortCombined [t] == sum (model.variable_deviation_temperature_positive_BT1 [setIndex_BT1, t] + model.variable_deviation_temperature_negative_BT1[setIndex_BT1, t] for setIndex_BT1 in model.set_buildings_BT1) +  sum (model.variable_deviation_temperature_positive_BT2 [setIndex_BT2, t] + model.variable_deviation_temperature_negative_BT2[setIndex_BT2, t] for setIndex_BT2 in model.set_buildings_BT2) + sum (model.variable_deviation_temperature_positive_BT4 [setIndex_BT4, t] + model.variable_deviation_temperature_negative_BT4[setIndex_BT4, t] for setIndex_BT4 in model.set_buildings_BT4) + sum (model.variable_deviation_temperature_positive_BT6 [setIndex_BT6, t] + model.variable_deviation_temperature_negative_BT6[setIndex_BT6, t] for setIndex_BT6 in model.set_buildings_BT6) +  sum (model.variable_deviation_temperature_positive_BT7 [setIndex_BT7, t] + model.variable_deviation_temperature_negative_BT7[setIndex_BT7, t] for setIndex_BT7 in model.set_buildings_BT7)

    model.constraint_totalDiscomfort = pyo.Constraint(model.set_timeslots, rule = totalDiscomfortRule)


    
    #Equations for calculating the total generation from PV
    def PVgenerationTotalRule (model, t):
        return model.variable_PVGenerationTotal [t] == sum (model.param_pvGenerationNominal_BT1 [setIndex_BT1, t] * SetUpScenarios.determinePVPeakOfBuildings(setIndex_BT1 - 1)  for setIndex_BT1 in model.set_buildings_BT1) + sum (model.param_pvGenerationNominal_BT2 [setIndex_BT2, t] * SetUpScenarios.determinePVPeakOfBuildings(SetUpScenarios.numberOfBuildings_BT1 + setIndex_BT2 - 1) for setIndex_BT2 in model.set_buildings_BT2) + sum (model.param_pvGenerationNominal_BT3 [setIndex_BT3, t] * SetUpScenarios.determinePVPeakOfBuildings(SetUpScenarios.numberOfBuildings_BT1 + SetUpScenarios.numberOfBuildings_BT2 + setIndex_BT3 - 1) for setIndex_BT3 in model.set_buildings_BT3)  + sum (model.param_pvGenerationNominal_BT4 [setIndex_BT4, t] * SetUpScenarios.determinePVPeakOfBuildings(SetUpScenarios.numberOfBuildings_BT1 + SetUpScenarios.numberOfBuildings_BT2 + SetUpScenarios.numberOfBuildings_BT3 + setIndex_BT4 - 1) for setIndex_BT4 in model.set_buildings_BT4) + sum (model.param_pvGenerationNominal_BT5 [setIndex_BT5, t] * SetUpScenarios.determinePVPeakOfBuildings(SetUpScenarios.numberOfBuildings_BT1 + SetUpScenarios.numberOfBuildings_BT2 + SetUpScenarios.numberOfBuildings_BT3 + SetUpScenarios.numberOfBuildings_BT4 + setIndex_BT5 - 1) for setIndex_BT5 in model.set_buildings_BT5) +  sum (model.param_pvGenerationNominal_BT6 [setIndex_BT6, t] * SetUpScenarios.determinePVPeakOfBuildings(SetUpScenarios.numberOfBuildings_BT1 + SetUpScenarios.numberOfBuildings_BT2 + SetUpScenarios.numberOfBuildings_BT3 + SetUpScenarios.numberOfBuildings_BT4 + SetUpScenarios.numberOfBuildings_BT5 + setIndex_BT6 - 1) for setIndex_BT6 in model.set_buildings_BT6) + sum (model.param_pvGenerationNominal_BT7 [setIndex_BT7, t] * SetUpScenarios.determinePVPeakOfBuildings(SetUpScenarios.numberOfBuildings_BT1 + SetUpScenarios.numberOfBuildings_BT2 + SetUpScenarios.numberOfBuildings_BT3 + SetUpScenarios.numberOfBuildings_BT4 + SetUpScenarios.numberOfBuildings_BT5 + SetUpScenarios.numberOfBuildings_BT6 + setIndex_BT7 - 1) for setIndex_BT7 in model.set_buildings_BT7)
        
    model.constraint_PVgenerationTotal = pyo.Constraint(model.set_timeslots, rule = PVgenerationTotalRule)
    
    
    

    #Equation for calculating the total electrical power
    def electricalPowerTotalRule (model, t):
        return model.variable_electricalPowerTotal [t] == sum (  model.variable_electricalPowerTotal_BT1 [setIndex_BT1, t] for setIndex_BT1 in model.set_buildings_BT1) + sum (  model.variable_electricalPowerTotal_BT2 [setIndex_BT2, t] for setIndex_BT2 in model.set_buildings_BT2) + sum (  model.variable_electricalPowerTotal_BT3 [setIndex_BT3, t] for setIndex_BT3 in model.set_buildings_BT3) + sum (  model.variable_electricalPowerTotal_BT4 [setIndex_BT4, t] for setIndex_BT4 in model.set_buildings_BT4) + sum (  model.variable_electricalPowerTotal_BT5 [setIndex_BT5, t] for setIndex_BT5 in model.set_buildings_BT5) + sum ( model.variable_electricalPowerTotal_BT6 [setIndex_BT6, t] for setIndex_BT6 in model.set_buildings_BT6) + sum (  model.variable_electricalPowerTotal_BT7 [setIndex_BT7, t] for setIndex_BT7 in model.set_buildings_BT7)
    
    model.constraint_electricalPowerTotal = pyo.Constraint(model.set_timeslots, rule = electricalPowerTotalRule)
    
    
    
    
    #Equations for the surplus power
    def surplusPowerTotalRule (model, t):
        return model.variable_surplusPowerTotal [t] == model.variable_RESGenerationTotal [t] - model.variable_electricalPowerTotal [t]
    
    model.constraint_surplusPowerTotal = pyo.Constraint(model.set_timeslots, rule = surplusPowerTotalRule)
    
    
    
    # Divide surplus energy into a positive and a negative part using the big-M approach
    def surplusPowerPartsRule (model, t):
        return model.variable_surplusPowerTotal [t] == model.variable_surplusPowerPositivePart [t] - model.variable_surplusPowerNegativePart [t]
    
    model.constraint_surplusPowerParts = pyo.Constraint(model.set_timeslots, rule = surplusPowerPartsRule)
    
    
    
    #Use big-M approach to restric the surplus power parts
    
    def surplusPowerPositiveRule (model, t):
        return model.variable_surplusPowerPositivePart [t] <= model.variable_help_isSurplusPowerPositive [t] * model.param_BigM_Surplus_Positive [t]
    
    model.constraint_surplusPowerPositive = pyo.Constraint(model.set_timeslots, rule = surplusPowerPositiveRule)
    
    def surplusPowerNegativeRule (model, t):
        return model.variable_surplusPowerNegativePart [t] <= (1 - model.variable_help_isSurplusPowerPositive [t]) * model.param_BigM_Surplus_Negative [t]
    
    model.constraint_surplusPowerNegative = pyo.Constraint(model.set_timeslots, rule = surplusPowerNegativeRule)
    
    
    
    #Equation for calculating the costs per timeslot
    def costsPerTimeSlotRule (model, t):
            return model.variable_costsPerTimeSlot [t] == (1 - model.variable_help_isSurplusPowerPositive [t]) * (model.variable_electricalPowerTotal [t] - model.variable_PVGenerationTotal [t]) * SetUpScenarios.timeResolution_InMinutes * 60 * (model.param_electricityPrice_In_Cents[t]/3600000) + model.variable_gasConsumptionPerTimeSlot_kWh [t] * SetUpScenarios.priceForGasInCentPerKWH
    
    model.constraint_costsPerTimeSlots = pyo.Constraint(model.set_timeslots, rule =costsPerTimeSlotRule )
    
    

    def revenuePerTimeSlotRule (model, t):
        return model.variable_revenuePerTimeSlot [t] == model.variable_help_isSurplusPowerPositive [t] * (model.variable_PVGenerationTotal [t] - model.variable_electricalPowerTotal [t]) * SetUpScenarios.timeResolution_InMinutes * 60 * (SetUpScenarios.revenueForFeedingBackElecticityIntoTheGrid_CentsPerkWh/3600000)
    
    model.constraint_revenuePerTimeSlots = pyo.Constraint(model.set_timeslots, rule = revenuePerTimeSlotRule)


    def gasConsumptionPerTimeSlotRule (model, t):
        return model.variable_gasConsumptionPerTimeSlot_kWh [t] == sum (model.variable_heatGenerationCoefficient_GasBoiler_BT6[setIndex_BT6, t] * SetUpScenarios.maximalPowerGasBoiler * SetUpScenarios.timeResolution_InMinutes * 60 for setIndex_BT6 in model.set_buildings_BT6)/3600000 + sum (model.variable_heatGenerationCoefficient_GasBoiler_BT7[setIndex_BT7, t] * SetUpScenarios.maximalPowerGasBoiler * SetUpScenarios.timeResolution_InMinutes * 60 for setIndex_BT7 in model.set_buildings_BT7)/3600000

    model.constraint_gasConsumptionPerTimeSlot = pyo.Constraint(model.set_timeslots, rule = gasConsumptionPerTimeSlotRule)



    
    #Objectives
    
    #Equation for the objective: Minimize surplus power
    def objective_minimizeSurplusPowerRuel (model):
        return model.variable_objectiveSurplusEnergy == sum(model.variable_surplusPowerPositivePart[t] for t in model.set_timeslots)
    
    model.constraint_objective_minimizeSurplusPower = pyo.Constraint( rule = objective_minimizeSurplusPowerRuel)

    #Equation for the objective: Minimize gas consumption
    def objectiveGasConsumption (model):
        return model.variable_objectiveGasConsumption == sum (model.variable_gasConsumptionPerTimeSlot_kWh[t] for t in model.set_timeslots)

    model.constraints_objective_minimizeGasConsumption = pyo.Constraint(rule= objectiveGasConsumption)
    
    #Equation for the objective: Minimize costs
    def objective_minimizeCostsRule (model):
        return model.variable_objectiveCosts == sum ((model.variable_costsPerTimeSlot [t] - model.variable_revenuePerTimeSlot [t]) for t in model.set_timeslots)
    
    model.constraints_objective_minimizeCosts = pyo.Constraint(rule = objective_minimizeCostsRule)
    
    
    #Equations for calculating the maxium load. The absolute function is linearized by using 2 greater or equal constraints
    
    def objective_maximumLoadRule_1 (model, t):
        return model.variable_objectiveMaximumLoad >= model.variable_electricalPowerTotal [t] - model.variable_PVGenerationTotal [t]
    
    model.constraints_objective_maxiumLoad_1 = pyo.Constraint(model.set_timeslots, rule = objective_maximumLoadRule_1)
    
    def objective_maximumLoadRule_2 (model, t):
        return model.variable_objectiveMaximumLoad >= model.variable_PVGenerationTotal [t] - model.variable_electricalPowerTotal [t] 
    
    model.constraints_objective_maxiumLoad_2 = pyo.Constraint(model.set_timeslots, rule = objective_maximumLoadRule_2)


    def objective_minimizeThermalDiscomfortRule (model):
        return model.variable_objectiveThermalDiscomfort == sum (model.variable_thermalDiscomfortCombined[t] for t in model.set_timeslots)

    model.constraint_objective_minimizeThermalDiscomfort = pyo.Constraint ( rule = objective_minimizeThermalDiscomfortRule)


    #Define additional constraints for the box-epsilon-method
    def epsilon_objective_minimizeCostsRule (model):
        if optParameters ["epsilon_objective_minimizeCosts_Active"] == True:
            return model.variable_objectiveCosts <= optParameters["epsilon_objective_minimizeCosts_TargetValue"]
        else:
            return pyo.Constraint.Skip

    model.constraint_epsilon_objective_minimizeCosts = pyo.Constraint(rule = epsilon_objective_minimizeCostsRule)


    def epsilon_objective_minimizeMaximumLoadRule(model):
        if optParameters["epsilon_objective_minimizePeakLoad_Active"] == True:
            return model.variable_objectiveMaximumLoad <= optParameters["epsilon_objective_minimizeMaximumLoad_TargetValue"]
        else:
            return pyo.Constraint.Skip

    model.constraint_epsilon_objective_minimizeMaximumLoad = pyo.Constraint(rule = epsilon_objective_minimizeMaximumLoadRule)


    def epsilon_objective_minimizeGasConsumptionRule (model):
        if optParameters["epsilon_objective_minimizeGasConsumption_Active"] == True:
            return model.variable_objectiveGasConsumption <= optParameters["epsilon_objective_minimizeGasConsumption_TargetValue"]
        else:
            return pyo.Constraint.Skip

    model.constraint_epsilon_objective_minimizeGasConsumption = pyo.Constraint(rule = epsilon_objective_minimizeGasConsumptionRule)


    def epsilon_objective_minimizeThermalDiscomfortsRule (model):
        if optParameters["epsilon_objective_minimizeThermalDiscomfort_Active"] == True:
            return model.variable_objectiveThermalDiscomfort <= optParameters["epsilon_objective_minimizeThermalDiscomfort_TargetValue"]
        else:
            return pyo.Constraint.Skip

    model.constraint_epsilon_objective_minimizeThermalDiscomfort = pyo.Constraint(rule = epsilon_objective_minimizeThermalDiscomfortsRule)















    #Define combined objective function for the optimization depending on the objectives specified in the file Run_Simulations
    
    def objectiveRule_combined_general (model):
        if optParameters['optimization_1Objective'] == True:
            
            
            if optParameters['optimizationGoal_minimizeSurplusEnergy'] == True:
               return  (model.variable_objectiveSurplusEnergy * SetUpScenarios.timeResolution_InMinutes * 60) /3600
    
          
            if optParameters['optimizationGoal_minimizePeakLoad'] == True:
                return model.variable_objectiveMaximumLoad
           
            if optParameters['optimizationGoal_minimizeCosts'] == True:
                return  model.variable_objectiveCosts

            if optParameters['optimizationGoal_minimizeThermalDiscomfort'] ==True:
                return model.variable_objectiveThermalDiscomfort

            if optParameters['optimizationGoal_minimizeGas'] == True:
                return model.variable_objectiveGasConsumption
            
                
        if optParameters['optimization_2Objective'] == True:
    
            if (optParameters['optimizationGoal_minimizeSurplusEnergy'] == True and optParameters['optimizationGoal_minimizePeakLoad'] == True):
                return optParameters['objective_minimizePeakLoad_weight'] * ((model.variable_objectiveMaximumLoad)/optParameters['objective_minimizePeakLoad_normalizationValue']) + optParameters['objective_minimizeSurplusEnergy_weight'] * (((model.variable_objectiveSurplusEnergy * SetUpScenarios.timeResolution_InMinutes * 60) /3600) /optParameters['objective_minimizeSurplusEnergy_normalizationValue'])
            
            if (optParameters['optimizationGoal_minimizeSurplusEnergy'] == True and optParameters['optimizationGoal_minimizeCosts'] == True):
                return optParameters['objective_minimizeCosts_weight'] * ((model.variable_objectiveCosts) /optParameters['objective_minimizeCosts_normalizationValue']) + optParameters['objective_minimizeSurplusEnergy_weight'] * (((model.variable_objectiveSurplusEnergy * SetUpScenarios.timeResolution_InMinutes * 60) /3600) /optParameters['objective_minimizeSurplusEnergy_normalizationValue'])
            
            if (optParameters['optimizationGoal_minimizePeakLoad'] == True and optParameters['optimizationGoal_minimizeCosts'] == True):
                return (optParameters['objective_minimizePeakLoad_weight'] * (model.variable_objectiveMaximumLoad /optParameters['objective_minimizePeakLoad_normalizationValue']) + optParameters['objective_minimizeCosts_weight'] * (model.variable_objectiveCosts /optParameters['objective_minimizeCosts_normalizationValue']))*100

            if (optParameters['optimizationGoal_minimizeCosts'] == True and optParameters['optimizationGoal_minimizeThermalDiscomfort'] == True):
                return optParameters['objective_minimizeThermalDiscomfort_weight'] * ((model.variable_objectiveThermalDiscomfort) /optParameters['objective_minimizeThermalDiscomfort_normalizationValue']) + optParameters['objective_minimizeCosts_weight'] * ((model.variable_objectiveCosts) /optParameters['objective_minimizeCosts_normalizationValue'])

        if optParameters['optimization_3Objectives'] == True:
            return optParameters['objective_minimizePeakLoad_weight'] * ((model.variable_objectiveMaximumLoad) /optParameters['objective_minimizePeakLoad_normalizationValue']) + optParameters['objective_minimizeCosts_weight'] * ((model.variable_objectiveCosts) /optParameters['objective_minimizeSurplusEnergy_normalizationValue']) + optParameters['objective_minimizeSurplusEnergy_weight'] * (((model.variable_objectiveSurplusEnergy * SetUpScenarios.timeResolution_InMinutes * 60) /3600) /optParameters['objective_minimizeSurplusEnergy_normalizationValue'])
              
    
    model.objective_combined_general = pyo.Objective( rule=objectiveRule_combined_general, sense =pyo.minimize)

    

    #Solve the model
    print("Start of solving")
    solver = pyo.SolverFactory('gurobi')
    solver.options['MIPGap'] = SetUpScenarios.solverOption_relativeGap_Central
    solver.options['TimeLimit'] = SetUpScenarios.solverOption_timeLimit_Central
    solution = solver.solve(model, tee=True)
    model.variable_objectiveThermalDiscomfort.pprint()
    
    #Help function for priting infeasible constraints if the model can't be solved
    #log_infeasible_constraints(model)
    mipGapPercentOfFoundSolution = -1
    timeForFindingOptimalSolution = -1

    #Check if the problem is solved or infeasible
    if (solution.solver.status == SolverStatus.ok  and solution.solver.termination_condition == TerminationCondition.optimal) or  solution.solver.termination_condition == TerminationCondition.maxTimeLimit:
        print("Result Status: Optimal")
        mipGapPercentOfFoundSolution = round((abs(solution.problem.upper_bound - solution.problem.lower_bound) / solution.problem.upper_bound)*100, 2)
        timeForFindingOptimalSolution = round(solution.solver.time,2)

        if SetUpScenarios.numberOfBuildings_BT1 >=1:
    
            #Create pandas dataframe for displaying the results of BT1
            outputVariables_list_BT1 = [model.param_helpTimeSlots_BT1, model.variable_heatGenerationCoefficient_SpaceHeating_BT1, model.variable_heatGenerationCoefficient_DHW_BT1, model.variable_help_OnlyOneStorage_BT1, model.variable_temperatureBufferStorage_BT1, model.variable_usableVolumeDHWTank_BT1,  model.variable_electricalPowerTotal_BT1, model.variable_pvGeneration_BT1, model.variable_windPowerAssigned_BT1,  model.variable_currentChargingPowerEV_BT1, model.variable_energyLevelEV_BT1, model.variable_SOC_EV_BT1, model.param_heatDemand_In_W_BT1, model.param_DHWDemand_In_W_BT1, model.param_electricalDemand_In_W_BT1, model.param_pvGenerationNominal_BT1, model.param_outSideTemperature_In_C, model.param_windAssignedNominal_BT1, model.param_availabilityPerTimeSlotOfEV_BT1, model.param_energyConsumptionEV_Joule_BT1, model.param_COPHeatPump_SpaceHeating_BT1, model.param_COPHeatPump_DHW_BT1, model.param_electricityPrice_In_Cents , model.set_timeslots]
            optimal_values_list_BT1 = [[pyo.value(model_item[key]) for key in model_item] for model_item in outputVariables_list_BT1] 
            results_BT1 = pd.DataFrame(optimal_values_list_BT1)
            results_BT1= results_BT1.T
            results_BT1 = results_BT1.rename(columns = {0:'timeslot', 1:'variable_heatGenerationCoefficient_SpaceHeating', 2:'variable_heatGenerationCoefficient_DHW', 3:'variable_help_OnlyOneStorage', 4:'variable_temperatureBufferStorage', 5:'variable_usableVolumeDHWTank',  6:'variable_electricalPowerTotal', 7:'variable_PVGeneration', 8:'variable_windPowerAssigned',  9:'variable_currentChargingPowerEV', 10:'variable_energyLevelEV_kWh', 11:'variable_SOC_EV', 12:'param_heatDemand_In_W', 13:'param_DHWDemand_In_W', 14:'param_electricalDemand_In_W', 15:'param_pvGenerationNominal', 16:'param_outSideTemperature_In_C', 17:'param_windAssignedNominal', 18:'param_availabilityPerTimeSlotOfEV', 19:'param_energyConsumptionEV', 20:'param_COPHeatPump_SpaceHeating', 21:'param_COPHeatPump_DHW', 22:'param_PriceElectricity [Cents]', 23:'set_timeslots'})
            cols = ['set_timeslots']
            results_BT1.set_index('set_timeslots', inplace=True)
            #Round values
            results_BT1['variable_temperatureBufferStorage'] = results_BT1['variable_temperatureBufferStorage'].round(2)
            results_BT1['variable_usableVolumeDHWTank'] = results_BT1['variable_usableVolumeDHWTank'].round(1)
            results_BT1['param_COPHeatPump_SpaceHeating'] = results_BT1['param_COPHeatPump_SpaceHeating'].round(3)
            results_BT1['param_COPHeatPump_DHW'] = results_BT1['param_COPHeatPump_DHW'].round(3)
            results_BT1['variable_SOC_EV'] = results_BT1['variable_SOC_EV'].round(2)
            results_BT1['variable_energyLevelEV_kWh'] = results_BT1['variable_energyLevelEV_kWh']/3600000
            results_BT1['variable_energyLevelEV_kWh'] = results_BT1['variable_energyLevelEV_kWh'].round(2)
            results_BT1['variable_heatGenerationCoefficient_SpaceHeating'] = results_BT1['variable_heatGenerationCoefficient_SpaceHeating'].round(4)
            results_BT1['variable_heatGenerationCoefficient_DHW'] = results_BT1['variable_heatGenerationCoefficient_DHW'].round(4)
            filePath_BT1 = folderPath + "/Combined_BT1.csv"
            results_BT1.to_csv(filePath_BT1, index=False,  sep =";") 

            #Create output vector in the correct format
            outputVector_heatGenerationCoefficientSpaceHeating_BT1 = results_BT1['variable_heatGenerationCoefficient_SpaceHeating'].to_numpy().reshape((SetUpScenarios.numberOfTimeSlotsPerDay,SetUpScenarios.numberOfBuildings_BT1), order='F')
            outputVector_heatGenerationCoefficientDHW_BT1 = results_BT1['variable_heatGenerationCoefficient_DHW'].to_numpy().reshape((SetUpScenarios.numberOfTimeSlotsPerDay,SetUpScenarios.numberOfBuildings_BT1), order='F')
            outputVector_chargingPowerEV_BT1 = results_BT1['variable_currentChargingPowerEV'].to_numpy().reshape((SetUpScenarios.numberOfTimeSlotsPerDay,SetUpScenarios.numberOfBuildings_BT1), order='F')
            outputVector_heatGenerationCoefficientSpaceHeating_BT1 = outputVector_heatGenerationCoefficientSpaceHeating_BT1.transpose()
            outputVector_heatGenerationCoefficientDHW_BT1 = outputVector_heatGenerationCoefficientDHW_BT1.transpose()
            outputVector_chargingPowerEV_BT1 = outputVector_chargingPowerEV_BT1.transpose()
            

        if SetUpScenarios.numberOfBuildings_BT2 >=1:        
            #Create pandas dataframe for displaying the results of BT2
            outputVariables_list_BT2 = [model.param_helpTimeSlots_BT2, model.variable_heatGenerationCoefficient_SpaceHeating_BT2, model.variable_heatGenerationCoefficient_DHW_BT2, model.variable_help_OnlyOneStorage_BT2, model.variable_temperatureBufferStorage_BT2, model.variable_usableVolumeDHWTank_BT2,  model.variable_electricalPowerTotal_BT2, model.variable_pvGeneration_BT2, model.variable_windPowerAssigned_BT2,  model.param_heatDemand_In_W_BT2, model.param_DHWDemand_In_W_BT2, model.param_electricalDemand_In_W_BT2, model.param_pvGenerationNominal_BT2, model.param_outSideTemperature_In_C, model.param_windAssignedNominal_BT2,  model.param_COPHeatPump_SpaceHeating_BT2, model.param_COPHeatPump_DHW_BT2, model.param_electricityPrice_In_Cents, model.set_timeslots]
            optimal_values_list_BT2 = [[pyo.value(model_item[key]) for key in model_item] for model_item in outputVariables_list_BT2]

            results_BT2 = pd.DataFrame(optimal_values_list_BT2)
            results_BT2= results_BT2.T
            results_BT2 = results_BT2.rename(columns = {0:'timeslot', 1:'variable_heatGenerationCoefficient_SpaceHeating', 2:'variable_heatGenerationCoefficient_DHW', 3:'variable_help_OnlyOneStorage', 4:'variable_temperatureBufferStorage', 5:'variable_usableVolumeDHWTank',    6:'variable_electricalPowerTotal',  7:'variable_pvGeneration', 8:'variable_windPowerAssigned', 9:'param_heatDemand_In_W', 10:'param_DHWDemand_In_W', 11:'param_electricalDemand_In_W', 12:'param_pvGenerationNominal', 13:'param_outSideTemperature_In_C', 14:'param_windAssignedNominal', 15:'param_COPHeatPump_SpaceHeating', 16:'param_COPHeatPump_DHW',  17:'param_PriceElectricity [Cents]', 18:'set_timeslots'})
            cols = ['set_timeslots']
            results_BT2.set_index('set_timeslots', inplace=True)
            results_BT2['variable_temperatureBufferStorage'] = results_BT2['variable_temperatureBufferStorage'].round(2)
            results_BT2['variable_usableVolumeDHWTank'] = results_BT2['variable_usableVolumeDHWTank'].round(1)
            results_BT2['param_COPHeatPump_SpaceHeating'] = results_BT2['param_COPHeatPump_SpaceHeating'].round(3)
            results_BT2['param_COPHeatPump_DHW'] = results_BT2['param_COPHeatPump_DHW'].round(3)
            results_BT2['variable_heatGenerationCoefficient_SpaceHeating'] = results_BT2['variable_heatGenerationCoefficient_SpaceHeating'].round(4)
            results_BT2['variable_heatGenerationCoefficient_DHW'] = results_BT2['variable_heatGenerationCoefficient_DHW'].round(4)
            filePath_BT2 = folderPath + "/Combined_BT2.csv"
            results_BT2.to_csv(filePath_BT2, index=False,  sep =";") 
            
            #Create output vector in the correct format
            outputVector_heatGenerationCoefficientSpaceHeating_BT2 = results_BT2['variable_heatGenerationCoefficient_SpaceHeating'].to_numpy().reshape((SetUpScenarios.numberOfTimeSlotsPerDay,SetUpScenarios.numberOfBuildings_BT2), order='F')
            outputVector_heatGenerationCoefficientDHW_BT2 = results_BT2['variable_heatGenerationCoefficient_DHW'].to_numpy().reshape((SetUpScenarios.numberOfTimeSlotsPerDay,SetUpScenarios.numberOfBuildings_BT2), order='F')
            outputVector_heatGenerationCoefficientSpaceHeating_BT2 = outputVector_heatGenerationCoefficientSpaceHeating_BT2.transpose()
            outputVector_heatGenerationCoefficientDHW_BT2 = outputVector_heatGenerationCoefficientDHW_BT2.transpose()
     
        if SetUpScenarios.numberOfBuildings_BT3 >=1:
            #Create pandas dataframe for displaying the results of BT3
            outputVariables_list_BT3 = [model.param_helpTimeSlots_BT3, model.variable_electricalPowerTotal_BT3, model.variable_pvGeneration_BT3, model.variable_windPowerAssigned_BT3, model.variable_currentChargingPowerEV_BT3, model.variable_energyLevelEV_BT3, model.variable_SOC_EV_BT3, model.param_electricalDemand_In_W_BT3, model.param_pvGenerationNominal_BT3, model.param_outSideTemperature_In_C, model.param_windAssignedNominal_BT3, model.param_availabilityPerTimeSlotOfEV_BT3, model.param_energyConsumptionEV_Joule_BT3, model.param_electricityPrice_In_Cents, model.set_timeslots]
            optimal_values_list_BT3 = [[pyo.value(model_item[key]) for key in model_item] for model_item in outputVariables_list_BT3] 
            results_BT3 = pd.DataFrame(optimal_values_list_BT3)
            results_BT3= results_BT3.T
            results_BT3 = results_BT3.rename(columns = {0:'timeslot', 1:'variable_electricalPower', 2:'variable_pvGeneration', 3:'variable_windPowerAssigned',  4:'variable_currentChargingPowerEV', 5:'variable_energyLevelEV_kWh', 6:'variable_SOC_EV', 7:'param_electricalDemand_In_W', 8:'param_pvGenerationNominal', 9:'param_outSideTemperature_In_C', 10:'param_windAssignedNominal', 11:'param_availabilityPerTimeSlotOfEV', 12:'param_energyConsumptionEV', 13:'param_PriceElectricity [Cents]', 14:'set_timeslots'})
            cols = ['set_timeslots']
            results_BT3.set_index('set_timeslots', inplace=True)
            results_BT3['variable_SOC_EV'] = results_BT3['variable_SOC_EV'].round(2)
            results_BT3['variable_energyLevelEV_kWh'] = results_BT3['variable_energyLevelEV_kWh']/3600000
            results_BT3['variable_energyLevelEV_kWh'] = results_BT3['variable_energyLevelEV_kWh'].round(2)
            
            filePath_BT3 = folderPath + "/Combined_BT3.csv"
            results_BT3.to_csv(filePath_BT3, index=False,  sep =";") 
            
            #Create output vector in the correct format
            outputVector_chargingPowerEV_BT3 = results_BT3['variable_currentChargingPowerEV'].to_numpy().reshape((SetUpScenarios.numberOfTimeSlotsPerDay,SetUpScenarios.numberOfBuildings_BT3), order='F')
            outputVector_chargingPowerEV_BT3 = outputVector_chargingPowerEV_BT3.transpose()
        
        if SetUpScenarios.numberOfBuildings_BT4 >=1:
           #Create pandas dataframe for displaying the results of BT4
            outputVariables_list_BT4 = [model.param_helpTimeSlots_BT4, model.variable_heatGenerationCoefficient_SpaceHeating_BT4, model.variable_temperatureBufferStorage_BT4,   model.variable_electricalPowerTotal_BT4, model.variable_pvGeneration_BT4, model.variable_windPowerAssigned_BT4,  model.param_heatDemand_In_W_BT4,  model.param_electricalDemand_In_W_BT4, model.param_pvGenerationNominal_BT4, model.param_outSideTemperature_In_C, model.param_windAssignedNominal_BT4,  model.param_COPHeatPump_SpaceHeating_BT4, model.param_electricityPrice_In_Cents, model.set_timeslots]
            #optimal_values_list_BT4 = [[pyo.value(model_item[key]) for key in model_item] for model_item in outputVariables_list_BT4]

            from typing import Union
            optimal_values_list_BT4 = []
            for model_item in outputVariables_list_BT4:
                if isinstance(model_item, Union[pyo.Set, pyo.RangeSet]):
                    optimal_values_list_BT4.append(list(model_item))
                else:
                    optimal_values_list_BT4.append([pyo.value(model_item[k]) for k in model_item.index_set()])

            results_BT4 = pd.DataFrame(optimal_values_list_BT4)
            results_BT4= results_BT4.T
            results_BT4 = results_BT4.rename(columns = {0:'timeslot', 1:'variable_heatGenerationCoefficient_SpaceHeating', 2:'variable_temperatureBufferStorage',  3:'variable_electricalPowerTotal', 4:'variable_pvGeneration', 5:'variable_windPowerAssigned', 6:'param_heatDemand_In_W', 7:'param_electricalDemand_In_W', 8:'param_pvGenerationNominal', 9:'param_outSideTemperature_In_C', 10:'param_windAssignedNominal', 11:'param_COPHeatPump_SpaceHeating', 12:'param_PriceElectricity [Cents]', 13:'set_timeslots'})
            cols = ['set_timeslots']
            results_BT4.set_index('set_timeslots', inplace=True)
            results_BT4['variable_temperatureBufferStorage'] = results_BT4['variable_temperatureBufferStorage'].round(2)
            results_BT4['param_COPHeatPump_SpaceHeating'] = results_BT4['param_COPHeatPump_SpaceHeating'].round(3)
            results_BT4['variable_heatGenerationCoefficient_SpaceHeating'] = results_BT4['variable_heatGenerationCoefficient_SpaceHeating'].round(4)
            filePath_BT4 = folderPath + "/Combined_BT4.csv"
            results_BT4.to_csv(filePath_BT4, index=False,  sep =";")
            
            #Create output vector in the correct format
            outputVector_heatGenerationCoefficientSpaceHeating_BT4 = results_BT4['variable_heatGenerationCoefficient_SpaceHeating'].to_numpy().reshape((SetUpScenarios.numberOfTimeSlotsPerDay,SetUpScenarios.numberOfBuildings_BT4), order='F')
            outputVector_heatGenerationCoefficientSpaceHeating_BT4 = outputVector_heatGenerationCoefficientSpaceHeating_BT4.transpose()
            

        if SetUpScenarios.numberOfBuildings_BT5 >=1:
            #Create pandas dataframe for displaying the results of BT5
            outputVariables_list_BT5 = [model.param_helpTimeSlots_BT5, model.variable_electricalPowerTotal_BT5, model.variable_pvGeneration_BT5, model.variable_windPowerAssigned_BT5, model.variable_currentChargingPowerBAT_BT5,  model.variable_currentDisChargingPowerBAT_BT5, model.variable_energyLevelBAT_BT5, model.variable_SOC_BAT_BT5, model.param_electricalDemand_In_W_BT5, model.param_pvGenerationNominal_BT5, model.param_outSideTemperature_In_C, model.param_windAssignedNominal_BT5, model.param_electricityPrice_In_Cents, model.set_timeslots]
            #optimal_values_list_BT5 = [[pyo.at(model_item[key]) for key in model_item] for model_item in outputVariables_list_BT5]
            optimal_values_list_BT5 = [[pyo.value(model_item[key]) for key in model_item] for model_item in outputVariables_list_BT5]
            results_BT5 = pd.DataFrame(optimal_values_list_BT5)
            results_BT5= results_BT5.T
            results_BT5 = results_BT5.rename(columns = {0:'timeslot', 1:'variable_electricalPower', 2:'variable_pvGeneration', 3:'variable_windPowerAssigned',  4:'variable_currentChargingPowerBAT', 5:'variable_currentDisChargingPowerBAT', 6:'variable_energyLevelBAT_kWh', 7:'variable_SOC_BAT', 8:'param_electricalDemand_In_W', 9:'param_pvGenerationNominal', 10:'param_outSideTemperature_In_C', 11:'param_windAssignedNominal', 12:'param_PriceElectricity [Cents]', 13:'set_timeslots'})
            cols = ['set_timeslots']
            results_BT5.set_index('set_timeslots', inplace=True)
            results_BT5['variable_SOC_BAT'] = results_BT5['variable_SOC_BAT'].round(2)
            results_BT5['variable_energyLevelBAT_kWh'] = results_BT5['variable_energyLevelBAT_kWh']/3600000
            results_BT5['variable_energyLevelBAT_kWh'] = results_BT5['variable_energyLevelBAT_kWh'].round(2)
            
            filePath_BT5 = folderPath + "/Combined_BT5.csv"
            results_BT5.to_csv(filePath_BT5, index=False,  sep =";") 
            
            #Create output vector in the correct format
            outputVector_chargingPowerBAT_BT5 = results_BT5['variable_currentChargingPowerBAT'].to_numpy().reshape((SetUpScenarios.numberOfTimeSlotsPerDay,SetUpScenarios.numberOfBuildings_BT5), order='F')
            outputVector_chargingPowerBAT_BT5 = outputVector_chargingPowerBAT_BT5.transpose()
            outputVector_dischargingPowerBAT_BT5 = results_BT5['variable_currentDisChargingPowerBAT'].to_numpy().reshape((SetUpScenarios.numberOfTimeSlotsPerDay,SetUpScenarios.numberOfBuildings_BT5), order='F')
            outputVector_dischargingPowerBAT_BT5 = outputVector_dischargingPowerBAT_BT5.transpose()


        if SetUpScenarios.numberOfBuildings_BT6 >=1:

            #Create pandas dataframe for displaying the results of BT6
            outputVariables_list_BT6 = [model.param_helpTimeSlots_BT6, model.variable_heatGenerationCoefficient_GasBoiler_BT6, model.variable_heatGenerationCoefficient_ElectricalHeatingElement_BT6, model.variable_energyLevelCombinedStorage_BT6,  model.variable_heatTransferCoefficient_StorageToRoom_BT6, model.variable_temperatureBuilding_BT6,  model.variable_electricalPowerTotal_BT6, model.variable_pvGeneration_BT6, model.variable_windPowerAssigned_BT6,   model.param_heatDemand_In_W_BT6, model.param_DHWDemand_In_W_BT6, model.param_electricalDemand_In_W_BT6, model.param_pvGenerationNominal_BT6, model.param_outSideTemperature_In_C, model.param_windAssignedNominal_BT6,  model.param_electricityPrice_In_Cents , model.set_timeslots]
            optimal_values_list_BT6 = [[pyo.value(model_item[key]) for key in model_item] for model_item in outputVariables_list_BT6]
            results_BT6 = pd.DataFrame(optimal_values_list_BT6)
            results_BT6= results_BT6.T
            results_BT6 = results_BT6.rename(columns = {0:'timeslot', 1:'variable_heatGenerationCoefficient_GasBoiler', 2:'variable_heatGenerationCoefficient_ElectricalHeatingElement', 3:'variable_energyLevelCombinedStorage', 4:'variable_heatTransferCoefficient_StorageToRoom', 5:'variable_temperatureBuilding',  6:'variable_electricalPowerTotal', 7:'variable_PVGeneration', 8:'variable_windPowerAssigned', 9:'param_heatDemand_In_W', 10:'param_DHWDemand_In_W', 11:'param_electricalDemand_In_W', 12:'param_pvGenerationNominal', 13:'param_outSideTemperature_In_C', 14:'param_windAssignedNominal',  15:'param_PriceElectricity [Cents]', 16:'set_timeslots'})
            cols = ['set_timeslots']
            results_BT6.set_index('set_timeslots', inplace=True)
            #Round values
            results_BT6['variable_temperatureBuilding'] = results_BT6['variable_temperatureBuilding'].round(2)
            results_BT6['variable_energyLevelCombinedStorage'] = round((results_BT6['variable_energyLevelCombinedStorage']/3600000),2)

            results_BT6['variable_heatGenerationCoefficient_GasBoiler'] = results_BT6['variable_heatGenerationCoefficient_GasBoiler'].round(4)
            results_BT6['variable_heatGenerationCoefficient_ElectricalHeatingElement'] = results_BT6['variable_heatGenerationCoefficient_ElectricalHeatingElement'].round(4)
            filePath_BT6 = folderPath + "/Combined_BT6.csv"
            results_BT6.to_csv(filePath_BT6, index=False,  sep =";")

            #Create output vector in the correct format
            outputVector_heatGenerationCoefficient_GasBoiler_BT6 = results_BT6['variable_heatGenerationCoefficient_GasBoiler'].to_numpy().reshape((SetUpScenarios.numberOfTimeSlotsPerDay,SetUpScenarios.numberOfBuildings_BT6), order='F')
            outputVector_heatGenerationCoefficient_ElectricalHeatingElement_BT6 = results_BT6['variable_heatGenerationCoefficient_ElectricalHeatingElement'].to_numpy().reshape((SetUpScenarios.numberOfTimeSlotsPerDay,SetUpScenarios.numberOfBuildings_BT6), order='F')
            outputVector_heatTransferCoefficient_StorageToRoom_BT6 = results_BT6['variable_heatTransferCoefficient_StorageToRoom'].to_numpy().reshape((SetUpScenarios.numberOfTimeSlotsPerDay,SetUpScenarios.numberOfBuildings_BT6), order='F')
            outputVector_heatGenerationCoefficient_GasBoiler_BT6 = outputVector_heatGenerationCoefficient_GasBoiler_BT6.transpose()
            outputVector_heatGenerationCoefficient_ElectricalHeatingElement_BT6 = outputVector_heatGenerationCoefficient_ElectricalHeatingElement_BT6.transpose()
            outputVector_heatTransferCoefficient_StorageToRoom_BT6 = outputVector_heatTransferCoefficient_StorageToRoom_BT6.transpose()


        if SetUpScenarios.numberOfBuildings_BT7 >=1:

            #Create pandas dataframe for displaying the results of BT7
            outputVariables_list_BT7 = [model.param_helpTimeSlots_BT7, model.variable_heatGenerationCoefficient_GasBoiler_BT7, model.variable_electricalPowerFanHeater_BT7, model.variable_temperatureBuilding_BT7,  model.variable_electricalPowerTotal_BT7, model.variable_pvGeneration_BT7, model.variable_windPowerAssigned_BT7,   model.param_heatDemand_In_W_BT7, model.param_electricalDemand_In_W_BT7, model.param_pvGenerationNominal_BT7, model.param_outSideTemperature_In_C, model.param_windAssignedNominal_BT7,  model.param_electricityPrice_In_Cents , model.set_timeslots]
            optimal_values_list_BT7 = [[pyo.value(model_item[key]) for key in model_item] for model_item in outputVariables_list_BT7]
            results_BT7 = pd.DataFrame(optimal_values_list_BT7)
            results_BT7= results_BT7.T
            results_BT7 = results_BT7.rename(columns = {0:'timeslot', 1:'variable_heatGenerationCoefficient_GasBoiler', 2:'variable_electricalPowerFanHeater', 3:'variable_temperatureBuilding',  4:'variable_electricalPowerTotal', 5:'variable_PVGeneration', 6:'variable_windPowerAssigned', 7:'param_heatDemand_In_W', 8:'param_electricalDemand_In_W', 9:'param_pvGenerationNominal', 10:'param_outSideTemperature_In_C', 11:'param_windAssignedNominal',  12:'param_PriceElectricity [Cents]', 13:'set_timeslots'})
            cols = ['set_timeslots']
            results_BT7.set_index('set_timeslots', inplace=True)
            #Round values
            results_BT7['variable_temperatureBuilding'] = results_BT7['variable_temperatureBuilding'].round(2)
            results_BT7['variable_heatGenerationCoefficient_GasBoiler'] = results_BT7['variable_heatGenerationCoefficient_GasBoiler'].round(4)
            results_BT7['variable_electricalPowerFanHeater'] = results_BT7['variable_electricalPowerFanHeater'].round(4)
            filePath_BT7 = folderPath + "/Combined_BT7.csv"
            results_BT7.to_csv(filePath_BT7, index=False,  sep =";")

            #Create output vector in the correct format
            outputVector_heatGenerationCoefficient_GasBoiler_BT7 = results_BT7['variable_heatGenerationCoefficient_GasBoiler'].to_numpy().reshape((SetUpScenarios.numberOfTimeSlotsPerDay,SetUpScenarios.numberOfBuildings_BT7), order='F')
            outputVector_electricalPowerFanHeater_BT7 = results_BT7['variable_electricalPowerFanHeater'].to_numpy().reshape((SetUpScenarios.numberOfTimeSlotsPerDay,SetUpScenarios.numberOfBuildings_BT7), order='F')
            outputVector_heatGenerationCoefficient_GasBoiler_BT7 = outputVector_heatGenerationCoefficient_GasBoiler_BT7.transpose()
            outputVector_electricalPowerFanHeater_BT7 = outputVector_electricalPowerFanHeater_BT7.transpose()


        #Create pandas dataframe for displaying the results of the whole residential area

        outputVariables_list_All = [model.variable_surplusPowerTotal, model.variable_surplusPowerPositivePart, model.variable_surplusPowerNegativePart, model.variable_help_isSurplusPowerPositive, model.variable_electricalPowerTotal, model.variable_RESGenerationTotal, model.variable_PVGenerationTotal, model.variable_costsPerTimeSlot, model.variable_revenuePerTimeSlot, model.param_outSideTemperature_In_C, model.param_electricityPrice_In_Cents, model.variable_gasConsumptionPerTimeSlot_kWh,  model.param_BigM_Surplus_Positive, model.param_BigM_Surplus_Negative,  model.variable_objectiveMaximumLoad, model.variable_objectiveSurplusEnergy, model.variable_objectiveCosts,model.variable_objectiveThermalDiscomfort, model.objective_combined_general, model.set_timeslots]
        optimal_values_list_All = [[pyo.value(model_item[key]) for key in model_item] for model_item in outputVariables_list_All] 
        results_All = pd.DataFrame(optimal_values_list_All)
        results_All= results_All.T
        results_All = results_All.rename(columns = { 0:'variable_surplusPowerTotal', 1:'variable_surplusPowerPositivePart', 2:'variable_surplusPowerNegativePart', 3:'variable_help_isSurplusPowerPositive', 4:'variable_electricalPowerTotal', 5:'variable_RESGenerationTotal', 6:'variable_pvGeneration', 7:'variable_costsPerTimeSlot', 8:'variable_revenuePerTimeSlot',  9:'param_outSideTemperature_In_C', 10:'param_electricityPrice_In_Cents', 11:'variable_gasConsumptionInKWH',  12:'param_BigM_Surplus_Positive', 13:'param_BigM_Surplus_Negative', 14:'variable_objectiveMaximumLoad_kW', 15:'variable_objectiveSurplusEnergy_kWh', 16:'variable_objectiveCosts_Euro', 17:'variable_objectiveThermalDiscomfort', 18:'objective_combined_general', 19:'set_timeslots'})
        cols = ['set_timeslots']
        results_All.set_index('set_timeslots', inplace=True)
        results_All ['variable_objectiveMaximumLoad_kW'] = results_All['variable_objectiveMaximumLoad_kW']/1000
        results_All ['variable_objectiveMaximumLoad_kW'] = results_All['variable_objectiveMaximumLoad_kW'].round(2)
        results_All ['variable_objectiveSurplusEnergy_kWh'] = results_All['variable_objectiveSurplusEnergy_kWh'] * ((SetUpScenarios.timeResolution_InMinutes * 60) /3600000)
        results_All ['variable_objectiveSurplusEnergy_kWh'] = results_All['variable_objectiveSurplusEnergy_kWh'].round(2)
        results_All ['variable_objectiveCosts_Euro'] = results_All['variable_objectiveCosts_Euro']/100
        results_All ['variable_objectiveCosts_Euro'] = results_All['variable_objectiveCosts_Euro'].round(2)
        results_All['variable_objectiveThermalDiscomfort'] = results_All['variable_objectiveThermalDiscomfort'].round(2)
        results_All ['objective_combined_general'] = results_All['objective_combined_general'].round(2)
        results_All['variable_gasConsumptionInKWH'] = results_All['variable_gasConsumptionInKWH'].round(2)
        filePath_All = folderPath + "/Combined_WholeResidentialArea.csv"
        results_All.to_csv(filePath_All, index=True,  sep =";")


    
        #Read the just created result files for all buildings and subdivide them into a file for each building
        sleep(0.5)
        if SetUpScenarios.numberOfBuildings_BT1 >=1:
            multiple_dataframes_BT1 = list(pd.read_csv(filePath_BT1, sep =";", chunksize=SetUpScenarios.numberOfTimeSlotsPerDay))
            for index in range (0, len(multiple_dataframes_BT1)):
                individual_dataframe_BT1 = multiple_dataframes_BT1[index]
                individual_dataframe_BT1.set_index('timeslot', inplace=True)
                filePath_Individual_BT1 = folderPath + "/BT1_Building_" + str(index+1) + ".csv"
                individual_dataframe_BT1.to_csv(filePath_Individual_BT1, index=True,  sep =";") 
      
        if SetUpScenarios.numberOfBuildings_BT2 >=1:
            multiple_dataframes_BT2 = list(pd.read_csv(filePath_BT2, sep =";", chunksize=SetUpScenarios.numberOfTimeSlotsPerDay))
            for index in range (0, len(multiple_dataframes_BT2)):
                individual_dataframe_BT2 = multiple_dataframes_BT2[index]
                individual_dataframe_BT2.set_index('timeslot', inplace=True)
                filePath_Individual_BT2 = folderPath + "/BT2_Building_" + str(index+1) + ".csv"
                individual_dataframe_BT2.to_csv(filePath_Individual_BT2, index=True,  sep =";") 
      
        if SetUpScenarios.numberOfBuildings_BT3 >=1:
            multiple_dataframes_BT3 = list(pd.read_csv(filePath_BT3, sep =";", chunksize=SetUpScenarios.numberOfTimeSlotsPerDay))
            for index in range (0, len(multiple_dataframes_BT3)):
                individual_dataframe_BT3 = multiple_dataframes_BT3[index]
                individual_dataframe_BT3.set_index('timeslot', inplace=True)
                filePath_Individual_BT3 = folderPath + "/BT3_Building_" + str(index+1) + ".csv"
                individual_dataframe_BT3.to_csv(filePath_Individual_BT3, index=True,  sep =";") 
    
        if SetUpScenarios.numberOfBuildings_BT4 >=1:
            multiple_dataframes_BT4 = list(pd.read_csv(filePath_BT4, sep =";", chunksize=SetUpScenarios.numberOfTimeSlotsPerDay))
            for index in range (0, len(multiple_dataframes_BT4)):
                individual_dataframe_BT4 = multiple_dataframes_BT4[index]
                individual_dataframe_BT4.set_index('timeslot', inplace=True)
                filePath_Individual_BT4 = folderPath + "/BT4_Building_" + str(index+1) + ".csv"
                individual_dataframe_BT4.to_csv(filePath_Individual_BT4, index=True,  sep =";") 
                
        if SetUpScenarios.numberOfBuildings_BT5 >=1:
            multiple_dataframes_BT5 = list(pd.read_csv(filePath_BT5, sep =";", chunksize=SetUpScenarios.numberOfTimeSlotsPerDay))
            for index in range (0, len(multiple_dataframes_BT5)):
                individual_dataframe_BT5 = multiple_dataframes_BT5[index]
                individual_dataframe_BT5.set_index('timeslot', inplace=True)
                filePath_Individual_BT5 = folderPath + "/BT5_Building_" + str(index+1) + ".csv"
                individual_dataframe_BT5.to_csv(filePath_Individual_BT5, index=True,  sep =";")

        if SetUpScenarios.numberOfBuildings_BT6 >=1:
            multiple_dataframes_BT6 = list(pd.read_csv(filePath_BT6, sep =";", chunksize=SetUpScenarios.numberOfTimeSlotsPerDay))
            for index in range (0, len(multiple_dataframes_BT6)):
                individual_dataframe_BT6 = multiple_dataframes_BT6[index]
                individual_dataframe_BT6.set_index('timeslot', inplace=True)
                filePath_Individual_BT6 = folderPath + "/BT6_Building_" + str(index+1) + ".csv"
                individual_dataframe_BT6.to_csv(filePath_Individual_BT6, index=True,  sep =";")

        if SetUpScenarios.numberOfBuildings_BT7 >=1:
            multiple_dataframes_BT7 = list(pd.read_csv(filePath_BT7, sep =";", chunksize=SetUpScenarios.numberOfTimeSlotsPerDay))
            for index in range (0, len(multiple_dataframes_BT7)):
                individual_dataframe_BT7 = multiple_dataframes_BT7[index]
                individual_dataframe_BT7.set_index('timeslot', inplace=True)
                filePath_Individual_BT7 = folderPath + "/BT7_Building_" + str(index+1) + ".csv"
                individual_dataframe_BT7.to_csv(filePath_Individual_BT7, index=True,  sep =";")
    
        
    elif (solution.solver.termination_condition == TerminationCondition.infeasible):
        # Print something when model is infeasible
        print ("Result Status: Infeasible")
    else:
        # Something else is wrong
        print("Solver Status: ", solution.solver.status)

    
    # Close the log file if output is printed on a log file and not on the console
    if printLogToFile == True:
        sys.stdout.close()
        sys.stdout = prev_stdout
          
    #Check if the outputs of the buildings have been assigned. If not, assign a value of -1 to them 
    if 'outputVector_heatGenerationCoefficientSpaceHeating_BT1' in locals():
        pass
    else:
        outputVector_heatGenerationCoefficientSpaceHeating_BT1 = np.zeros(0)       
    
    if 'outputVector_heatGenerationCoefficientDHW_BT1' in locals():
        pass
    else:
        outputVector_heatGenerationCoefficientDHW_BT1 = np.zeros(0)  
    
    if 'outputVector_chargingPowerEV_BT1' in locals():
        pass
    else:
        outputVector_chargingPowerEV_BT1 = np.zeros(0) 
   
    if 'outputVector_heatGenerationCoefficientSpaceHeating_BT2' in locals():
        pass
    else:
        outputVector_heatGenerationCoefficientSpaceHeating_BT2 = np.zeros(0)
   
    if 'outputVector_heatGenerationCoefficientDHW_BT2' in locals():
        pass
    else:
        outputVector_heatGenerationCoefficientDHW_BT2 = np.zeros(0)
   
    if 'outputVector_chargingPowerEV_BT3' in locals():
        pass
    else:
        outputVector_chargingPowerEV_BT3 = np.zeros(0)
   
    if 'outputVector_heatGenerationCoefficientSpaceHeating_BT4' in locals():
        pass
    else:
        outputVector_heatGenerationCoefficientSpaceHeating_BT4 = np.zeros(0)
        
    if 'outputVector_chargingPowerBAT_BT5' in locals():
        pass
    else:
        outputVector_chargingPowerBAT_BT5 = np.zeros(0)
        
    if 'outputVector_dischargingPowerBAT_BT5' in locals():
        pass
    else:
        outputVector_dischargingPowerBAT_BT5 = np.zeros(0)

    if 'outputVector_heatGenerationCoefficient_GasBoiler_BT6' in locals():
        pass
    else:
        outputVector_heatGenerationCoefficient_GasBoiler_BT6 = np.zeros(0)

    if 'outputVector_heatGenerationCoefficient_ElectricalHeatingElement_BT6' in locals():
        pass
    else:
        outputVector_heatGenerationCoefficient_ElectricalHeatingElement_BT6 = np.zeros(0)

    if 'outputVector_heatTransferCoefficient_StorageToRoom_BT6' in locals():
        pass
    else:
        outputVector_heatTransferCoefficient_StorageToRoom_BT6 = np.zeros(0)

    if 'outputVector_heatGenerationCoefficient_GasBoiler_BT7' in locals():
        pass
    else:
        outputVector_heatGenerationCoefficient_GasBoiler_BT7 = np.zeros(0)

    if 'outputVector_electricalPowerFanHeater_BT7' in locals():
        pass
    else:
        outputVector_electricalPowerFanHeater_BT7 = np.zeros(0)


    if includeObjectivesInReturnStatementCentralized == False:
        return outputVector_heatGenerationCoefficientSpaceHeating_BT1, outputVector_heatGenerationCoefficientDHW_BT1, outputVector_chargingPowerEV_BT1, outputVector_heatGenerationCoefficientSpaceHeating_BT2, outputVector_heatGenerationCoefficientDHW_BT2, outputVector_chargingPowerEV_BT3, outputVector_heatGenerationCoefficientSpaceHeating_BT4, outputVector_chargingPowerBAT_BT5, outputVector_dischargingPowerBAT_BT5, outputVector_heatGenerationCoefficient_GasBoiler_BT6, outputVector_heatGenerationCoefficient_ElectricalHeatingElement_BT6, outputVector_heatTransferCoefficient_StorageToRoom_BT6, outputVector_heatGenerationCoefficient_GasBoiler_BT7, outputVector_electricalPowerFanHeater_BT7

    if includeObjectivesInReturnStatementCentralized == True:
        objectiveMaximumLoad = pyo.value(model.variable_objectiveMaximumLoad)
        objectiveSurplusEnergy = pyo.value(model.variable_objectiveSurplusEnergy)
        objectiveCosts = pyo.value(model.variable_objectiveCosts)
        objectiveThermalDiscomfort = pyo.value(model.variable_objectiveThermalDiscomfort)

        return outputVector_heatGenerationCoefficientSpaceHeating_BT1, outputVector_heatGenerationCoefficientDHW_BT1, outputVector_chargingPowerEV_BT1, outputVector_heatGenerationCoefficientSpaceHeating_BT2, outputVector_heatGenerationCoefficientDHW_BT2, outputVector_chargingPowerEV_BT3, outputVector_heatGenerationCoefficientSpaceHeating_BT4, outputVector_chargingPowerBAT_BT5, outputVector_dischargingPowerBAT_BT5, outputVector_heatGenerationCoefficient_GasBoiler_BT6, outputVector_heatGenerationCoefficient_ElectricalHeatingElement_BT6, outputVector_heatTransferCoefficient_StorageToRoom_BT6, outputVector_heatGenerationCoefficient_GasBoiler_BT7, outputVector_electricalPowerFanHeater_BT7, objectiveMaximumLoad, objectiveSurplusEnergy, objectiveCosts, objectiveThermalDiscomfort, mipGapPercentOfFoundSolution, timeForFindingOptimalSolution
