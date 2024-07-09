"""
This file sets up the scenarios for the residential area by specifying all the relevant parameters of the  heat pump, building, EV,
stationary battery (not used in the paper), gas boiler (not used in the paper), fan heater (not used in the paper) and solver
"""

import pandas as pd
import numpy as np
from random import random

#Set the parameters for the model
import os
import config

#Set some parameters for the model


#Specify the number of Buildings buildings
numberOfBuildings_BT1 = 4 * 3
numberOfBuildings_BT2 = 4 * 3
numberOfBuildings_BT3 = 0
numberOfBuildings_BT4 = 2 * 3
numberOfBuildings_BT5 = 0
numberOfBuildings_BT6 = 0
numberOfBuildings_BT7 = 0
numberOfBuildings_Total = numberOfBuildings_BT1 + numberOfBuildings_BT2 + numberOfBuildings_BT3 + numberOfBuildings_BT4 + numberOfBuildings_BT5 + numberOfBuildings_BT6 + numberOfBuildings_BT7


#Solver options
solverOption_relativeGap_normalDecentral= 1.0 / 100 # Unit: [%/100] = [%] / [100]
solverOption_timeLimit_normalDecentral = 5 * 60 # Unit: [seconds] =[min]*[seconds/min]
solverOption_relativeGap_Central = 0.1 / 100 # Unit: [%/100] = [%] / [100]
solverOption_timeLimit_Central = 10 * 60 # Unit: [seconds] =[min]*[seconds/min]

#Parameters of the renewable Energy Sources (RES): PV system and windTurbine
averagePVPeak = 5000 # Unit: [W]
maximalDeviationFromPVPeak = 3000 # Unit: [W]
percentageBuildingsWithPV = 00  # Unit: [%]
powerOfWindTurbinePerBuilding = 0 * 1000   # Unit: [W]=[kW]*[W/kW]
maximalPowerOfWindTurbine = numberOfBuildings_Total * powerOfWindTurbinePerBuilding   # Unit: [W]=[kW]*[W/kW]
revenueForFeedingBackElecticityIntoTheGrid_CentsPerkWh = 0


numberOfBuildings_WithEV = numberOfBuildings_BT1 + numberOfBuildings_BT3

considerWindTurbine = False

useRandomlyAssignedAvailabilityPatternsForTheEV = False
useMonteCarloMethodForScenarioCreation = False
numberOfScenariosPerDayUsingMonteCarloMethod = 5

# Choose price data for the Optimization; Options: 'Basic', 'Scaled'
typeOfPriceData = 'Scaled'

timeResolution_InMinutes = 30 # Unit: [minutes]
numberOfTimeSlotsPerDay = int(1440/timeResolution_InMinutes)
alternativeCaseScenario = False


#Parameters of the thermal storage systems
maximalBufferStorageTemperature = 23 # Unit: [°C]
minimalBufferStorageTemperature = 21 # Unit: [°C]
initialBufferStorageTemperature = 21.75 # Unit: [°C]
idealComfortTemperature = 22  # Unit: [°C]
allowedTemperatureDeviationForOptimalComfort = 0.5 # Unit: [°C]
endBufferStorageTemperatureAllowedDeviationFromInitalValue = 0.4  # Unit: [°C]
endBufferStorageTemperatureAllowedDeviationFromInitalValue_ForCorrection = 1000  # Unit: [°C]
temperatureOfTheHotWaterInTheDHWTank = 45 # Unit: [°C]
supplyTemperatureOfTheSpaceHeating = 30 # Unit: [°C]
capacityOfBufferStorage = 9800 # Unit: [liter]; Calculation: 140 m^2 (Heated area) * 0,07 m (width of concrete)
capacityOfBufferStorage_BT4_MFH = 63000# Unit: [liter]  #12 appartments a 75m^2 (Heated area) * 0,07 m (width of concrete), 29 people living there
capacityOfDHWTank = 250 # Unit: [liter]

initialUsableVolumeDHWTank = 125 # Unit: [liter]
endUsableVolumeDHWTankAllowedDeviationFromInitialValue = 30  # Unit: [liter] --> For the calculation of the discomfort
endUsableVolumeDHWTankAllowedDeviationFromInitialValueOptimization = 0  # Unit: [liter]
endUsableVolumeDHWTankAllowedDeviationFromInitialValue_ForCorrection = 1000  # Unit: [liter]
densityOfWater = 1 # Unit: [kg/l]
densityOfCement = 2.4 # Unit: [kg/l]
specificHeatCapacityOfWater = 4181 # Unit: [J/kg * K]
specificHeatCapacityOfCement = 1000 # Unit: [J/kg * K]
standingLossesBufferStorage = 45  # Unit: [W]
standingLossesBufferStorage_BT4_MFH = 45 * 5  # Unit: [W]
standingLossesDHWTank = 45  # Unit: [W]

maximumBufferStorageTemperature_CorrectionNecessary = 23.1 # Unit: [°C]
minimumBufferStorageTemperature_CorrectionNecessary = 20.9 # Unit: [°C]

maximumBufferStorageTemperature_PhysicalLimit = 23.3 # Unit: [°C]
minimumBufferStorageTemperature_PhysicalLimit = 20.7 # Unit: [°C]

maximumBufferStorageTemperature_ConventionalControl = 22.5 # Unit: [°C]
minimumBufferStorageTemperature_ConventionalControl = 21.1 # Unit: [°C]

maximumUsableVolumeDHWTank_CorrectionNecessary = 255 # Unit: [liter]
minimumUsableVolumeDHWTank_CorrectionNecessary = 45 # Unit: [liter]

maximumUsableVolumeDHWTank_PhysicalLimit = 270 # Unit: [liter]
minimumUsableVolumeDHWTank_PhysicalLimit = 0 # Unit: [liter]

maximumCapacityDHWTankOptimization = 240 # Unit: [liter]
minimumCapacityDHWTankOptimization = 60 # Unit: [liter]

maximumUsableVolumeDHWTank_ConventionalControl = 200 # Unit: [liter]
minimumUsableVolumeDHWTank_ConventionalControl = 60 # Unit: [liter]


#Adjust parameters to the time resolution
if timeResolution_InMinutes ==30:
    maximalBufferStorageTemperature = 22.8 # Unit: [°C]
    minimalBufferStorageTemperature = 21.2 # Unit: [°C]
    
    maximumBufferStorageTemperature_CorrectionNecessary = 23.2 # Unit: [°C]
    minimumBufferStorageTemperature_CorrectionNecessary = 20.8 # Unit: [°C]
    
    maximumBufferStorageTemperature_PhysicalLimit = 23.8 # Unit: [°C]
    minimumBufferStorageTemperature_PhysicalLimit = 20.2 # Unit: [°C]
    
    maximumBufferStorageTemperature_ConventionalControl = 22.8 # Unit: [°C]
    minimumBufferStorageTemperature_ConventionalControl = 21.2 # Unit: [°C]
    
    maximumUsableVolumeDHWTank_CorrectionNecessary = 260 # Unit: [liter]
    minimumUsableVolumeDHWTank_CorrectionNecessary = 40 # Unit: [liter]
 
    maximumCapacityDHWTankOptimization = 250 # Unit: [liter]
    minimumCapacityDHWTankOptimization = 50 # Unit: [liter]
    
    maximumUsableVolumeDHWTank_ConventionalControl = 160 # Unit: [liter]
    minimumUsableVolumeDHWTank_ConventionalControl = 80 # Unit: [liter]
    

    
if timeResolution_InMinutes ==60:
    
    maximalBufferStorageTemperature = 23.2 # Unit: [°C]
    minimalBufferStorageTemperature = 20.8 # Unit: [°C]
    
    maximumBufferStorageTemperature_CorrectionNecessary = 23.3 # Unit: [°C]
    minimumBufferStorageTemperature_CorrectionNecessary = 20.7 # Unit: [°C]
    
    maximumBufferStorageTemperature_PhysicalLimit = 23.5 # Unit: [°C]
    minimumBufferStorageTemperature_PhysicalLimit = 20.5 # Unit: [°C]
    
    maximumBufferStorageTemperature_ConventionalControl = 22.8 # Unit: [°C]
    minimumBufferStorageTemperature_ConventionalControl = 21.2 # Unit: [°C]
    
    maximumUsableVolumeDHWTank_CorrectionNecessary = 265 # Unit: [liter]
    minimumUsableVolumeDHWTank_CorrectionNecessary = 35 # Unit: [liter]
    
    maximumCapacityDHWTankOptimization = 260 # Unit: [liter]
    minimumCapacityDHWTankOptimization = 40 # Unit: [liter]
    
    maximumUsableVolumeDHWTank_ConventionalControl = 220 # Unit: [liter]
    minimumUsableVolumeDHWTank_ConventionalControl = 50 # Unit: [liter] 


#Parameters of the heat pump

#Bosch Compress 6800i AW  series
electricalPower_HP = 3000  # Unit: [W]
electricalPower_HP_BT4_MFH = 5 * 3000  # Unit: [W]
COP_CalculationValue1_TemperatureDifference  = 28 # Unit: [K]
COP_CalculationValue1_COP = 4.8   # Unit: dimensionless
COP_CalculationValue2_TemperatureDifference = 33  # Unit: [K]
COP_CalculationValue2_COP = 3.9   # Unit: dimensionless
minimalModulationdDegree_HP = 20   # Unit: [%]





#Parameters of the electric vehicle EV (currently Opel Ampera-e)
capacityMaximal_EV = 60  * 3600000 # Unit: [J]=[kWh]*[J/kWh]
initialSOC_EV = 40  # Unit: [%]
endSOC_EVAllowedDeviationFromInitalValue = 5   # Unit: [%]
endSOC_EVAllowedDeviationFromInitalValue_ForCorrection = 5   # Unit: [%]
targetSOCAtEndOfOptimization_EV  = initialSOC_EV  # Unit: [%]
chargingEfficiency_EV = 89   # Unit: [%]
chargingPowerMaximal_EV  = 4.7 * 1000  # Unit: [W]=[kW]*[W/kW]
energyConsumptionPer100km = 17.5 * 3600000  # Unit: [J]=[kWh]*[J/kWh]
averageLengthOfRides_km = 45  # Unit: [km] 
maximalDeviationOfRides_km  = 25 * 2   # Unit: [km]
numberOfDifferntAvailabilityPatterns = 20
modulationDegreeCharging_ConventionalControl = 100
timeslotInMinutesForForcedChargingOfTheEV = 1440 * 0.85 # Unit: Minute of the day
socThresholdForForcedCharging = 30  # Unit: [%]


#Parameters of the battery storage system (currently sonnenBatterie 10)
capacityMaximal_BAT = 5  * 3600000 # Unit: [J]=[kWh]*[J/kWh]
initialSOC_BAT = 0  # Unit: [%]
endSOC_BATAllowedDeviationFromInitalValueLowerLimit = 1000   # Unit: [%]
endSOC_BATAllowedDeviationFromInitalValueUpperLimit = 1000  # Unit: [%]
endSOC_BATAllowedDeviationFromInitalValue_ForCorrection = 1000   # Unit: [%]
targetSOCAtEndOfOptimization_BAT  = initialSOC_BAT  # Unit: [%]
chargingEfficiency_BAT = 0.968   # Unit:
dischargingEfficiency_BAT = 0.968 * 0.95   # Unit:
chargingPowerMaximal_BAT  = 3.4 * 1000  # Unit: [W]=[kW]*[W/kW]


#Parameters for BT6:Gas boiler with electrical heating element in a combined storage tank
volumeOfTheCombinedStorage = 492   # Unit: [l]
standingLossesCombinedStorage = 63    # Unit: [W]
supplyTemperatureSpaceHeating_BT6 = 60    # Unit: [°C]
returnTemperatureSpaceHeating_BT6 = 45    # Unit: [°C]
usableVolumeOfCombinedStorageForDHW = 192   # Unit: [l]
temperatureOfTheHotWaterInTheCombinedStorage = 50    # Unit: [°C]
heatCapacityOfBuildingMassPerSquareMeter = 165000  # Unit: [J/K*m^2]
areaOfTheBuilding = 140  # Unit: [m^2]
totalHeatCapacityOfTheBuilding = heatCapacityOfBuildingMassPerSquareMeter * areaOfTheBuilding # Unit: [J/K]
minimalModulationdDegree_GasBoiler = (1900 / 11000) *100  #Unit: [%]
maximalPowerGasBoiler = 11000   # Unit: [W]
maximalPowerHeatingSystem = maximalPowerGasBoiler  #Unit: [W]
efficiency_GasBoiler = 0.99 # Unit: [W]
maximalPowerElectricalHeatingElement = 6000 # Unit: [W]
efficiency_ElectricalHeatingElement = 0.99 #Unit: Dimensionless quantity
minimumTemperatureBuilding = 21 # Unit: [°C]
maximumTemperatureBuilding = 23  # Unit: [°C]
initialTemperatureBuilding = 22  # Unit: [°C]
minimumPhysicalTemperatureBuilding = 19 # Unit: [°C]
maximumPhysicalTemperatureBuilding = 27 # Unit: [°C]
endTemperatureBuildingAllowedDeviationFromInitalValue = 100  # Unit: [°C]
maximumEnergyContentCombinedStorage = (volumeOfTheCombinedStorage - usableVolumeOfCombinedStorageForDHW) * (supplyTemperatureSpaceHeating_BT6 -returnTemperatureSpaceHeating_BT6) * specificHeatCapacityOfWater + usableVolumeOfCombinedStorageForDHW * (temperatureOfTheHotWaterInTheCombinedStorage - 10) * specificHeatCapacityOfWater
initialEnergyContentCombinedStorage = 0.25 * maximumEnergyContentCombinedStorage
priceForGasInCentPerKWH = 6   # Unit: [Cent]

#Additional parameters for BT7: Bas boiler with electrical fan heater
efficiency_ElectricalFanHeater = 0.99 #Unit: Dimensionless quantity
electricalPowerFanHeater_Stage1 = 900 # Unit: [W]
electricalPowerFanHeater_Stage2 = 1300 # Unit: [W]
electricalPowerFanHeater_Stage3 = 2000 # Unit: [W]




#Determine Energy Consumption of EVs

numberOfEVsTotal = numberOfBuildings_BT1 + numberOfBuildings_BT3

energyConsumptionOfEVs_Joule = np.zeros( numberOfTimeSlotsPerDay)

lengthOfRidesInKMForTheDifferentEV = np.zeros(numberOfEVsTotal)
totalEnergyConusumptionPerRideInJoule = np.zeros(numberOfEVsTotal)
indexOfAvailabilityPatternForTheEV= np.zeros(numberOfEVsTotal).astype(int)
numberOfDrivingTimeSlotsForTheEV = np.zeros(numberOfEVsTotal)


#This method generates the energy consumption pattern for the EV based on its driving pattern and length of their daily rydes
def generateEVEnergyConsumptionPatterns( array_AvailabilityForTheEV, indexWithinAllEVs):


    #Calculate length of rides for the EVs
    if (numberOfEVsTotal>1):
        helpValueIncrementKMPerVehicle = maximalDeviationOfRides_km/(numberOfEVsTotal-1)
    else:
        helpValueIncrementKMPerVehicle = maximalDeviationOfRides_km

        
    for i in range (numberOfEVsTotal):
        lengthOfRidesInKMForTheDifferentEV[i]= averageLengthOfRides_km - 0.5*maximalDeviationOfRides_km +i*helpValueIncrementKMPerVehicle
    if numberOfEVsTotal ==1:
        lengthOfRidesInKMForTheDifferentEV[0] = averageLengthOfRides_km
    mixTheValuesOfAnArray(lengthOfRidesInKMForTheDifferentEV)
    for i in range (numberOfEVsTotal):
        totalEnergyConusumptionPerRideInJoule[i] = (lengthOfRidesInKMForTheDifferentEV[i]/100) * energyConsumptionPer100km
    
    availabilityPatternOfEV = array_AvailabilityForTheEV.copy();

        
    #Determine number of driving time slots of the EV
    numberOf0EntriesInTheArray =0 #A 0-Entry in the availability dataset of the EVs means that during this time slot the EV was driving
    for j in range (len(availabilityPatternOfEV)):
        if availabilityPatternOfEV[j] == 0:
            numberOf0EntriesInTheArray = numberOf0EntriesInTheArray + 1
    numberOfDrivingTimeSlotsForTheEV  = numberOf0EntriesInTheArray
    


    #Calculate the energy consumption for every timeslot when driving (assuming a constant energy use during the rides)

    constantEnergyPerTimeSlot = totalEnergyConusumptionPerRideInJoule [indexWithinAllEVs] / numberOfDrivingTimeSlotsForTheEV 
    for j in range (len(availabilityPatternOfEV)):
        if availabilityPatternOfEV[j] == 0:
            energyConsumptionOfEVs_Joule [j] = constantEnergyPerTimeSlot
        else:
            energyConsumptionOfEVs_Joule [j] = 0

    return energyConsumptionOfEVs_Joule
                
               
                
#Determine the PV peak of the different buildings (not considered in the paper)

def determinePVPeakOfBuildings(indexBuildingTotal):
    pvPeaksOfBuildings = np.zeros (numberOfBuildings_Total)
    numberOfBuildingsWithoutPV = int(numberOfBuildings_Total * (1 - (percentageBuildingsWithPV/100)))
    numberOfBuildingWithPV = numberOfBuildings_Total - numberOfBuildingsWithoutPV
    
    if (numberOfBuildingWithPV>1):
        helpValue_pvIncrementPerBuilding = (2*maximalDeviationFromPVPeak)/(numberOfBuildingWithPV - 1)
    else:
        helpValue_pvIncrementPerBuilding =0
    
    for i in range (numberOfBuildingsWithoutPV):
        pvPeaksOfBuildings [i] = 0
    for i in range (numberOfBuildingWithPV):
        pvPeaksOfBuildings [numberOfBuildingsWithoutPV + i] = (averagePVPeak - maximalDeviationFromPVPeak) + i * helpValue_pvIncrementPerBuilding
    if numberOfBuildingWithPV == 1:
        pvPeaksOfBuildings [numberOfBuildingsWithoutPV] = averagePVPeak
    

    
    mixTheValuesOfAnArray(pvPeaksOfBuildings)
    pvPeaksOfBuildings  = pvPeaksOfBuildings.round(0)

    return pvPeaksOfBuildings [indexBuildingTotal]
        
     

#Assign wind power to the different buildings with an equal distribution (not considered in the paper)

def calculateAssignedWindPowerNominalPerBuilding (currentDay, indexOfBuildingTotal):
    windPowerAssignedNominalPerBuilding = np.zeros ((numberOfBuildings_Total, numberOfTimeSlotsPerDay))
    
    df_windData_original = pd.read_csv(os.path.join(config.DIR_OUTSIDE_TEMPERATURE_ONE_MINUTE_DAYS, 'Outside_Temperature_1Minute_Day' + str(currentDay) + '.csv'), sep =";")
    
    #Adjust time resolution
    df_windData_original['Time'] = pd.to_datetime(df_windData_original['Time'], format = '%d.%m.%Y %H:%M')
    df_windData = df_windData_original.set_index('Time').resample(str(timeResolution_InMinutes) +'Min').mean()
    arrayTimeSlots = [i for i in range (1, numberOfTimeSlotsPerDay + 1)]
    df_windData['Timeslot'] = arrayTimeSlots
    df_windData = df_windData.set_index('Timeslot')    
    
    #Equal assignment of nominal wind generation to the buildings
    if considerWindTurbine == True:
        for i in range (numberOfBuildings_Total):
            for j in range (numberOfTimeSlotsPerDay):
                windPowerAssignedNominalPerBuilding [i] [j] = df_windData ['Wind [nominal]'] [j] / numberOfBuildings_Total
    
    return windPowerAssignedNominalPerBuilding [indexOfBuildingTotal]
        



#Calculate the COP of the (air-source) heat pump for all time slots based on outside temperature values
def calculateCOP(temperatureValues):
    cop_heatPump_SpaceHeating = np.zeros(numberOfTimeSlotsPerDay)
    cop_heatPump_DHW = np.zeros(numberOfTimeSlotsPerDay)

    
    for i in range (0, numberOfTimeSlotsPerDay):
        temperatureDifferenceSinkSource_SpaceHeating = supplyTemperatureOfTheSpaceHeating - temperatureValues[i + 1]
        temperatureDifferenceSinkSource_DHW = temperatureOfTheHotWaterInTheDHWTank - temperatureValues[i + 1]
        linearEquation_slope_m = (COP_CalculationValue2_COP - COP_CalculationValue1_COP)/(COP_CalculationValue2_TemperatureDifference - COP_CalculationValue1_TemperatureDifference)
        linearEquation_intersection_c = COP_CalculationValue2_COP - linearEquation_slope_m * COP_CalculationValue2_TemperatureDifference
        cop_heatPump_SpaceHeating [i] = linearEquation_intersection_c + linearEquation_slope_m * temperatureDifferenceSinkSource_SpaceHeating
        if cop_heatPump_SpaceHeating [i]<1:
            cop_heatPump_SpaceHeating [i] =1
        cop_heatPump_DHW [i] = linearEquation_intersection_c + linearEquation_slope_m * temperatureDifferenceSinkSource_DHW
        if cop_heatPump_DHW [i]<1:
            cop_heatPump_DHW [i] = 1
            
    return cop_heatPump_SpaceHeating, cop_heatPump_DHW


#Calculate the COP of the heat pump for a single time slot
def calculateCOP_SingleTimeSlot(temperatureValue):
    cop_heatPump_SpaceHeating = np.zeros(numberOfTimeSlotsPerDay)
    cop_heatPump_DHW = np.zeros(numberOfTimeSlotsPerDay)
    
    temperatureDifferenceSinkSource_SpaceHeating = supplyTemperatureOfTheSpaceHeating - temperatureValue
    temperatureDifferenceSinkSource_DHW = temperatureOfTheHotWaterInTheDHWTank - temperatureValue
    linearEquation_slope_m = (COP_CalculationValue2_COP - COP_CalculationValue1_COP)/(COP_CalculationValue2_TemperatureDifference - COP_CalculationValue1_TemperatureDifference)
    linearEquation_intersection_c = COP_CalculationValue2_COP - linearEquation_slope_m * COP_CalculationValue2_TemperatureDifference
    cop_heatPump_SpaceHeating  = linearEquation_intersection_c + linearEquation_slope_m * temperatureDifferenceSinkSource_SpaceHeating
    if cop_heatPump_SpaceHeating <1:
        cop_heatPump_SpaceHeating =1
    cop_heatPump_DHW = linearEquation_intersection_c + linearEquation_slope_m * temperatureDifferenceSinkSource_DHW
    if cop_heatPump_DHW <1:
        cop_heatPump_DHW = 1
            
    return cop_heatPump_SpaceHeating, cop_heatPump_DHW


#This method mixes the values of the array lengthOfRidesInKMForTheDifferentEV and pVPeak such that there is no strong concentration of the values at the end of the array but rather the values are more equally distributed
def mixTheValuesOfAnArray(array):
    helpCounter =0
    for i in range (len(array)):
        helpCounter= helpCounter+1
        if helpCounter==2 and  i<len(array)/2:
            helpCounter =0
            firstValueOfArray = array [i]
            secondValueOfArray = array [len(array) - (i+1)]
            array [i] = secondValueOfArray
            array [len(array) - (i+1)] = firstValueOfArray
