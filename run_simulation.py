import fastf1
import os
import math 
import random 

from model import Model
from simulation import Simulator
from strategy import StrategyCreator

from main import saveResults

import argparse

parser = argparse.ArgumentParser(description='Run F1 Strategy Simulation for a specific place.')

parser.add_argument('--place', type=str, help='The name of the place/event to run the simulation for.')
parser.add_argument('--year', type=int, help='The year of the event to run the simulation for.')
parser.add_argument('--simulations', type=int, default=10000, help='The number of simulations to run for each strategy pair.')
parser.add_argument('--force', action='store_true', help='Force re-run the simulation even if data already exists.')
parser.add_argument('--output', type=str, default='data/', help='The directory to save the simulation results.')


args = parser.parse_args()

year = args.year if args.year else 2024
place = args.place + " Grand Prix"
simulations = args.simulations
force = args.force
output_dir = args.output

def runSimulationWithPlace(place, year, simulations):
    model = Model(year, place)
    model.getLaps()
    if model.getLaps() is None:
        print(f"No lap data available for {place}. Skipping simulation.")
        return
    model.createCompoundModels()

    simulator = Simulator(model)

    strategyCreator = StrategyCreator(model)
    oneStopStrategies, twoStopStrategies = strategyCreator.createAllPossibleStrategies()

    results = runSampledMontecarlo(oneStopStrategies, twoStopStrategies, simulator, simulations)
    saveResults(results, model)

def runSampledMontecarlo(oneStopStrategies, twoStopStrategies, simulator, simulations):
    n = min(len(oneStopStrategies), len(twoStopStrategies))
    print(f"Total one stop strategies: {len(oneStopStrategies)}, Total two stop strategies: {len(twoStopStrategies)}. Sampling {n} strategies from each for the Monte Carlo simulation.")
    results = []
    repeatFactor = math.ceil(simulations / n)
    print(f"Running {n} strategy pairs with {repeatFactor} simulations each for a total of {n * repeatFactor} simulations.")

    for r in range(repeatFactor):
        pairs = random.sample(twoStopStrategies, n)
        print("Running sim round " + str(r+1))
        for i, oneStop in enumerate(oneStopStrategies[:n]):
            twoStop = pairs[i]
            result = simulator.runPairedSimulation(oneStop, twoStop, i * (r+1))
            
            results.append(result)

    return results


if os.path.exists("data/" + place) and not force:
    print(f"Data for {place} already exists. Skipping simulation.")
else:
    print(f"Running simulation for {place} in {year}...")
    runSimulationWithPlace(place, year, simulations)

