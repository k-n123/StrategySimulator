from model import Model
from simulation import Simulator
from strategy import StrategyCreator

import fastf1
from datetime import timedelta
import numpy as np
import math
import pandas as pd

import matplotlib.pyplot as plt
import random
import os

if not os.path.exists('data/cache'):
    os.makedirs('data/cache')
fastf1.Cache.enable_cache('data/cache')  # Enable caching to speed up data retrieval

def secondsToTime(seconds):
    delta= timedelta(seconds=seconds)
    return str(delta)

def formatResults(result):
    return [secondsToTime(result[0]), secondsToTime(result[1]), result[2]]


def runSampledMontecarlo(oneStopStrategies, twoStopStrategies, simulator):
    n = min(len(oneStopStrategies), len(twoStopStrategies))
    print(f"Total one stop strategies: {len(oneStopStrategies)}, Total two stop strategies: {len(twoStopStrategies)}. Sampling {n} strategies from each for the Monte Carlo simulation.")
    results = []
    repeatFactor = math.ceil(10000 / n)
    print(f"Running {n} strategy pairs with {repeatFactor} simulations each for a total of {n * repeatFactor} simulations.")

    for r in range(repeatFactor):
        pairs = random.sample(twoStopStrategies, n)
        print("Running sim round " + str(r+1))
        for i, oneStop in enumerate(oneStopStrategies[:n]):
            twoStop = pairs[i]
            result = simulator.runPairedSimulation(oneStop, twoStop, i * (r+1))
            
            results.append(result)

    return results

def saveResults(results, model):
    df = pd.DataFrame(results, columns=['One Stop Time', 'Two Stop Time', 'Difference'])
    df.to_csv("data/" + model.place, index=False)


def runSimulationWithPlace(place):
    model = Model(2024, place)
    model.getLaps()
    if model.getLaps() is None:
        print(f"No lap data available for {place}. Skipping simulation.")
        return
    model.createCompoundModels()

    simulator = Simulator(model)

    strategyCreator = StrategyCreator(model)
    oneStopStrategies, twoStopStrategies = strategyCreator.createAllPossibleStrategies()

    results = runSampledMontecarlo(oneStopStrategies, twoStopStrategies, simulator)
    saveResults(results, model)



season = fastf1.get_event_schedule(2025, include_testing=False)['EventName'].tolist()
for place in season:
    if os.path.exists("data/" + place):
        print(f"Data for {place} already exists. Skipping simulation.")
        continue
    print(f"Running simulation for {place}...")
    runSimulationWithPlace(place)

