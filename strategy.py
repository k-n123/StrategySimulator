
from itertools import product
import numpy as np


class StrategyCreator:
    def __init__(self, model):
        self.model = model

        self.maxStintBounds = {
            "SOFT": 29,
            "MEDIUM": 41,
            "HARD": 54,
            "INTERMEDIATE": 44,
            "WET": 12
        }
        self.compoundOrder = {
            "SOFT": 0,
            "MEDIUM": 1,
            "HARD": 2,
            "INTERMEDIATE": 0,
            "WET": 0
        }

        # Use if extra pruning required 
        self.numBuckets = 5
        self.stintBuckets = self.generateStintBuckets()

        
    def generateStintBuckets(self):
        buckets = {}
        for compound in self.model.getCompoundBounds():
            min_life, max_life = self.model.getCompoundBounds()[compound]
            max_stint = max_life + 1  # maximum stint length without extrapolation

            # If max_stint is small, just use full range
            if max_stint <= self.numBuckets:
                buckets[compound] = list(range(1, max_stint+1))
            else:
                # Generate evenly spaced buckets
                buckets[compound] = list(np.linspace(1, max_stint, self.numBuckets, dtype=int))
        return buckets
    
    def getPossibleCompounds(self):
        laps = self.model.getLaps()
        lapsByCompound = self.model.sortLapsByCompound(laps)
        return list(lapsByCompound.keys())
    
    def checkStintBounds(self, compound, stintLength):
        compound = compound.upper()
        _, maxLife = self.model.getCompoundBounds()[compound]
        maxAllowed = self.maxStintBounds.get(compound, maxLife)

        # Maximum safe stint length given training data

        return 1 <= stintLength <= maxAllowed
    
    def isMonotonic(self, compounds):
        return all(
            self.compoundOrder[compounds[i]] <= self.compoundOrder[compounds[i+1]]
            for i in range(len(compounds)-1)
        )

    def respectsStintDurations(self, compounds, stints):
        for i in range(len(compounds)-1):
            if self.compoundOrder[compounds[i]] < self.compoundOrder[compounds[i+1]]:
                if stints[i] > stints[i+1]:
                    return False
        return True

    def createStintLengthPossibilities(self, raceLength, numStints):
        stintLengths = []
        if numStints == 3:
            for i in range(1, raceLength):
                for j in range(1, raceLength - i):
                    k = raceLength - i - j
                    stintLengths.append((i, j, k))
        elif numStints == 2:
            for i in range(1, raceLength):
                j = raceLength - i
                stintLengths.append((i, j))
        return stintLengths

    # This can be used with stint buckets for pruning if needed
    def applyStintBuckets(self, compound, stintLength):
        compound = compound.upper()
        if compound in self.stintBuckets:
            # Find closest bucket
            buckets = self.stintBuckets[compound]
            closest = min(buckets, key=lambda x: abs(x - stintLength))
            return closest
        return stintLength

    def createAllPossibleStrategies(self):
        compounds = self.getPossibleCompounds()
        raceLength = self.model.getRaceLength()
        oneStopStrategies = []
        twoStopStrategies = []

        # 2 Stop Strategies
        for compound_combo in product(compounds, repeat = 3):
            # Prune non-monotonic strategies
            if not self.isMonotonic(compound_combo):
                continue
            # Prune all soft strategies
            if all(c == "SOFT" for c in compound_combo):
                continue
            
            for stint_combo in self.createStintLengthPossibilities(raceLength, 3):
                # Prune invalid stints
                if not all(self.checkStintBounds(c, s) for c, s in zip(compound_combo, stint_combo)):
                    continue
                if not self.respectsStintDurations(compound_combo, stint_combo):
                    continue

                # Apply stint buckets for pruning
                stint_combo_bucketed = tuple(self.applyStintBuckets(c, s) for c, s in zip(compound_combo, stint_combo))

                strategy = {
                    'compound1': compound_combo[0],
                    'compound2': compound_combo[1],
                    'compound3': compound_combo[2],
                    'stint1': stint_combo_bucketed[0],
                    'stint2': stint_combo_bucketed[1],
                    'stint3': stint_combo_bucketed[2]
                }
                twoStopStrategies.append(strategy)
        
        # 1 Stop Strategies
        for compound_combo in product(compounds, repeat=2):
            if not self.isMonotonic(compound_combo):
                continue
            if all(c == "SOFT" for c in compound_combo):
                continue

            for stint_combo in self.createStintLengthPossibilities(raceLength, 2):
                if not all(self.checkStintBounds(c, s) for c, s in zip(compound_combo, stint_combo)):
                    continue
                if not self.respectsStintDurations(compound_combo, stint_combo):
                    continue

                stint_combo_bucketed = tuple(self.applyStintBuckets(c, s) for c, s in zip(compound_combo, stint_combo))

                strategy = {
                    'compound1': compound_combo[0],
                    'compound2': compound_combo[1],
                    'stint1': stint_combo_bucketed[0],
                    'stint2': stint_combo_bucketed[1]
                }
                oneStopStrategies.append(strategy)
        
        return (oneStopStrategies, twoStopStrategies)



