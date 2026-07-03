import fastf1

from model import Model
from sklearn.linear_model import LinearRegression
import numpy as np

class Simulator:
    def __init__(self, model):
        '''
        Initializes the simulator with the specified model.
        Args:
            model (Model): The model to be used for the simulation
        '''
        self.model = model

    # Simulate lap and stint WITHOUT NOISE  
    def simulateLap(self, compound, tyreLife):
        '''
        Simulates a lap time based on the specified tyre compound and tyre life using the linear regression model for that compound.
        Args:
            compound (str): The tyre compound to be used for the simulation
            tyreLife (float): The tyre life to be used for the simulation
        Returns:
            lapTime (float): The simulated lap time for the specified tyre compound and tyre life
        '''
        compound = compound.upper()
        model = self.model.getCompoundModel(compound)
        if model is None:
            raise ValueError("Model for the specified tyre compound not found.")
        X = np.array([[tyreLife]])
        return model.predict(X)[0]
    
    def simulateStint(self, compound, startTyreLife, endTyreLife):
        '''
        Simulates a stint time based on the specified tyre compound and tyre life range using the linear regression model for that compound.
        Args:
            compound (str): The tyre compound to be used for the simulation
            startTyreLife (float): The starting tyre life for the simulation
            endTyreLife (float): The ending tyre life for the simulation
        Returns:
            stintTime (float): The simulated stint time for the specified tyre compound and tyre life range
        '''
        compound = compound.upper()
        model = self.model.getCompoundModel(compound)
        if model is None:
            raise ValueError("Model for the specified tyre compound not found.")
        tyreLives = np.arange(startTyreLife, endTyreLife + 1).reshape(-1, 1)
        lapTimes = model.predict(tyreLives)


        return np.sum(lapTimes)
    
    # Simulate lap and stint WITH NOISE
    def simulateLapWithNoise(self, compound, tyreLife, rng, lapNoise=None):
        '''
        Simulates a lap time with added noise based on the specified tyre compound, tyre life, and noise standard deviation.
        Args:
            compound (str): The tyre compound to be used for the simulation
            tyreLife (float): The tyre life to be used for the simulation
            rng (np.random.Generator): The random number generator to be used for the simulation
            lapNoise (float): The standard deviation of the noise to be added to the simulated lap time
        Returns:
            lapTime (float): The simulated lap time with added noise for the specified tyre compound and tyre life
        '''

        compound = compound.upper()
        model = self.model.getCompoundModel(compound)
        if model is None:
            raise ValueError("Model for the specified tyre compound not found.")
        
        bounds = self.model.getCompoundBounds()[compound]
        maxLife = bounds[1]
        if tyreLife > maxLife:
            raise ValueError(f"Tyre life {tyreLife} exceeds maximum life {maxLife} for compound {compound} based on training data.")

        X = np.array([[tyreLife]])
        meanLapTime = model.predict(X)[0]

        if lapNoise is not None:
            noise = lapNoise
        else:
            noise = rng.choice(self.model.getModelResiduals()[compound])

        assert tyreLife <= bounds[1], (
            f"Extrapolation: {compound} tyreLife={tyreLife} > max={bounds[1]}"
        )

        return meanLapTime + noise

    def simulateStintWithNoise(self, compound, startTyreLife, endTyreLife, rng, stintNoise=None):
        '''
        Simulates stint time with added noise based on compound, tyre life, and provided noise
        Args:
            compound (str): The tyre compound to be used for the simulation
            startTyreLife (float): The starting tyre life for the simulation
            endTyreLife (float): The ending tyre life for the simulation
            rng (np.random.Generator): The random number generator to be used for the simulation
            stintNoise (float): The standard deviation of the noise to be added to the simulated lap times
        Returns:
            stintTime (float): The simulated stint time with added noise for the specified tyre compound and tyre life range
        '''
        bounds = self.model.getCompoundBounds()[compound]
        maxLife = bounds[1]
        
        stintTime = 0.0
        if stintNoise is None:
            stintNoise = [0.0] * (endTyreLife - startTyreLife)  # fallback

        for i, lapNoise in enumerate(stintNoise):
            tyreLife = startTyreLife + i
            if tyreLife > maxLife:
                break

            lapTime = self.simulateLapWithNoise(compound, tyreLife, rng, lapNoise)
            stintTime += lapTime

        return stintTime

    # Simulate One and Two Stop Races WITHOUT NOISE
    def simulateOneStopRace(self, compound1, compound2, stint1TyreLife, stint2TyreLife):
        '''
        Simulates a one-stop race with the specified tyre compounds and stint tyre life ranges.
        Args:
            compound1 (str): The tyre compound for the first stint
            compound2 (str): The tyre compound for the second stint
            stint1TyreLife (float): The tyre life for the first stint
            stint2TyreLife (float): The tyre life for the second stint
        Returns:
            raceTime (float): The simulated race time for the one-stop strategy
        '''
        stint1Time = self.simulateStint(compound1, 0, stint1TyreLife)
        stint2Time = self.simulateStint(compound2, 0, stint2TyreLife)
        return stint1Time + stint2Time
    
    def simulateTwoStopRace(self, compound1, compound2, compound3, stint1TyreLife, stint2TyreLife, stint3TyreLife):
        '''
        Simulates a two-stop race with the specified tyre compounds and stint tyre life ranges. 
        Args:
            compound1 (str): The tyre compound for the first stint
            compound2 (str): The tyre compound for the second stint
            compound3 (str): The tyre compound for the third stint
            stint1TyreLife (float): The tyre life for the first stint
            stint2TyreLife (float): The tyre life for the second stint
            stint3TyreLife (float): The tyre life for the third stint
        Returns:
            raceTime (float): The simulated race time for the two-stop strategy
        '''
        stint1Time = self.simulateStint(compound1, 0, stint1TyreLife)
        stint2Time = self.simulateStint(compound2, 0, stint2TyreLife)
        stint3Time = self.simulateStint(compound3, 0, stint3TyreLife)
        return stint1Time + stint2Time + stint3Time
    
    # Simulate One and Two stop races with NOISE
    def simulateOneStopRaceWithNoise(self, compound1, compound2, stint1tyreLife, stint2TyreLife, rng, raceNoise, pitLosses):
        '''
        Simulates a one stop race with the random race noise added to the stint times
        Args:
            compound1 (str): The tyre compound for the first stint
            compound2 (str): The tyre compound for the second stint
            stint1TyreLife (float): The tyre life for the first stint
            stint2TyreLife (float): The tyre life for the second stint
            rng (np.random.Generator): The random number generator to be used for the simulation
            raceNoise (np.array): An array of random noise values for the entire race
            pitLosses (np.array): An array of pit stop losses for the race
        Returns:
            raceTime (float): The simulated race time for the one-stop strategy with added noise
        '''

        n1 = stint1tyreLife
        n2 = stint2TyreLife

        noise1 = raceNoise[:n1]
        noise2 = raceNoise[n1:n1+n2]

        pitLoss1 = pitLosses[0] if len(pitLosses) > 0 else 0.0

        stint1Time = self.simulateStintWithNoise(compound1, 0, stint1tyreLife, rng, stintNoise=noise1)
        stint2Time = self.simulateStintWithNoise(compound2, 0, stint2TyreLife, rng, stintNoise=noise2)
        return stint1Time + stint2Time + pitLoss1
    
    def simulateTwoStopRaceWithNoise(self, compound1, compound2, compound3, stint1tyreLife, stint2TyreLife, stint3TyreLife, rng, raceNoise, pitLosses):
        '''
        Simulates a two stop race with the random race noise added to the stint times
        Args:
            compound1 (str): The tyre compound for the first stint
            compound2 (str): The tyre compound for the second stint
            compound3 (str): The tyre compound for the third stint
            stint1TyreLife (float): The tyre life for the first stint
            stint2TyreLife (float): The tyre life for the second stint
            stint3TyreLife (float): The tyre life for the third stint
            rng (np.random.Generator): The random number generator to be used for the simulation
            raceNoise (np.array): An array of random noise values for the entire race
            pitLosses (np.array): An array of pit stop losses for the race
        Returns:
            raceTime (float): The simulated race time for the two-stop strategy with added noise
        '''
        n1 = stint1tyreLife
        n2 = stint2TyreLife
        n3 = stint3TyreLife

        i = 0
        noise1 = raceNoise[i:i+n1]; i += n1
        noise2 = raceNoise[i:i+n2]; i += n2
        noise3 = raceNoise[i:i+n3]

        pitLoss1 = pitLosses[0] if len(pitLosses) > 0 else 0.0
        pitLoss2 = pitLosses[1] if len(pitLosses) > 1 else 0.0

        stint1Time = self.simulateStintWithNoise(compound1, 1, stint1tyreLife, rng, stintNoise=noise1)
        stint2Time = self.simulateStintWithNoise(compound2, 1, stint2TyreLife, rng, stintNoise=noise2)
        stint3Time = self.simulateStintWithNoise(compound3, 1, stint3TyreLife - 1, rng, stintNoise=noise3)
        return stint1Time + stint2Time + stint3Time + pitLoss1 + pitLoss2

    # Run a singular paired simulation of one and two stop strategies with the same random noise for both strategies
    def runPairedSimulation(self, strategy1, strategy2, seed):
        '''
        Runs a paired simulation of two strategies with the same random noise for both strategies.
        Args:
            strategy1 (dict): A dictionary containing the stint compounds for the first strategy
            strategy2 (dict): A dictionary containing the stint compounds for the second strategy
            seed (int): The seed for the random number generator
        Returns:
            array: An array containing the simulated race times for both strategies and the delta of (one stop - two stop)
        '''
        rng = np.random.default_rng(seed=seed)
        raceNoise = self.generateRaceNoise(numLaps=self.model.getRaceLength(), rng=rng)  
        pitLosses = self.calculatePitGain(numStops=2, rng=rng)  # Assuming a maximum of 2 pit stops for the two-stop strategy

        

        raceTime1 = self.simulateOneStopRaceWithNoise(strategy1['compound1'], strategy1['compound2'], strategy1['stint1'], strategy1['stint2'], rng, raceNoise, pitLosses)
        raceTime2 = self.simulateTwoStopRaceWithNoise(strategy2['compound1'], strategy2['compound2'], strategy2['compound3'], strategy2['stint1'], strategy2['stint2'], strategy2['stint3'], rng, raceNoise, pitLosses)

        return [float(raceTime1), float(raceTime2), float(raceTime1 - raceTime2)]

    def calculatePitGain(self, numStops, rng):
        '''
        Calculates the pit stop time gain for the specified Grand Prix based on historical data.
        Args:
            GPname (str): The name of the Grand Prix for which to calculate the pit stop time gain
        Returns:
            pitGain (float): The calculated pit stop time gain for the specified Grand Prix
        '''
        self.pit_mean = 22.0
        self.pit_std = 0.7

        return rng.normal(self.pit_mean, self.pit_std, size=numStops)
    
    def generateRaceNoise(self, numLaps, rng):
        '''Generates random noise for an entire race distance
        Args:
            numLaps (int): The number of laps in the race
            rng (np.random.Generator): The random number generator to be used for the simulation
        Returns:
            race_noise (np.array): An array of random noise values for the entire race distance
        '''
        all_residuals = np.concatenate(list(self.model.getModelResiduals().values()))

        race_noise = rng.choice(all_residuals, size=numLaps)
        return race_noise
