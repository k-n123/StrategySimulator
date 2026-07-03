import fastf1
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

class Model:
    def __init__(self, year, place):

        '''Initializes the model with the specified year and place, and loads the race session.
        Args:
            year (int): Year of the race
            place (str): Location of the race
        '''

        self.year = year
        self.place = place
        self.session = fastf1.get_session(year, place, 'R')
        self.session.load()
        self.compoundBounds = {}

    def getCompoundBounds(self):
        '''
        Retrieves the bounds for a specific tyre compound based on the lap data.
        Args:
            compound (str): The tyre compound for which to retrieve the bounds
        Returns:
            bounds (dict): A dictionary containing the minimum and maximum tyre life values for all compounds
        '''
        lapsByCompound = self.sortLapsByCompound(self.getLaps())
        compounds = list(lapsByCompound.keys())

        for compound in compounds:
            X, _ = self.getTrainingData(compound)
            minLife = int(np.ceil(X.min()))
            maxLife = int(np.floor(X.max()))
            
            self.compoundBounds[compound] = (minLife, maxLife)
        
        return self.compoundBounds
    
    def getRaceLength(self):
        '''Retrieves the length of the race in laps.
        Args:
            None
        Returns:
            raceLength (int): The length of the race in laps
        '''
        return len(self.session.laps['LapNumber'].unique())

    def getLaps(self):

        '''Retrieves the lap data from that race in a modified pd dataframe.
        
        Args:
            None
        Returns:
            laps (pd.DataFrame): A dataframe containing the driver, lap number, lap time, compound, and tyre life for each lap
            
        '''

        laps = self.session.laps[['Driver', 'LapNumber', 'LapTime', 'Compound', 'TyreLife']].copy()
        alteredLapTimes = laps['LapTime'].dt.total_seconds()
        laps['LapTime'] = alteredLapTimes
        laps = laps.dropna(subset=['LapTime', 'TyreLife'])
        return laps
    

    def sortLapsByCompound(self, laps):
        '''
        Sorts lap by compounds
        Args:
            laps (pd.DataFrame): A dataframe containing the driver, lap number, lap time, compound, and tyre life for each lap
        Returns:
            lapsByCompound (dict): A dictionary where the keys are the tyre compounds and the values are tuples of lap times and tyre lives for each compound
        '''
        lapsByCompound = {}
        for compound in laps['Compound'].unique():
            rows = laps[laps['Compound'] == compound]
            lapsByCompound[compound] = (rows['LapTime'].to_numpy(), rows['TyreLife'].to_numpy())
        return lapsByCompound

    def createCompoundModels(self):
        '''
        Creates a dictionary of linear regression models for each tyre compound used in the Grand Prix.

        Args:
            None
        Returns:
            compoundModels (dict): A dictionary where the keys are the tyre compounds and the values are the linear regression models for each compound
        '''

        self.compoundModels = {}
        laps = self.getLaps()
        lapsByCompound = self.sortLapsByCompound(laps)
        for compound, (lapTimes, tyreLives) in lapsByCompound.items():
            model = LinearRegression()
            X = tyreLives.reshape(-1, 1)
            y = lapTimes.reshape(-1, 1)
            model.fit(X, y)
            self.compoundModels[compound] = model

        return self.compoundModels
        
    def getTrainingData(self, compound):
        '''
        Retrieves the training data for a specific tyre compound.
        Args:
            compound (str): The tyre compound for which to retrieve the training data
        Returns:
            X (np.array): The tyre life values for the specified compound
            y (np.array): The lap time values for the specified compound
        '''
        laps = self.getLaps()
        lapsByCompound = self.sortLapsByCompound(laps)
        if compound not in lapsByCompound:
            raise ValueError("Specified compound not found in the lap data.")
        lapTimes, tyreLives = lapsByCompound[compound]
        return tyreLives.reshape(-1, 1), lapTimes.reshape(-1, 1)



    def getAllModels(self):
        '''
        Retrieves the dictonary of linear regression models for each tyre compound used in the Grand Prix
        '''
        return self.compoundModels
    
    def getCompoundModel(self, compound):
        '''
        Retrieves the linear regression model for a specific tyre compound
        Args:
            compound (str): The tyre compound for which to retrieve the model
        Returns:
            model (LinearRegression): The linear regression model for the specified tyre compound
        '''
        return self.compoundModels.get(compound, None)
    
    def getModelResiduals(self):
        '''
        Calculates the residuals for each tyre compound's linear regression model and stores them in a dictionary.
        Args:
            None
        Returns:
            residuals (dict): A dictionary where the keys are the tyre compounds and the values are the residuals for each compound's linear regression model
        '''

        self.residuals = {}

        for compound in self.getAllModels().keys():
            model = self.getCompoundModel(compound)
            X, y = self.getTrainingData(compound)

            y_pred = model.predict(X)
            res = (y - y_pred).ravel()
            res -= res.mean()
            res = np.clip(res, -2, 2)
            self.residuals[compound] = res
    
        return self.residuals