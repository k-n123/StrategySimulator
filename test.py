import fastf1
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
session = fastf1.get_event_schedule(2025, include_testing=False)

print(session['EventName'])


session = fastf1.get_session(2024, 'Austrian Grand Prix', 'R')
session.load()
laps = session.laps
print(laps)