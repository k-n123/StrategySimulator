import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import fastf1

# Load all event names
season = fastf1.get_event_schedule(2025, include_testing=False)['EventName'].tolist()

# Mean Difference, Standard Deviation, Effect Size, Lower Bound, Upper Bound, Confidence, P-value, Test Statistic, DF, 



for place in season:
    # Save a histogram 
    df = pd.read_csv("data/" + place)
    differences = df['Difference']
    hist = plt.hist(differences, bins=20, edgecolor='black')
    plt.title(f"Distribution of Time Differences for {place}")
    plt.xlabel("Time Difference (Two Stop Time - One Stop Time) in Seconds")
    plt.ylabel("Frequency")
    plt.grid(axis='y', alpha=0.75)
    plt.savefig("plots/" + place + "_histogram.png")
    plt.clf()

    print(f"Mean Difference for {place}: {differences.mean()} seconds")
    print(f"Standard Deviation for {place}: {differences.std()} seconds")
    print(f"Effect Size (Cohen's d) for {place}: {differences.mean() / differences.std()}")

    # Perform one-sample t-test against a mean of 0
    t_statistic, p_value = stats.ttest_1samp(differences, 0)
    print(f"One-sample t-test for {place}: t-statistic = {t_statistic}, p-value = {p_value}")

    # Calculate confidence interval
    confidence_level = 1-(0.05/24)
    degrees_freedom = len(differences) - 1
    sample_mean = differences.mean()
    sample_std = differences.std()
    confidence_interval = stats.t.interval(confidence_level, degrees_freedom, loc=sample_mean, scale=sample_std/np.sqrt(len(differences)))
    print(f"{int(confidence_level*100)}% confidence interval for the mean difference in {place}: ({confidence_interval[0]:.2f}, {confidence_interval[1]:.2f})\n")



