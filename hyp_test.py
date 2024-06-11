import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
import seaborn as sns
import itertools

# Load the entropy.csv file
df = pd.read_csv('entropy.csv')

# Define the composers of interest
composers_of_interest = ['Bach', 'Beethoven', 'Chopin', 'Liszt']

# Define the entropy metrics to be analyzed
entropy_metrics = ['entropy_pitch', 'entropy_interval', 'entropy_chord', 'entropy_time_interval']

# Initialize a list to store the test results
test_results = []

# Conduct hypothesis tests for all pairs of composers for all entropy metrics
for metric in entropy_metrics:
    for composer1, composer2 in itertools.combinations(composers_of_interest, 2):
        # Filter the data for the two composers
        group1 = df[df['canonical_composer'].str.contains(composer1, case=False, na=False)][metric].dropna()
        group2 = df[df['canonical_composer'].str.contains(composer2, case=False, na=False)][metric].dropna()
        
        # Perform the two-sample t-test
        t_stat, p_value = ttest_ind(group1, group2, equal_var=False)
        
        # Determine if the null hypothesis should be rejected at the 95% confidence level
        reject_null = p_value < 0.05
        
        # Store the result
        test_results.append({
            'entropy_metric': metric.replace('entropy_', ''),
            'composer1': composer1,
            'composer2': composer2,
            't_statistic': t_stat,
            'p_value': p_value,
            'can_divide': reject_null
        })

# Convert the test results to a DataFrame
test_results_df = pd.DataFrame(test_results)

# Display the test results
print("Hypothesis Test Results:")
print(test_results_df)

# Save the test results to a CSV file
test_results_df.to_csv('hypothesis_test_results.csv', index=False)

# For better readability in the console
pd.set_option('display.float_format', '{:.6f}'.format)
print("\nDetailed Hypothesis Test Results with P-values and Null Hypothesis Rejection:")
print(test_results_df.to_string(index=False))