import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the entropy.csv file
df = pd.read_csv('entropy.csv')

# Define the composers of interest
composers_of_interest = ['Bach', 'Beethoven', 'Chopin', 'Liszt']

# Create a new column in the dataframe for each composer of interest
for composer in composers_of_interest:
    df[composer] = df['canonical_composer'].str.contains(composer, case=False, na=False)

# count the number of songs that contains each composer in 'canonical_composer'
for composer in composers_of_interest:
    print(f"Number of songs by {composer}: {df[composer].sum()}")

# Create a long-form dataframe for plotting
plot_data = pd.melt(df, id_vars=['entropy_pitch', 'entropy_interval', 'entropy_chord', 'entropy_time_interval'], value_vars=composers_of_interest, 
                    var_name='composer', value_name='is_present')

# Filter to include only the rows where the composer is present
plot_data = plot_data[plot_data['is_present']]

# Plot the 2D scatter plot for each pair of entropy metrics
entropy_pairs = [
    ('entropy_pitch', 'entropy_interval'),
    ('entropy_pitch', 'entropy_chord'),
    ('entropy_pitch', 'entropy_time_interval'),
    ('entropy_interval', 'entropy_chord'),
    ('entropy_interval', 'entropy_time_interval'),
    ('entropy_chord', 'entropy_time_interval')
]

for x_metric, y_metric in entropy_pairs:
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=plot_data, x=x_metric, y=y_metric, hue='composer', style='composer', palette='tab10')
    plt.title(f'{x_metric} vs {y_metric} by Composer')
    plt.xlabel(x_metric.replace('entropy_', ''))
    plt.ylabel(y_metric.replace('entropy_', ''))
    plt.legend(title='Composer', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.savefig(f'{x_metric}_vs_{y_metric}_scatter_plot.png', bbox_inches='tight')
    plt.clf()

# Calculate mean and standard deviation for each composer group
composer_stats = plot_data.groupby('composer').agg({
    'entropy_pitch': ['mean', 'std'],
    'entropy_interval': ['mean', 'std'],
    'entropy_chord': ['mean', 'std'],
    'entropy_time_interval': ['mean', 'std']
}).reset_index()

composer_stats.columns = [
    'composer',
    'entropy_pitch_mean',
    'entropy_pitch_std',
    'entropy_interval_mean',
    'entropy_interval_std',
    'entropy_chord_mean',
    'entropy_chord_std',
    'entropy_time_interval_mean',
    'entropy_time_interval_std']

print("Composer statistics (mean and standard deviation):")
print(composer_stats)

# Save the entropy statistics to a CSV file
composer_stats.to_csv('entropy_statistics.csv', index=False)

# For each entropy metric, plot the mean and standard deviation for each composer
for entropy_metric in ['entropy_pitch', 'entropy_interval', 'entropy_chord', 'entropy_time_interval']:
    plt.figure(figsize=(10, 6))
    sns.barplot(data=plot_data, x='composer', y=entropy_metric, ci='sd', palette='tab10')
    plt.title(f'{entropy_metric} by Composer')
    plt.xlabel('Composer')
    plt.ylabel(entropy_metric.replace('entropy_', ''))
    plt.xticks(rotation=45)
    plt.savefig(f'barplot_{entropy_metric}.png', bbox_inches='tight')
    plt.clf()

# Calculate the correlation constants between entropy metrics
correlation_matrix = df[['entropy_pitch', 'entropy_interval', 'entropy_chord', 'entropy_time_interval']].corr()

# Rename the columns and index for the heatmap to remove 'entropy_' prefix
correlation_matrix.columns = [col.replace('entropy_', '') for col in correlation_matrix.columns]
correlation_matrix.index = [idx.replace('entropy_', '') for idx in correlation_matrix.index]

print("Correlation matrix:")
print(correlation_matrix)

# Plot the correlation matrix as a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0, linewidths=.5, fmt=".2f")
plt.title('Correlation Matrix of Entropy Metrics')
plt.savefig('entropy_correlation_matrix.png', bbox_inches='tight')
plt.show()