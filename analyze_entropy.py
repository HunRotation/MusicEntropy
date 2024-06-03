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

# Create a long-form dataframe for plotting
plot_data = pd.melt(df, id_vars=['entropy_pitch', 'entropy_interval'], value_vars=composers_of_interest, 
                    var_name='composer', value_name='is_present')

# Filter to include only the rows where the composer is present
plot_data = plot_data[plot_data['is_present']]

# Plot the 2D scatter plot
plt.figure(figsize=(10, 6))

# Use seaborn to create a scatter plot with different colors for each composer group
sns.scatterplot(data=plot_data, x='entropy_pitch', y='entropy_interval', hue='composer', palette='tab10')

plt.title('Entropy Pitch vs Entropy Interval by Composer')
plt.xlabel('Entropy Pitch')
plt.ylabel('Entropy Interval')
plt.legend(title='Composer', bbox_to_anchor=(1.05, 1), loc='upper left')

# Save the plot as an image file
plt.savefig('entropy_scatter_plot.png', bbox_inches='tight')

# Display the plot
plt.show()

# Calculate mean and standard deviation for each composer group
composer_stats = plot_data.groupby('composer').agg({
    'entropy_pitch': ['mean', 'std'],
    'entropy_interval': ['mean', 'std']
}).reset_index()

composer_stats.columns = ['composer', 'entropy_pitch_mean', 'entropy_pitch_std', 'entropy_interval_mean', 'entropy_interval_std']

print("Composer statistics (mean and standard deviation):")
print(composer_stats)

# Save the entropy statistics to a CSV file
composer_stats.to_csv('entropy_statistics.csv', index=False)