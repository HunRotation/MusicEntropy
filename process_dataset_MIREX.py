import pandas as pd
import os
from datetime import datetime
from collections import Counter

# Define a custom function to process the Size column
def process_size(size_str):
    return int(size_str.replace('KB', '').replace(',', '').strip())

# Define a custom function to process the Last Modified column
def process_last_modified(date_str):
    return datetime.strptime(date_str.strip(), '%d-%m-%Y')

# Define a custom function to process the Genre column
def process_genre(genre_str):
    if isinstance(genre_str, str):
        return [genre.strip().lower() for genre in genre_str.split(',')]
    return []

# Read the CSV file
df = pd.read_csv('MIREX-like_mood/dataset/MIREX/dataset info.csv', sep=';', skip_blank_lines=True, encoding='utf-16le')

# Remove leading and trailing whitespaces from column names
df.columns = df.columns.str.strip()

# Drop the last empty column if it exists
if 'Unnamed' in df.columns[-1]:
    df = df.iloc[:, :-1]

# Drop rows without genre information
df = df.dropna(subset=['Genre'])

# Process columns
df['Track'] = df['Track'].fillna(0).astype(int)
df['Year'] = df['Year'].fillna(0).astype(int)
df['Length'] = df['Length'].astype(float)
df['Size'] = df['Size'].apply(process_size)
df['Last Modified'] = df['Last Modified'].apply(process_last_modified)
df['Genre'] = df['Genre'].apply(process_genre)

# Directory containing the .mid files
midi_dir = 'MIREX-like_mood/dataset/MIREX/MIDIs/'

# Function to check if the corresponding .mid file exists
def midi_file_exists(filename):
    midi_filename = filename.replace('.mp3', '.mid')
    return os.path.exists(os.path.join(midi_dir, midi_filename))

# Drop rows where the corresponding .mid file does not exist
df = df[df['Filename'].apply(midi_file_exists)]

# Display the DataFrame
print(df)

# Count the number of songs for each genre word
genre_counter = Counter()
rock_count = 0
pop_count = 0
soul_funk_rnb_count = 0

for genres in df['Genre']:
    genre_counter.update(genres)
    if any('rock' in genre for genre in genres):
        rock_count += 1
    if any('pop' in genre for genre in genres):
        pop_count += 1
    if any(any(sub in genre for sub in ['rhythm and blues', 'soul', 'funk', 'r&b']) for genre in genres):
        soul_funk_rnb_count += 1

# Create the result dictionary
result = dict(genre_counter)
result['total'] = len(df)

# Print the dictionary in descending order of the count
for genre, count in sorted(result.items(), key=lambda item: item[1], reverse=True):
    print(f"{genre}: {count}")

# Print the additional counts
print(f"Number of songs with 'rock' genre: {rock_count}")
print(f"Number of songs with 'pop' genre: {pop_count}")
print(f"Number of songs with 'soul', 'funk', or 'r&b' genre: {soul_funk_rnb_count}")