from collections import Counter
import pandas as pd
import argparse
import numpy as np
import mido
from itertools import combinations
from tqdm import tqdm
from chord_extractor.extractors import Chordino

base_dir = 'dataset/maestro-v3.0.0/'

def filter_composers(file_path, composer_names):
    # Define the data types for the columns
    dtypes = {
        'canonical_composer': 'string',
        'canonical_title': 'string',
        'split': 'string',
        'year': 'int',
        'midi_filename': 'string',
        'audio_filename': 'string',
        'duration': 'float'
    }

    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path, dtype=dtypes)

    # Filter the rows that include any of the given names in 'canonical_composer'
    pattern = '|'.join(composer_names)
    filtered_df = df[df['canonical_composer'].str.contains(pattern, case=False, na=False)]
    
    return filtered_df

# count songs that contains each composer names
def count_songs(filtered_df, composer_names):
    counts = {name: 0 for name in composer_names}
    for name in composer_names:
        counts[name] = filtered_df['canonical_composer'].str.contains(name, case=False, na=False).sum()
    return counts

def entropy(counts):
    total_counts = sum(counts.values())
    probs = [count / total_counts for count in counts.values()]
    return -sum([p * np.log2(p) for p in probs])

# get the essential information dict of the midi file
def get_midi_info(midi_file):
    # Load the MIDI file
    midi = mido.MidiFile(midi_file)
    
    # get the tempo
    for msg in midi:
        if msg.type == 'set_tempo':
            secs_per_beat = msg.tempo / 1000000
    
    # Iterate through all the messages in the MIDI file
    midi_dict = {}
    
    for track in midi.tracks:
        midi_dict[track.name] = []
        cumul_time = 0
        for msg in track:
            cumul_time += msg.time
            if msg.type == 'note_on' and msg.velocity > 0:
                midi_dict[track.name].append({'type': 'note_on', 'time': cumul_time, 'note': msg.note, 'real_time': cumul_time / midi.ticks_per_beat * secs_per_beat})

    return midi_dict


# Calculate the entropy of the pitch distribution in a MIDI file
def calculate_entropy_pitch(midi_dict):
    # Initialize a counter for note pitches
    note_counts = Counter()

    # Iterate through all the messages in the MIDI file
    for track in midi_dict.values():
        for msg in track:            
            note_counts[msg['note']] += 1    

    # Calculate the entropy using the counts
    return entropy(dict(note_counts))

# Calculate the entropy of the interval distribution in a MIDI file
def calculate_entropy_interval(midi_dict):
    intervals = []
    
    for track in midi_dict.values():
        if not track:
            continue

        # Sort by time
        track.sort(key=lambda x: x['time'])

        # Group notes by rounded time
        grouped_notes = {}
        for note in track:
            rounded_time = note['time'] // 12 * 12
            if rounded_time not in grouped_notes:
                grouped_notes[rounded_time] = []
            grouped_notes[rounded_time].append(note['note'])

        # Calculate intervals
        previous_notes = []
        for time in sorted(grouped_notes.keys()):
            current_notes = grouped_notes[time]
            if previous_notes:
                for prev_note in previous_notes:
                    for curr_note in current_notes:
                        intervals.append(curr_note - prev_note)
            previous_notes = current_notes

    # Count the intervals and calculate entropy
    interval_counts = Counter(intervals)
    return entropy(interval_counts)

# Extract the chord from a MIDI file
def extract_chord(midi_path):
    # Extract the chord
    chordino = Chordino()
    conversion_file_path = chordino.preprocess(midi_path)
    chords = chordino.extract(conversion_file_path)

    return [chord.chord for chord in chords if chord.chord != 'N'] 

# Calculate the entropy of the chord distribution in a MIDI file
def calculate_entropy_chord(midi_path):
    chords = extract_chord(midi_path)
    chord_counts = Counter(chords)
    return entropy(chord_counts)

def calculate_entropy(df, verbose=False):
    # Initialize lists to store the results
    canonical_composer = []
    midi_filename = []
    entropy_pitch = []
    entropy_interval = []
    entropy_chord = []
    
    # Total number of MIDI files
    total_files = len(df)

    # Calculate entropy for the remaining rows with a progress bar
    for i in tqdm(range(total_files), desc="Processing MIDI files"):
        canonical_composer.append(df.iloc[i]['canonical_composer'])
        midi_filename.append(df.iloc[i]['midi_filename'])
        midi_file_path = base_dir + df.iloc[i]['midi_filename']
        midi_dict = get_midi_info(midi_file_path)
        entropy_pitch_value = calculate_entropy_pitch(midi_dict)
        if (verbose):
            print(f'{df.iloc[i]["midi_filename"]}, pitch: {entropy_pitch_value}')
        entropy_interval_value = calculate_entropy_interval(midi_dict)
        if (verbose):
            print(f'{df.iloc[i]["midi_filename"]}, interval: {entropy_interval_value}')
        entropy_chord_value = calculate_entropy_chord(midi_file_path)
        if (verbose):
            print(f'{df.iloc[i]["midi_filename"]}, chord: {entropy_chord_value}')
        entropy_pitch.append(entropy_pitch_value)
        entropy_interval.append(entropy_interval_value)
        entropy_chord.append(entropy_chord_value)    
    # Create the new dataframe with the results
    result_df = pd.DataFrame({
        'canonical_composer': canonical_composer,
        'midi_filename': midi_filename,
        'entropy_pitch': entropy_pitch,
        'entropy_interval': entropy_interval
    })
    
    return result_df


'''
if __name__ == '__main__':
    #print(calculate_entropy_pitch(base_dir + '2018/MIDI-Unprocessed_Chamber2_MID--AUDIO_09_R3_2018_wav--1.midi'))
    sample_dict = get_midi_info(base_dir + '2018/MIDI-Unprocessed_Chamber2_MID--AUDIO_09_R3_2018_wav--1.midi')
    #for l in sample_dict.values():
    #    for msgdict in l:
    #        print(msgdict)
    print(calculate_entropy_pitch(sample_dict))
    print(calculate_entropy_interval(sample_dict))
    print(calculate_entropy_chord(base_dir + '2018/MIDI-Unprocessed_Chamber2_MID--AUDIO_09_R3_2018_wav--1.midi'))
'''

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate the entropy of the given composers.')
    parser.add_argument('-c', '--composer_names', nargs='+', help='The names of the composers to filter.')
    parser.add_argument('-v', '--verbose', action='store_true', help='print the calculated entropy values.')
    args = parser.parse_args()

    df = filter_composers(base_dir + 'maestro-v3.0.0.csv', args.composer_names)
    if (args.verbose):
        print(count_songs(df, args.composer_names))

    result_df = calculate_entropy(df, args.verbose)

    # Save the result DataFrame as a CSV file
    result_df.to_csv('entropy.csv', index=False)

    # Display the result DataFrame
    print(result_df)