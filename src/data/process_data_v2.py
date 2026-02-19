import pandas as pd
import numpy as np
import glob
import os
import sys
from sklearn.preprocessing import LabelEncoder

# Add utils to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
sys.path.append(os.path.join(project_root, 'src', 'utils'))

# Assuming ConcatenarCSV is in utils/funciones.py, but we can just implement simple reading here since we are refactoring.
# from funciones import ConcatenarCSV

def load_and_combine_data(raw_data_path):
    """Loads and combines all CSV files from the raw data directory."""
    all_files = glob.glob(os.path.join(raw_data_path, "*.csv"))
    print(f"Found {len(all_files)} files in {raw_data_path}")
    
    df_list = []
    for filename in all_files:
        df = pd.read_csv(filename, index_col=None, header=0)
        df_list.append(df)
    
    if not df_list:
        raise ValueError("No CSV files found in raw data directory.")
        
    combined_df = pd.concat(df_list, axis=0, ignore_index=True)
    return combined_df

def clean_and_encode_data(df):
    """Cleans the dataframe and encodes categorical variables."""
    
    # Drop columns that are not useful for prediction or have too many nulls
    cols_to_drop = [
        'score', 'tourney_name', 'winner_name', 'loser_name', 'minutes',
        'l_1stIn', 'l_1stWon', 'l_2ndWon', 'l_ace', 'l_svpt', 'l_SvGms', 'l_bpFaced', 'l_df', 'l_bpSaved',
        'w_1stIn', 'w_1stWon', 'w_2ndWon', 'w_SvGms', 'w_ace', 'w_svpt', 'w_bpFaced', 'w_bpSaved', 'w_df',
        'winner_seed', 'winner_entry', 'loser_seed', 'loser_entry'
    ]
    
    # Drop if they exist
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    
    # Drop rows with any null values
    original_len = len(df)
    df = df.dropna()
    print(f"Dropped {original_len - len(df)} rows containing null values. Remaining: {len(df)}")
    
    # Encoding Categorical Variables
    # We must ensure consistent encoding for winner/loser shared features (hand, ioc)
    
    # Pairs of columns that share the same domain
    shared_features = [('winner_hand', 'loser_hand'), ('winner_ioc', 'loser_ioc')]
    
    for col1, col2 in shared_features:
        if col1 in df.columns and col2 in df.columns:
            le = LabelEncoder()
            # Fit on all unique values from both columns
            all_values = pd.concat([df[col1], df[col2]]).unique()
            le.fit(all_values)
            df[col1] = le.transform(df[col1])
            df[col2] = le.transform(df[col2])
            print(f"Encoded {col1} and {col2}")
            
    # Other categorical columns
    categorical_cols = ['tourney_id', 'surface', 'tourney_level', 'round']
    
    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            print(f"Encoded {col}")

    return df

def restructure_player_data(df):
    """
    Restructures data from 'winner'/'loser' format to 'player_1'/'player_2' format
    with a binary target 'player_1_wins' (1 if player_1 won, 0 if player_2 won).
    Randomly swaps players to balance the classes.
    """
    
    # Identify winner and loser columns
    winner_cols = [c for c in df.columns if c.startswith('winner_')]
    loser_cols = [c for c in df.columns if c.startswith('loser_')]
    
    # Common columns (match info)
    common_cols = [c for c in df.columns if c not in winner_cols and c not in loser_cols]
    
    print("Restructuring data... this might take a moment.")
    
    # Create a random mask for swapping (True means swap, False means keep)
    mask = np.random.rand(len(df)) > 0.5
    
    # Initialize new DataFrames for p1 and p2 features
    df_p1 = pd.DataFrame(index=df.index)
    df_p2 = pd.DataFrame(index=df.index)
    
    # Process features
    base_features = set([c.replace('winner_', '').replace('loser_', '') for c in winner_cols + loser_cols])
    
    for base in base_features:
        w_col = f"winner_{base}"
        l_col = f"loser_{base}"
        
        if w_col in df.columns and l_col in df.columns:
            # If mask is True: p1 gets loser data, p2 gets winner data (Target=0)
            # If mask is False: p1 gets winner data, p2 gets loser data (Target=1)
            
            df_p1[f"p1_{base}"] = np.where(mask, df[l_col], df[w_col])
            df_p2[f"p2_{base}"] = np.where(mask, df[w_col], df[l_col])
    
    # Combine everything
    df_common = df[common_cols].copy()
    
    # Target: 0 if swapped (winner is p2), 1 if kept (winner is p1)
    target = np.where(mask, 0, 1)
    
    result_df = pd.concat([df_common, df_p1, df_p2], axis=1)
    result_df['target'] = target
    
    print(f"Data restructured. Target distribution: {result_df['target'].value_counts(normalize=True).to_dict()}")
    
    return result_df

def main():
    raw_path = os.path.join(project_root, 'src', 'data', 'raw')
    processed_path = os.path.join(project_root, 'src', 'data', 'processed')
    
    output_file = os.path.join(processed_path, 'df_ready_for_model.csv')
    
    print(f"Reading data from {raw_path}")
    df = load_and_combine_data(raw_path)
    
    print("Cleaning and Encoding data...")
    df = clean_and_encode_data(df)
    
    print("Restructuring data...")
    df_model = restructure_player_data(df)
    
    # Ensure processed directory exists
    os.makedirs(processed_path, exist_ok=True)
    
    print(f"Saving to {output_file}")
    df_model.to_csv(output_file, index=False)
    print("Done!")

if __name__ == "__main__":
    main()

