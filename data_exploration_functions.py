"""
Step-by-step functions to understand data loading and processing for views_pipeline_core.
Run each function individually to see what happens.
"""
import pandas as pd
import numpy as np
from views_pipeline_core.data.handlers import PGMDataset


def step_1_load_raw_data():
    """Step 1: Load the raw parquet file and examine its structure."""
    print("=== STEP 1: Loading Raw Data ===")

    df = pd.read_parquet('hack_data/preds_001.parquet')

    print(f"Data shape: {df.shape}")
    print(f"Index names: {df.index.names}")
    print(f"Columns: {list(df.columns)}")

    print("\nFirst few rows:")
    print(df.head())

    print("\nColumn types:")
    for col in df.columns:
        print(f"  {col}: {df[col].dtype}")

    return df


def step_2_separate_predictions_and_metadata(df):
    """Step 2: Separate prediction columns from metadata columns."""
    print("\n=== STEP 2: Separating Predictions from Metadata ===")

    # Identify prediction columns (start with 'pred_')
    pred_cols = [col for col in df.columns if col.startswith('pred_')]
    meta_cols = [col for col in df.columns if not col.startswith('pred_')]

    print(f"Prediction columns: {pred_cols}")
    print(f"Metadata columns: {meta_cols}")

    # Split the dataframe
    predictions_df = df[pred_cols]
    metadata_df = df[meta_cols]

    print(f"\nPredictions shape: {predictions_df.shape}")
    print(f"Metadata shape: {metadata_df.shape}")

    print("\nPredictions sample:")
    print(predictions_df.head())

    print("\nMetadata sample:")
    print(metadata_df.head())

    return predictions_df, metadata_df


def step_3_examine_prediction_arrays(predictions_df):
    """Step 3: Look at the structure of prediction arrays."""
    print("\n=== STEP 3: Examining Prediction Arrays ===")

    # Take the first prediction
    first_pred = predictions_df.iloc[0]

    for col in predictions_df.columns:
        array_data = first_pred[col]
        print(f"\n{col}:")
        print(f"  Type: {type(array_data)}")
        print(f"  Shape: {np.array(array_data).shape}")
        print(f"  First 5 values: {array_data[:5]}")
        print(f"  All unique values: {np.unique(array_data)}")

    # Show index of this first prediction
    print(f"\nThis prediction is for:")
    print(f"  Month ID: {predictions_df.index[0][0]}")
    print(f"  Grid ID: {predictions_df.index[0][1]}")

    return first_pred


def step_4_create_pgm_dataset(predictions_df):
    """Step 4: Create PGMDataset from predictions-only dataframe."""
    print("\n=== STEP 4: Creating PGMDataset ===")

    print("Creating PGMDataset from predictions-only data...")

    try:
        dataset = PGMDataset(source=predictions_df)
        print("✓ Successfully created PGMDataset!")

        print(f"Dataset properties:")
        print(f"  Number of entities (grid cells): {dataset.num_entities}")
        print(f"  Number of time steps: {dataset.num_time_steps}")
        print(f"  Number of features: {dataset.num_features}")
        print(f"  Prediction variables: {dataset.pred_vars}")
        print(f"  Is prediction dataset: {dataset.is_prediction}")

        return dataset

    except Exception as e:
        print(f"✗ Failed to create PGMDataset: {e}")
        return None


def step_5_calculate_map_estimates(dataset, sample_size=100):
    """Step 5: Calculate MAP estimates using views_pipeline_core."""
    print(f"\n=== STEP 5: Calculating MAP Estimates (sample size: {sample_size}) ===")

    if dataset is None:
        print("No dataset available!")
        return None

    # For demonstration, let's work with a smaller subset
    print("Working with a small subset for demonstration...")

    # Get the original dataframe and take a subset
    original_data = dataset.dataframe
    small_subset = original_data.head(sample_size)

    print(f"Subset shape: {small_subset.shape}")

    # Create a new dataset from the subset
    small_dataset = PGMDataset(source=small_subset)

    print("Calculating MAP estimates...")
    map_estimates = small_dataset.calculate_map()

    print(f"MAP estimates shape: {map_estimates.shape}")
    print(f"MAP estimates columns: {list(map_estimates.columns)}")

    print("\nSample MAP estimates:")
    print(map_estimates.head())

    print("\nComparing original predictions vs MAP estimates:")
    for i in range(min(3, len(small_subset))):
        idx = small_subset.index[i]
        print(f"\nGrid {idx[1]}, Month {idx[0]}:")

        for pred_col in small_dataset.pred_vars:
            original_array = small_subset.loc[idx, pred_col]
            map_col = f"{pred_col}_map"
            map_value = map_estimates.loc[idx, map_col]

            print(f"  {pred_col}:")
            print(f"    Original array first 5: {original_array[:5]}")
            print(f"    MAP estimate: {map_value}")

    return map_estimates


def step_6_calculate_confidence_intervals(dataset, sample_size=100):
    """Step 6: Calculate confidence intervals using views_pipeline_core."""
    print(f"\n=== STEP 6: Calculating Confidence Intervals (sample size: {sample_size}) ===")

    if dataset is None:
        print("No dataset available!")
        return None

    # Work with small subset again
    original_data = dataset.dataframe
    small_subset = original_data.head(sample_size)
    small_dataset = PGMDataset(source=small_subset)

    confidence_levels = [0.5, 0.9, 0.99]
    hdi_results = {}

    for alpha in confidence_levels:
        print(f"\nCalculating {int(alpha*100)}% HDI...")
        hdi = small_dataset.calculate_hdi(alpha=alpha)
        hdi_results[alpha] = hdi

        print(f"HDI shape: {hdi.shape}")
        print(f"HDI columns: {list(hdi.columns)}")
        print(f"Sample HDI values:")
        print(hdi.head(2))

    print("\n=== Comparing Different Confidence Levels ===")
    # Show how confidence intervals get wider as confidence increases
    sample_idx = small_subset.index[0]
    pred_var = small_dataset.pred_vars[0]

    print(f"For grid {sample_idx[1]}, month {sample_idx[0]}, {pred_var}:")
    for alpha in confidence_levels:
        lower_col = f"{pred_var}_hdi_lower"
        upper_col = f"{pred_var}_hdi_upper"

        lower = hdi_results[alpha].loc[sample_idx, lower_col]
        upper = hdi_results[alpha].loc[sample_idx, upper_col]
        width = upper - lower

        print(f"  {int(alpha*100)}% confidence: [{lower:.6f}, {upper:.6f}] (width: {width:.6f})")

    return hdi_results


def step_7_understand_the_data_flow():
    """Step 7: Put it all together - understand the complete data flow."""
    print("\n=== STEP 7: Complete Data Flow Summary ===")

    print("Here's what happens in the complete data processing pipeline:")
    print()
    print("1. RAW DATA (preds_001.parquet)")
    print("   - Contains prediction arrays + metadata")
    print("   - Each prediction array has 32 values (32-month forecast)")
    print("   - Index: (month_id, priogrid_id)")
    print()
    print("2. SEPARATION")
    print("   - Split into predictions_df (pred_* columns) and metadata_df (other columns)")
    print("   - predictions_df is what PGMDataset expects")
    print()
    print("3. PGMDataset CREATION")
    print("   - PGMDataset(source=predictions_df)")
    print("   - Validates format and enables VIEWS-specific calculations")
    print()
    print("4. STATISTICAL CALCULATIONS")
    print("   - calculate_map(): Maximum A Posteriori estimates")
    print("   - calculate_hdi(alpha): Confidence intervals at different levels")
    print("   - These use the distribution information in the prediction arrays")
    print()
    print("5. FINAL RESULT")
    print("   - MAP estimates: single 'best' value for each prediction")
    print("   - HDI bounds: uncertainty ranges around the estimates")
    print("   - Can be combined with metadata for complete API responses")


# Helper function to run all steps
def run_all_steps():
    """Run all steps in sequence."""
    print("Running complete data loading and processing demonstration...\n")

    # Step 1
    df = step_1_load_raw_data()

    # Step 2
    predictions_df, metadata_df = step_2_separate_predictions_and_metadata(df)

    # Step 3
    first_pred = step_3_examine_prediction_arrays(predictions_df)

    # Step 4
    dataset = step_4_create_pgm_dataset(predictions_df)

    # Step 5 (with small sample for speed)
    map_estimates = step_5_calculate_map_estimates(dataset, sample_size=10)

    # Step 6 (with small sample for speed)
    hdi_results = step_6_calculate_confidence_intervals(dataset, sample_size=10)

    # Step 7
    step_7_understand_the_data_flow()

    print("\n" + "="*50)
    print("DEMONSTRATION COMPLETE!")
    print("="*50)

    return {
        'raw_data': df,
        'predictions': predictions_df,
        'metadata': metadata_df,
        'dataset': dataset,
        'map_estimates': map_estimates,
        'hdi_results': hdi_results
    }


if __name__ == "__main__":
  # run step one, two, three and four for quick check
  df = step_1_load_raw_data()
  predictions_df, metadata_df = step_2_separate_predictions_and_metadata(df)
  first_pred = step_3_examine_prediction_arrays(predictions_df)
  dataset = step_4_create_pgm_dataset(predictions_df)
  # Add step 5 for MAP estimates calculation
  map_estimates = step_5_calculate_map_estimates(dataset, sample_size=10)