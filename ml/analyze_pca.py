import pandas as pd
import joblib
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def analyze_pca_components():
    """
    Loads the saved PCA and Scaler objects to show which original
    features are most important for each principal component.
    """
    try:
        # Load the fitted preprocessors
        scaler_path = os.path.join(SCRIPT_DIR, 'scaler.pkl')
        scaler = joblib.load(scaler_path)

        pca_path = os.path.join(SCRIPT_DIR, 'pca.pkl')
        pca = joblib.load(pca_path)
    except FileNotFoundError as e:
        print(f"Error: Could not find preprocessor files. Run the training script first. Details: {e}")
        return

    # Get the original feature names from the scaler
    original_features = scaler.feature_names_in_

    # Create a DataFrame to view the makeup of each Principal Component
    # The `pca.components_` attribute holds the weight of each original feature
    # for each new component.
    components_df = pd.DataFrame(
        pca.components_,
        columns=original_features,
        index=[f'PC-{i+1}' for i in range(pca.n_components_)]
    )

    print("--- Principal Component Composition ---")
    print("Each value is the 'weight' or 'loading' of an original feature on the new component.")
    print("A large absolute value means the feature is influential for that component.\n")
    print(components_df)

    print("\n--- Explained Variance Ratio ---")
    print("This shows the percentage of the total information (variance) each component captures.")
    for i, ratio in enumerate(pca.explained_variance_ratio_):
        print(f"  PC-{i+1}: {ratio:.2%}")

    print(f"\nTotal Variance Explained by {pca.n_components_} components: {sum(pca.explained_variance_ratio_):.2%}")


if __name__ == '__main__':
    analyze_pca_components()