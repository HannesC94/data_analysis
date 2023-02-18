from typing import Union, Dict
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import matplotlib
from pandas_profiling import ProfileReport

INPUT_DATA_PATH = Path("/home/hannes/projects/kaggle/data/input/titanic")
OUPUT_DATA_PATH = Path("/home/hannes/projects/kaggle/data/output/titanic")

# Load the data into a Pandas dataframe
def load_data(filepath: Path):
    df = pd.read_csv(filepath)
    return df


def save_data(
    data: Union[pd.DataFrame, pd.Series, Figure, ProfileReport], folder: Path, name: str
):
    # Make directory
    folder.mkdir(parents=True, exist_ok=True)

    # Save data
    if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
        data.to_csv(folder / f"{name}.csv")
    elif isinstance(data, Figure):
        data.savefig(folder / f"{name}.png")
    elif isinstance(data, ProfileReport):
        data.to_file(folder / f"{name}.html")


def save_data_in_dict(data_dict: dict, folder: Path):
    for name, data in data_dict.items():
        save_data(data, folder=folder, name=name)


def get_summary(df: pd.DataFrame):
    df_summary = pd.DataFrame(index=df.columns)
    # Get the data types of each column
    col_types = df.dtypes
    df_summary["col_types"] = col_types

    # Get the counts
    df_summary["n_null"] = df.isna().sum(axis=0)

    df_summary["n_notnull"] = df.notna().sum(axis=0)

    # Get number of unique values
    df_summary["unique"] = df.nunique()

    # Add Information about numeric columns
    numeric_data = df.describe().loc[["mean", "std", "max"], :].transpose()
    # - Add columns with nan values
    df_summary[numeric_data.columns] = np.nan
    df_summary.update(numeric_data)

    return df_summary


def compute_value_counts(
    df: pd.DataFrame, max_unique_values: int, generate_figures: bool = False
):
    # Compute unique values and determine columns to use for value counts
    n_unique = df.nunique()
    columns_to_check = n_unique.index[n_unique < max_unique_values]

    # Compute dfs with value counts
    value_counts = {
        f"value_count_{col}": df[col].value_counts() for col in columns_to_check
    }

    figures = {}
    if generate_figures:
        for df_counts in value_counts.values():
            fig, ax = plt.subplots()
            ax.bar(df_counts.index, df_counts.values)
            ax.set_title(df_counts.name)
            figures[df_counts.name] = fig

    return value_counts, figures


def get_correlation(df: pd.DataFrame):
    """Compute correlation coefficients between numeric variables."""
    pass


def get_relationships():
    """Compute simple groupby relationships."""
    pass


def generate_report(df: pd.DataFrame):
    profile = ProfileReport(df)
    return profile


def generic_summary(input_file_path: Path, output_folder: Path):
    # Load data
    df = load_data(filepath=input_file_path)

    # Save a sample dataframe
    df.sample(10).to_csv(output_folder / "sample.csv")

    # Get sumamry
    df_summary = get_summary(df)
    save_data(df_summary, folder=output_folder, name="summary")

    # Get value counts
    value_counts, figures = compute_value_counts(
        df, max_unique_values=10, generate_figures=True
    )
    save_data_in_dict(value_counts, folder=output_folder / "value_counts")
    save_data_in_dict(figures, folder=output_folder / "value_counts")

    # Create pandas profiling report
    profile = generate_report(df)
    save_data(profile, folder=output_folder, name="pandas_profiling_report")


if __name__ == "__main__":
    generic_summary(INPUT_DATA_PATH / "train.csv", OUPUT_DATA_PATH)
