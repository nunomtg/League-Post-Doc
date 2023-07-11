from typing import Dict, Tuple
import polars as pl
from polars import DataFrame
import torch


def get_dataset_cleaned() -> Tuple[DataFrame, Dict[str, int]]:
    """
    This function reads a CSV file containing match data from the game League of Legends
    cleans and transforms the data, and returns the cleaned DataFrame and a dictionary
    mapping champion names to unique indices.

    Returns:
        Tuple[DataFrame, Dict[str, int]]: The cleaned DataFrame and the mapping from
        champion names to indices.
    """
    # Define the path to the dataset
    PATH = "dataset\\2023_LoL_esports_match_data_from_OraclesElixir.csv"
    # Define the columns needed for the analysis
    COLUMNS_NEEDED = ["gameid", "league", "side", "position", "champion", "result"]
    # Define the data types for each column
    COLUMNS_TYPES = {
        "gameid": pl.Utf8,
        "league": pl.Utf8,
        "side": pl.Utf8,
        "position": pl.Utf8,
        "champion": pl.Utf8,
        "result": pl.Int8,
    }

    # Read the dataset, selecting only the needed columns and converting them to the
    # specified types
    df = pl.read_csv(
        source=PATH, has_header=True, columns=COLUMNS_NEEDED, dtypes=COLUMNS_TYPES
    ).drop_nulls()

    # Assert that each game is listed exactly 10 times (once for each player)
    assert df["gameid"].n_unique() == df.shape[0] / 10

    # Create a list of unique champions
    LIST_OF_CHAMPIONS = df["champion"].unique().to_list()

    # Create a mapping from champion names to unique indices
    CHAMP_TO_IDX = {champ: idx for idx, champ in enumerate(LIST_OF_CHAMPIONS)}

    # Add the 'champion_idx' column
    df = df.with_columns(pl.col("champion").apply(lambda x: CHAMP_TO_IDX.get(x, x)))

    # Group the data by game, league, side, and result, aggregating the champions into lists
    df = df.groupby(["gameid", "league", "side", "result"]).agg(
        pl.col("champion").apply(list).alias("champions")
    )

    # Split the DataFrame into two based on 'side'
    df_blue = (
        df.filter(df["side"] == "Blue")
        .drop("side")
        .rename({"champions": "Blue_champions"})
    )
    df_red = (
        df.filter(df["side"] == "Red")
        .drop("side")
        .rename({"champions": "Red_champions"})
    )

    # Update 'result' column to indicate the winning side
    df_blue = df_blue.with_columns(
        pl.col("result").apply(lambda x: "Blue" if x == 1 else "Red").alias("result")
    )

    df_red = df_red.with_columns(
        pl.col("result").apply(lambda x: "Red" if x == 1 else "Blue").alias("result")
    )

    # Ensure that the 'result' column is consistent between the 'Blue' and 'Red' DataFrames
    assert (
        (df_blue.sort(by="gameid")["result"])
        .eq(df_red.sort(by="gameid")["result"])
        .all()
    )

    # Join the 'Blue' and 'Red' DataFrames back together
    df = df_blue.join(df_red, on=["gameid", "league", "result"])

    # Convert the 'result' column to a binary format
    df = df.with_columns(
        pl.col("result").apply(lambda x: 1 if x == "Blue" else 0).alias("result_binary")
    )

    return (df, CHAMP_TO_IDX)


def save_data(df: DataFrame, type: str = "train") -> None:
    """
    Saves the given DataFrame into multiple tensor files based on the team and the result.

    Args:
        df (DataFrame): The DataFrame to save.
        type (str, optional): The type of the data, either 'train' or 'test'.
            Defaults to "train".
    """
    assert type in ["train", "test"]

    # Convert DataFrame columns to tensors
    blue_champs, red_champs, result = get_torch_tensors(df)

    # Save the tensors to files
    torch.save(blue_champs, f=f"dataset\\blue_champions_{type}.pt")
    torch.save(red_champs, f=f"dataset\\red_champions_{type}.pt")
    torch.save(result, f=f"dataset\\result_{type}.pt")


def get_torch_tensors(df: DataFrame) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Converts columns in the DataFrame into PyTorch tensors.

    Args:
        df (DataFrame): The DataFrame to convert to tensors.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: The blue team, red team, and result tensors.
    """
    return (
        torch.tensor(df["Blue_champions"].to_list()),
        torch.tensor(df["Red_champions"].to_list()),
        torch.tensor(df["result_binary"].to_list()),
    )


def generate_dataset() -> Dict[str, int]:
    """
    Cleans the dataset, splits it into training and testing sets, and saves each set to a file.

    Returns:
        Dict[str, int]: A dictionary mapping champion names to unique indices.
    """
    # Clean and shuffle the dataset
    df, CHAMP_TO_IDX = get_dataset_cleaned()
    df = df.sample(fraction=1, shuffle=True)

    # Calculate the size of the testing set (20% of total)
    test_size = 20 * df.shape[0] // 100

    # Split the dataset into testing and training sets
    test, train = df.head(test_size), df.tail(-test_size)

    # Save the training and testing data to files
    save_data(train, type="train")
    save_data(test, type="test")

    return CHAMP_TO_IDX
