from typing import Dict, Tuple
import polars as pl
from polars import DataFrame
import torch


ALL_CHAMPIONS = [
    "Aatrox",
    "Ahri",
    "Akali",
    "Akshan",
    "Alistar",
    "Amumu",
    "Anivia",
    "Annie",
    "Aphelios",
    "Ashe",
    "Aurelion Sol",
    "Azir",
    "Bard",
    "Bel'Veth",
    "Blitzcrank",
    "Brand",
    "Braum",
    "Caitlyn",
    "Camille",
    "Cassiopeia",
    "Cho'Gath",
    "Corki",
    "Darius",
    "Diana",
    "Dr. Mundo",
    "Draven",
    "Ekko",
    "Elise",
    "Evelynn",
    "Ezreal",
    "Fiddlesticks",
    "Fiora",
    "Fizz",
    "Galio",
    "Gangplank",
    "Garen",
    "Gnar",
    "Gragas",
    "Graves",
    "Gwen",
    "Hecarim",
    "Heimerdinger",
    "Illaoi",
    "Irelia",
    "Ivern",
    "Janna",
    "Jarvan IV",
    "Jax",
    "Jayce",
    "Jhin",
    "Jinx",
    "K'Sante",
    "Kai'Sa",
    "Kalista",
    "Karma",
    "Karthus",
    "Kassadin",
    "Katarina",
    "Kayle",
    "Kayn",
    "Kennen",
    "Kha'Zix",
    "Kindred",
    "Kled",
    "Kog'Maw",
    "LeBlanc",
    "Lee Sin",
    "Leona",
    "Lillia",
    "Lissandra",
    "Lucian",
    "Lulu",
    "Lux",
    "Malphite",
    "Malzahar",
    "Maokai",
    "Master Yi",
    "Milio",
    "Miss Fortune",
    "Mordekaiser",
    "Morgana",
    "Naafiri",
    "Nami",
    "Nasus",
    "Nautilus",
    "Neeko",
    "Nidalee",
    "Nilah",
    "Nocturne",
    "Nunu & Willump",
    "Olaf",
    "Orianna",
    "Ornn",
    "Pantheon",
    "Poppy",
    "Pyke",
    "Qiyana",
    "Quinn",
    "Rakan",
    "Rammus",
    "Rek'Sai",
    "Rell",
    "Renata Glasc",
    "Renekton",
    "Rengar",
    "Riven",
    "Rumble",
    "Ryze",
    "Samira",
    "Sejuani",
    "Senna",
    "Seraphine",
    "Sett",
    "Shaco",
    "Shen",
    "Shyvana",
    "Singed",
    "Sion",
    "Sivir",
    "Skarner",
    "Sona",
    "Soraka",
    "Swain",
    "Sylas",
    "Syndra",
    "Tahm Kench",
    "Taliyah",
    "Talon",
    "Taric",
    "Teemo",
    "Thresh",
    "Tristana",
    "Trundle",
    "Tryndamere",
    "Twisted Fate",
    "Twitch",
    "Udyr",
    "Urgot",
    "Varus",
    "Vayne",
    "Veigar",
    "Vel'Koz",
    "Vex",
    "Vi",
    "Viego",
    "Viktor",
    "Vladimir",
    "Volibear",
    "Warwick",
    "Wukong",
    "Xayah",
    "Xerath",
    "Xin Zhao",
    "Yasuo",
    "Yone",
    "Yorick",
    "Yuumi",
    "Zac",
    "Zed",
    "Zeri",
    "Ziggs",
    "Zilean",
    "Zoe",
    "Zyra",
]

NAME_CONVERTER = {
    "AurelionSol": "Aurelion Sol",
    "Belveth": "Bel'Veth",
    "Chogath": "Cho'Gath",
    "DrMundo": "Dr. Mundo",
    "FiddleSticks": "Fiddlesticks",
    "JarvanIV": "Jarvan IV",
    "KSante": "K'Sante",
    "Kaisa": "Kai'Sa",
    "Khazix": "Kha'Zix",
    "KogMaw": "Kog'Maw",
    "Leblanc": "LeBlanc",
    "LeeSin": "Lee Sin",
    "MasterYi": "Master Yi",
    "MissFortune": "Miss Fortune",
    "MonkeyKing": "Wukong",
    "Nunu": "Nunu & Willump",
    "RekSai": "Rek'Sai",
    "Renata": "Renata Glasc",
    "TahmKench": "Tahm Kench",
    "TwistedFate": "Twisted Fate",
    "Velkoz": "Vel'Koz",
    "XinZhao": "Xin Zhao",
}


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

    # Create a mapping from champion names to unique indices
    CHAMP_TO_IDX = {champ: idx for idx, champ in enumerate(ALL_CHAMPIONS)}

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


def get_challenger(region: str) -> DataFrame:
    """
    This function reads a CSV file containing match data from Challenger games
    in the game League of Legends, cleans and transforms the data, and returns the cleaned DataFrame.

    Returns:
        DataFrame: The cleaned DataFrame.
    """

    assert region in ["kr", "na", "eu", "europe"]

    # Define the path to the dataset
    PATH = f"dataset\\match_details_{region}_challengers.csv"

    # Read the dataset using polars read_csv function
    df = pl.read_csv(PATH)

    # Rename 'matchID' column to 'gameid'
    df = df.rename({"matchID": "gameid"})

    # Add a new 'league' column with a constant value "KRChallenger"
    # Add a new 'side' column based on 'teamID': "Blue" if 'teamID' is 100, otherwise "Red"
    df = df.with_columns(
        pl.col("gameid").apply(lambda x: "KRChallenger").alias("league"),
        pl.col("teamID").apply(lambda x: "Blue" if x == 100 else "Red").alias("side"),
    )

    # Delete 'teamID' column
    df = df.drop("teamID")

    # Create a mapping from champion names to unique indices
    CHAMP_TO_IDX = {champ: idx for idx, champ in enumerate(ALL_CHAMPIONS)}

    # Replace the names in the 'champion' column with their corresponding indices
    df = df.with_columns(
        pl.col("champion")
        .apply(lambda x: NAME_CONVERTER.get(x, x))
        .apply(lambda x: CHAMP_TO_IDX.get(x, x))
    )

    # Group the data by 'gameid', 'league', 'side', and 'result',
    # aggregating the champions into lists named 'champions'
    df = df.groupby(["gameid", "league", "side", "result"]).agg(
        pl.col("champion").apply(list).alias("champions")
    )

    # Split the DataFrame into two based on 'side': one for the 'Blue' side and one for the 'Red' side
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
        pl.col("result").apply(lambda x: "Blue" if x else "Red").alias("result")
    )

    df_red = df_red.with_columns(
        pl.col("result").apply(lambda x: "Red" if x else "Blue").alias("result")
    )

    # Ensure that the 'result' column is consistent between the 'Blue' and 'Red' DataFrames
    assert (
        (df_blue.sort(by="gameid")["result"])
        .eq(df_red.sort(by="gameid")["result"])
        .all()
    )

    # Join the 'Blue' and 'Red' DataFrames back together on 'gameid', 'league', and 'result'
    df = df_blue.join(df_red, on=["gameid", "league", "result"])

    # Convert the 'result' column to a binary format:
    # '1' if the winning side is 'Blue', and '0' if it's 'Red'
    df = df.with_columns(
        pl.col("result").apply(lambda x: 1 if x == "Blue" else 0).alias("result_binary")
    )

    return df


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
    # KR Challengers
    # df2 = get_challenger("kr")
    # df3 = get_challenger("eu")
    df4 = get_challenger("europe")

    # Add the KR Challengers to the dataset
    # df = df.vstack(df2)
    # df = df.vstack(df3)
    df = df.vstack(df4)

    df = df.sample(fraction=1, shuffle=True)

    # Calculate the size of the testing set (20% of total)
    test_size = 20 * df.shape[0] // 100

    # Split the dataset into testing and training sets
    test, train = df.head(test_size), df.tail(-test_size)

    # Save the training and testing data to files
    save_data(train, type="train")
    save_data(test, type="test")

    return df, CHAMP_TO_IDX
