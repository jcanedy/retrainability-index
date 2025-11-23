from prefect import flow, task
import polars as pl
from pipeline.extract.readers import read_excel
from pipeline.transform import workforce_development_boards
from pipeline.load import writers

@task
def task_workforce_board_excels_read() -> pl.DataFrame:
    dict_workforce_boards = {}
    dict_workforce_boards[2023] = read_excel("data/raw/workforce_development_board_codes/wdb_codes_2023.xlsx", engine="xlsx2csv", read_options={ "skip_rows": 8 })
    dict_workforce_boards[2022] = read_excel("data/raw/workforce_development_board_codes/wdb_codes_2022.xlsx", engine="xlsx2csv", read_options={ "skip_rows": 8 })
    dict_workforce_boards[2021] = read_excel("data/raw/workforce_development_board_codes/wdb_codes_2021.xlsx", engine="xlsx2csv", read_options={ "skip_rows": 8 })
    dict_workforce_boards[2020] = read_excel("data/raw/workforce_development_board_codes/wdb_codes_2020.xlsx", engine="xlsx2csv", read_options={ "skip_rows": 8 })
    dict_workforce_boards[2019] = read_excel("data/raw/workforce_development_board_codes/wdb_codes_2019.xlsx", engine="xlsx2csv", read_options={ "skip_rows": 8 })
    dict_workforce_boards[2018] = read_excel("data/raw/workforce_development_board_codes/wdb_codes_2018.xlsx", engine="xlsx2csv", read_options={ "skip_rows": 8 })
    dict_workforce_boards[2017] = read_excel("data/raw/workforce_development_board_codes/wdb_codes_2017.xlsx", engine="xlsx2csv", read_options={ "skip_rows": 8 })
    dict_workforce_boards[2016] = read_excel("data/raw/workforce_development_board_codes/wdb_codes_2016.xlsx", engine="xlsx2csv", read_options={ "skip_rows": 8 })

    df_workforce_boards = pl.concat(dict_workforce_boards.values(), how="diagonal")
    return df_workforce_boards

@task
def task_workforce_development_boards_normalize(df: pl.DataFrame) -> pl.DataFrame:
    return workforce_development_boards.normalize(df)

@task
def task_workforce_development_boards_filter(df: pl.DataFrame) -> pl.DataFrame:
    return workforce_development_boards.filter(df)

@task
def task_workforce_development_boards_join_with_datacommons_variables(df: pl.DataFrame) -> pl.DataFrame:
    return workforce_development_boards.join_with_datacommons_variables(df)

@task
def task_workforce_development_boards_group(df: pl.DataFrame) -> pl.DataFrame:
    return workforce_development_boards.group(df)

@task
def task_workforce_development_boards_all_write(df: pl.DataFrame):

    df_grouped = (
        df
        .group_by(["program_year", "state", "workforce_board_code"])
        .agg([
            pl.col("jurisdiction").count().alias("jurisdiction_count"),
            pl.col("jurisdiction")
        ])
        .with_columns(
            pl.col("workforce_board_code").cast(pl.String)
        )
        .sort(pl.col("program_year"), descending=True)
        .unique()
    )

    writers.write_parquet(df_grouped, "data/processed/workforce_boards/workforce_boards_all_grouped.parquet", compression="zstd")

    return

@task
def task_workforce_development_boards_write(df: pl.DataFrame):
    df_grouped = task_workforce_development_boards_group(df)

    writers.write_parquet(df, "data/processed/workforce_boards/workforce_boards.parquet", compression="zstd")
    writers.write_parquet(df_grouped, "data/processed/workforce_boards/workforce_boards_grouped.parquet", compression="zstd")

    return 


@flow
def pipeline_workforce_development_boards() -> None:
    df_workforce_boards = task_workforce_board_excels_read()
    df_workforce_boards = task_workforce_development_boards_normalize(df_workforce_boards)
    df_workforce_boards = task_workforce_development_boards_filter(df_workforce_boards)
    
    task_workforce_development_boards_all_write(df_workforce_boards)
    
    df_workforce_boards = task_workforce_development_boards_join_with_datacommons_variables(df_workforce_boards)
    task_workforce_development_boards_write(df_workforce_boards)

    return 

if __name__ == "__main__":
    pipeline_workforce_development_boards()