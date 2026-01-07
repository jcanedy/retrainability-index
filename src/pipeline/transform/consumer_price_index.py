import polars as pl

def normalize(df: pl.DataFrame) -> pl.DataFrame:

    df = df.rename({
        "Year": "year",
        "Jan": "jan",
        "Feb": "feb",
        "Mar": "mar",
        "Apr": "apr",
        "May": "may",
        "Jun": "jun",
        "Jul": "jul",
        "Aug": "aug",
        "Sep": "sep",
        "Oct": "oct",
        "Nov": "nov",
        "Dec": "dec",
        "Annual": "annual",
        "HALF1": "h1",
        "HALF2": "h2"
    })

    return df