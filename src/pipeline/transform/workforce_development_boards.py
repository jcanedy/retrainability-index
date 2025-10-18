import polars as pl
from pipeline.extract.fetchers import (
    fetch_datacommons_ids, 
    fetch_datacommons_land_areas, 
    fetch_datacommons_stats
)

NAMES_TO_DCIDS_OVERRIDE = {
    'St. Croix Island, Virgin Islands': "geoId/78010",
    'Austin City, Less Austin City Part In Williamson County, Texas': "geoId/4805000",
    'Balance Of Collin County Less Dallas City Part, Texas': "geoId/48085",
    'Balance Of Denton County Less Dallas City Part, Texas': "geoId/48121",
    'New Shoreham Town, Rhode Island': "geoId/4450500",
    'Lares Municipio, Puerto Rico': "geoId/72081",
    'Hormigueros Municipio, Puerto Rico': "geoId/72067",
    'Canovas Municipio, Puerto Rico': None,
    'Luquillo Municipio, Puerto Rico': "geoId/72089",
    'Toa Baja Municipio, Puerto Rico': "geoId/72137",
    'Bayamon Municipio, Puerto Rico': "geoId/72021",
    'Arroyo Municipio, Puerto Rico': "geoId/72015",
    'Balance Of Clackamas County Less Portland City, Oregon': "geoId/41005",
    'Franklin County (Including The City Of Columbus), Ohio': "geoId/39049",
    'Mahoning County Less Youngstown City, Ohio': "geoId/39099",
    'Summit County Including City Of Akron, Ohio': "geoId/39153",
    'Hempstead Town In Nassau County, New York': "geoId/3634000",
    'Hempstead Town, New York': "geoId/3634000",
    'Essex County Less Newark City, New Jersey': "geoId/34013",
    'Div. E & T Tradereadjustact, New Jersey': None,
    'Famis Stateside, New Jersey': None,
    'Nj Trenton Central Office, New Jersey': None,
    'Response Team, New Jersey': None,
    'Ui Statewide, New Jersey': None,
    'Workfirst Operations, New Jersey': None,
    'Carson City, Nevada': "geoId/3209700",
    'Balance Of St. Louis County Less Duluth City, Minnesota': "geoId/27137",
    'Balance Of Hennepin County Less Minneapolis City, Minnesota': "geoId/27053",
    'Upper Peninsula, Michigan': None,
    'Central Upper Peninsula, Michigan': None,
    'Dighton Town, Massachusetts': "geoId/2516950",
    'Hin25055Le Town, Massachusetts': None,
    'Windsor Town, Massachusetts': "geoId/2580685",
    'Wenham Town, Massachusetts': "wikidataId/Q2418343",
    'Carver Town, Massachusetts': "wikidataId/Q372315",
    'Shutesbury Town, Massachusetts': "geoId/2561905",
    'Balance Of Bossier Parish Less Shreveport City, Louisiana': "geoId/22015",
    'Balance Of Caddo Parish Less Shreveport City, Louisiana': "geoId/22015",
    'Terrebonne Consortium, Louisiana': "geoId/22109",
    'Kentucky Statewide, Kentucky': None,
    'Adair, Iowa': "geoId/19001",
    'Audubon, Iowa': "geoId/19009",
    'Cherokee, Iowa': "geoId/19035",
    'Fremont, Iowa': "geoId/19071",
    'Greene, Iowa': "geoId/19073",
    'Hamilton, Iowa': "geoId/19079",
    'Humboldt, Iowa': "geoId/19091",
    'Monona, Iowa': "geoId/19133",
    'Osceola, Iowa': "geoId/19143",
    'Plymouth, Iowa': "geoId/19149",
    'Pocahontas, Iowa': "geoId/19151",
    'Shelby, Iowa': "geoId/19165",
    'Union, Iowa': "geoId/19175",
    'Balance Of Du Page Co Less Chicago City, Illinois': "geoId/17043",
    'Balance Of Cook Co Less:, Illinois': "geoId/17031",
    'Hanover Township, Illinois': "geoId/1732694",
    'Maine Township, Illinois': "geoId/1746162",
    'Newtown Town, Connecticut': "geoId/0952980",
    'Contra Costa County Less Richmond City, California': "geoId/06013",
    'Alameda County Less Oakland City, California': "geoId/06001",
    'Lakewood City, California': "geoId/0639892",
    'Balance Of Pulaski County Less Little Rock City, Arkansas': "geoId/05119",
    'Camp Verde Reservation, Arizona': None,
    'Hualapai Reservation, Arizona': None,
    'Yavapai Reservation, Arizona': None,
    'Balance Of Gila County Less, Arizona': "geoId/04007",
    'Balance Of Pinal County Less, Arizona': "geoId/04021",
    'Maricopa Reservation, Arizona': None,
    'Maricopa Reservation In Pinal County, Arizona': None,
    'Papago Reservation, Arizona': None,
    'Payson Community Of Yavapai-Apache, Arizona': None,
    'Camp Verde Reservation In Yavapai County, Arizona': None,
    'Fort Mc Dowell Reservation In Maricopa County, Arizona': None,
    'Hualapai Reservation In Coconino County, Arizona': None,
    'Papago Reservation In Maricopa County, Arizona': None,
    'Pascua Yaqui Reservation, Arizona': None,
    'Pascua Yaqui Reservation In Pima County, Arizona': None,
    'Payson Community Of Yavapai-Apache In Gila County, Arizona': None,
    'Salt River Reservation, Arizona': None,
    'Salt River Reservation In Maricopa County, Arizona': None,
    'Yavapai Reservation In Yavapai County, Arizona': None,
    'Fort Mc Dowell Reservation, Arizona': None,
    'Rose Island, American Samoa': None,
    'Wade Hampton Borough, Alaska': None,
    'Anchorage/Mat-Su Economic Region, Alaska': "geoId/02020",
    'Gulf Coast Economic Region, Alaska': None,
    'Interior Economic Region, Alaska': None,
    'Northern Economic Region, Alaska': None,
    'Southeast Economic Region, Alaska': None,
    'Southwest Economic Region, Alaska': None,
    'Hudson County Less Jersey City, New Jersey': "geoId/34017",
    "District Of Coulumia, District Of Columbia": "geoId/11001",
    "Mobile City/Mobile County, Alabama": "geoId/01097",
    "Seattle-King County, Washington": "geoId/53033",
    "Queens County, New York": "geoId/36081"
}

def normalize(df: pl.DataFrame) -> pl.DataFrame:
    """Normalize column names to be snakecase."""

    df_normalized = df.rename({
        "Program Year": "program_year",
        "Region": "region",
        "State/Territory": "state",
        "ETA Code": "workforce_board_code",
        "Local Board Name": "local_board",
        "Jurisdiction Name": "jurisdiction",
        "Created Timestamp": "created_timestamp",
        "Modified Timestamp": "modified_timestamp",
        "Status": "status"
    })

    return df_normalized

def filter(df: pl.DataFrame) -> pl.DataFrame:
    """Filter rows and columns."""

    df_filtered = (
        df
        .filter(
            pl.col("status").is_in(["Approved", "Edit-Pending Approval"])
        )
        .drop(["created_timestamp", "modified_timestamp", "status"])
        .sort("jurisdiction", descending=False)

        # Group by program_year, state, and workforce_board_code and collect jurisdiction as tuples
        .group_by(["program_year", "state", "workforce_board_code"])
        .agg([
            pl.col("jurisdiction")
        ])

        # Remove duplicate rows that repeat across program years with same jurisdiction
        .sort(["program_year", "state"], descending=[True, False])
        .unique(subset=["workforce_board_code", "state", "jurisdiction"])
        .sort(["program_year", "state"], descending=[True, False])

        # Make sure there is 1 jurisdiction per row and combine with state into new column
        .explode("jurisdiction")
        .with_columns(
            (pl.col("jurisdiction") + pl.lit(", ") + pl.col("state"))
            .alias("jurisdiction_state")
        )
    )
    
    return df_filtered

def join_with_datacommons_variables(df: pl.DataFrame, names_to_dcids_override: dict[str, str] = NAMES_TO_DCIDS_OVERRIDE) -> pl.DataFrame:
    """Join additional statistical variables from Data Commons (datacommons.org)."""

    # Get the unique jurisdiction, state names from df.
    names = df["jurisdiction_state"].unique().to_list()

    # Fetch Data Commons ids for each unique jurisdiction, state.
    names_to_dcid = fetch_datacommons_ids(names)

    # Override ids that were potentially not fetched successfully.
    names_to_dcid = names_to_dcid | names_to_dcids_override
    
    df_joined = df.with_columns(
        pl.col("jurisdiction_state")
        .replace(names_to_dcid)
        .alias("dcid")
    )

    # Get all ids which are not null 
    dcids = df_joined["dcid"].drop_nulls().unique().to_list()

    # Fetch land areas (in square meters) by Data Commons id
    dcids_to_land_area = fetch_datacommons_land_areas(dcids)
    dcids_to_stats = fetch_datacommons_stats(dcids)

    df_joined = (
        df_joined
        .join(
            dcids_to_land_area, on="dcid", how="left"
        )
        .join(
            dcids_to_stats, on="dcid", how="left"
        )
        .with_columns(
            (pl.col("population") / pl.col("land_area_sqm") * 1e6)
            .alias("population_per_sqkm")
        )
    )

    return df_joined

def group(df: pl.DataFrame) -> pl.DataFrame:
    """Group by program year, state, and workforce board code, applying appropriate aggregation."""
    df_grouped = (
        df
        .group_by(["program_year", "state", "workforce_board_code"])
        .agg([
            pl.col("population_per_sqkm").mean(),
            pl.col("median_age").mean(),
            pl.col("median_income").mean(),
            pl.col("jurisdiction").count().alias("jurisdiction_count")
        ])
        .with_columns(
            pl.col("workforce_board_code").cast(pl.String)
        )
        .sort(pl.col("program_year"), descending=True)
        .unique()
    )

    return df_grouped