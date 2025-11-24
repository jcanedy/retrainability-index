import polars as pl
from pipeline.extract.fetchers import (
    fetch_datacommons_ids, 
    fetch_datacommons_land_areas, 
    fetch_datacommons_stats
)
from pipeline.extract.readers import (
    read_excel,
    read_csv,
    read_parquet
)

NAMES_TO_DCIDS_OVERRIDE = {
    'St. Croix Island, Virgin Islands': ["geoId/78010"],
    'Austin City, Less Austin City Part In Williamson County, Texas': ["geoId/48453", "geoId/48209", "geoId/48021"],
    'Balance Of Collin County Less Dallas City Part, Texas': ["geoId/48085"],
    'Balance Of Denton County Less Dallas City Part, Texas': ["geoId/48121"],
    'New Shoreham Town, Rhode Island': ["geoId/44009"],
    'East Providence City, Rhode Island': ["geoId/44007"],
    'West Greenwich Town, Rhode Island': ["geoId/44003"],
    'Lares Municipio, Puerto Rico': ["geoId/72081"],
    'Hormigueros Municipio, Puerto Rico': ["geoId/72067"],
    'Canovas Municipio, Puerto Rico': ["geoId/72029"],
    'Luquillo Municipio, Puerto Rico': ["geoId/72089"],
    'Toa Baja Municipio, Puerto Rico': ["geoId/72137"],
    'Bayamon Municipio, Puerto Rico': ["geoId/72021"],
    'Arroyo Municipio, Puerto Rico': ["geoId/72015"],
    'Loiza Municipio, Puerto Rico': ["geoId/72087"],
    'Balance Of Clackamas County Less Portland City, Oregon': ["geoId/41005"],
    'Franklin County (Including The City Of Columbus), Ohio': ["geoId/39049"],
    'Mahoning County Less Youngstown City, Ohio': ["geoId/39099"],
    'Summit County Including City Of Akron, Ohio': ["geoId/39153"],
    'Hamilton County (Including The City Of Cincinnati), Ohio': ["geoId/39061"],
    'Hempstead Town In Nassau County, New York': ["geoId/36059"],
    'Hempstead Town, New York': ["geoId/36059"],
    'Essex County Less Newark City, New Jersey': ["geoId/34013"],
    'Div. E & T Tradereadjustact, New Jersey': None,
    'Famis Stateside, New Jersey': None,
    'Nj Trenton Central Office, New Jersey': None,
    'Response Team, New Jersey': None,
    'Ui Statewide, New Jersey': None,
    'Workfirst Operations, New Jersey': None,
    'Carson City, Nevada': ["geoId/3209700"],
    'Balance Of St. Louis County Less Duluth City, Minnesota': ["geoId/27137"],
    'Balance Of Hennepin County Less Minneapolis City, Minnesota': ["geoId/27053"],
    'Upper Peninsula, Michigan': ["geoId/26053", "geoId/26083", "geoId/26061", "geoId/26131", "geoId/26013", "geoId/26071", "geoId/26103", "geoId/26043", "geoId/26109", "geoId/26003", "geoId/26041", "geoId/26153", "geoId/26095", "geoId/26033", " geoId/26097"],
    'Central Upper Peninsula, Michigan': ["geoId/26053", "geoId/26083", "geoId/26061", "geoId/26131", "geoId/26013", "geoId/26071", "geoId/26103", "geoId/26043", "geoId/26109", "geoId/26003", "geoId/26041", "geoId/26153", "geoId/26095", "geoId/26033", " geoId/26097"],
    'Dighton Town, Massachusetts': ["geoId/25005"],
    'Hin25055Le Town, Massachusetts': ["geoId/25003"],
    'Windsor Town, Massachusetts': ["geoId/25003"],
    'Wenham Town, Massachusetts': ["geoId/25009"],
    'Carver Town, Massachusetts': ["geoId/25023"],
    'Shutesbury Town, Massachusetts': ["geoId/25011"],
    'Northbridge Town, Massachusetts': ["geoId/25027"],
    'Sherborn Town, Massachusetts': ["geoId/25017"],
    'Petersham Town, Massachusetts': ["geoId/25027"],
    'North Adams City, Massachusetts': ["geoId/25003"],
    'Nahant Town, Massachusetts': ["geoId/25009"],
    'Pepperell Town, Massachusetts': ["geoId/25017"],
    'Groveland Town, Massachusetts': ["geoId/25009"],
    'West Tisbury Town, Massachusetts': ["geoId/25007"],
    'Long Beach City In Nassau County, New York': ["geoId/36059"],
    'Balance Of Bossier Parish Less Shreveport City, Louisiana': ["geoId/22015"],
    'Balance Of Caddo Parish Less Shreveport City, Louisiana': ["geoId/22015"],
    'Terrebonne Consortium, Louisiana': ["geoId/22109"],
    'Kentucky Statewide, Kentucky': None,
    'Adair, Iowa': ["geoId/19001"],
    'Audubon, Iowa': ["geoId/19009"],
    'Cherokee, Iowa': ["geoId/19035"],
    'Fremont, Iowa': ["geoId/19071"],
    'Greene, Iowa': ["geoId/19073"],
    'Hamilton, Iowa': ["geoId/19079"],
    'Humboldt, Iowa': ["geoId/19091"],
    'Monona, Iowa': ["geoId/19133"],
    'Osceola, Iowa': ["geoId/19143"],
    'Plymouth, Iowa': ["geoId/19149"],
    'Pocahontas, Iowa': ["geoId/19151"],
    'Shelby, Iowa': ["geoId/19165"],
    'Union, Iowa': ["geoId/19175"],
    'Balance Of Du Page Co Less Chicago City, Illinois': ["geoId/17043"],
    'Balance Of Cook Co Less:, Illinois': ["geoId/17031"],
    'Hanover Township, Illinois': ["geoId/1732694"],
    'Maine Township, Illinois': ["geoId/1746162"],
    'Newtown Town, Connecticut': ["geoId/0952980"],
    'Artesia City In Los Angeles County, California': ["geoId/06037"],
    'Arcadia City In Los Angeles County, California': ["geoId/06037"],
    'La Canada/Flintridge City In Los Angeles County, California': ["geoId/06037"],
    'Duarte City In Los Angeles County, California': ["geoId/06037"],
    'Gardena City In Los Angeles County, California': ["geoId/06037"],
    'Monrovia City In Los Angeles County, California': ["geoId/06037"],
    'Downey City In Los Angeles County, California': ["geoId/06037"],
    'Pasadena City In Los Angeles County, California': ["geoId/06037"],
    'Contra Costa County Less Richmond City, California': ["geoId/06013"],
    'Alameda County Less Oakland City, California': ["geoId/06001"],
    'Lawndale City, California': ["geoId/06001"],
    'Hermosa Beach City In Los Angeles County, California': ["geoId/06001"],
    'Burbank City In Los Angeles County, California': ["geoId/06001"],
    'Carson City In Los Angeles County, California': ["geoId/06001"],
    'Manhattan Beach City In Los Angeles County, California': ["geoId/06001"],
    'Milpitas City In Santa Clara County, California': ["geoId/06085"],
    'Inglewood City In Los Angeles County, California': ["geoId/06085"],
    'Hawthorne City In Los Angeles County, California': ["geoId/06085"],
    'Hawaiian Gardens City In Los Angeles County, California': ["geoId/06085"],
    'Redondo Beach City In Los Angeles County, California': ["geoId/06085"],
    'Bellflower City In Los Angeles County, California': ["geoId/06085"],
    'El Segundo City In Los Angeles County, California': ["geoId/06001"],
    'South Pasadena City In Los Angeles County, California': ["geoId/06001"],
    'Lomita City In Los Angeles County, California': ["geoId/06001"],
    'Lawndale City In Los Angeles County, California': ["geoId/06001"],
    'Glendale City In Los Angeles County, California': ["geoId/06001"],
    'Torrance City In Los Angeles County, California': ["geoId/06001"],
    'Norwalk City In Los Angeles County, California': ["geoId/06001"],
    'Sierra Madre City In Los Angeles County, California': ["geoId/06001"],
    'Lakewood City In Los Angeles County, California': ["geoId/06001"],
    'Cerritos City In Los Angeles County, California': ["geoId/06001"],
    'Lakewood City, California': ["geoId/06037"],
    'Danbury City, Connecticut': ["geoId/09001"],
    'North Stonington Town, Connecticut': ["geoId/09011"],
    'Chicago City In Cook/Du Page Counties, Illinois': ["geoId/17043", "geoId/17031"],
    'Balance Of Pulaski County Less Little Rock City, Arkansas': ["geoId/05119"],
    'Phoenix City, Arizona': ["geoId/04013"],
    'Camp Verde Reservation, Arizona': None,
    'Hualapai Reservation, Arizona': None,
    'Yavapai Reservation, Arizona': None,
    'Balance Of Gila County Less, Arizona': ["geoId/04007"],
    'Balance Of Pinal County Less, Arizona': ["geoId/04021"],
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
    'Skagway-Yakutat-Angoon Borough, Alaska': ["geoId/02230"],
    'Wade Hampton Borough, Alaska': ["geoId/02158"],
    'Anchorage/Mat-Su Economic Region, Alaska': ["geoId/02020", "geoId/02170"],
    'Gulf Coast Economic Region, Alaska': ["geoId/02150", "geoId/02122", "geoId/02063", "geoId/02066"],
    'Interior Economic Region, Alaska': ["geoId/02290", "geoId/02068", "geoId/02090", "geoId/02240"],
    'Northern Economic Region, Alaska': ["geoId/02185", "geoId/02188", "geoId/02180"],
    'Southeast Economic Region, Alaska': ["geoId/02282", "geoId/02105", "geoId/02220", "geoId/02198", "geoId/02130", "geoId/02275", "geoId/02195", "geoId/02110", "geoId/02185", "geoId/02100"],
    'Southwest Economic Region, Alaska': ["geoId/02158", "geoId/02050", "geoId/02070", "geoId/02060", "geoId/02164", "geoId/02013", "geoId/02016"],
    'Hudson County Less Jersey City, New Jersey': ["geoId/34017"],
    "District Of Coulumia, District Of Columbia": ["geoId/11001"],
    "Mobile City/Mobile County, Alabama": ["geoId/01097"],
    "Seattle-King County, Washington": ["geoId/53033"],
    "Queens County, New York": ["geoId/36081"],
    "St. John Island, Virgin Islands": None,
    'County Of Charles City, Virginia': ["geoId/51036"]
}

def _read_county_diversity_index() -> pl.DataFrame:

    df = read_excel("data/raw/workforce_development_board_codes/county_diversity_index.xlsx", engine="xlsx2csv")

    df = df.select(
        pl.format(
            "geoId/{}", 
            pl.col("region_code").cast(pl.String).str.zfill(5)
        ).alias("dcid"),
        pl.col("diversity_index_2021").alias("diversity_index")
    )

    return df

def _read_county_household_debt() -> pl.DataFrame:

    df = read_csv("data/raw/workforce_development_board_codes/county_household_debt.csv")

    df = df.group_by(
        ["year", "area_fips"]
    ).agg(
        pl.col("low").mean(),
        pl.col("high").mean()
    ).select(
        pl.col("year").alias("program_year"),
        pl.format(
            "geoId/{}", 
            pl.col("area_fips").cast(pl.String).str.zfill(5)
        ).alias("dcid"),
        pl.col("low").alias("household_debt_to_income_low"),
        pl.col("high").alias("household_debt_to_income_high")
    )

    return df

def _read_county_average_commute_time() -> pl.DataFrame:
    df = read_parquet("data/raw/workforce_development_board_codes/county_average_commute_time.parquet")

    df = df.select(
        pl.format(
            "geoId/{}",
            pl.col("geoid")
        ).alias("dcid"),
        pl.col("commuting_time").alias("mean_commuting_time_min")
    )

    return df

def _read_county_rural_urban_continuum_codes() -> pl.DataFrame:
    df = read_excel("data/raw/workforce_development_board_codes/county_rural_urban_continuum_codes_2023.xlsx", engine="xlsx2csv")

    df = df.select(
        pl.format(
            "geoId/{}", 
            pl.col("FIPS").cast(pl.String).str.zfill(5)
        ).alias("dcid"),
        pl.col("RUCC_2023").alias("rucc")
    )

    return df

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
    names_to_dcid = {k: [v] for k, v in names_to_dcid.items()}
    names_to_dcid.update(names_to_dcids_override)

    df_joined = (
        df.with_columns(
            pl.col("jurisdiction_state")
            .replace(
                names_to_dcid,
                default=None,
                return_dtype=pl.List(pl.String)
            )
            .alias("dcid")
        )
        .explode("dcid")
    )

    # Get all ids which are not null 
    dcids = df_joined["dcid"].drop_nulls().unique().to_list()

    # Fetch land areas (in square meters) by Data Commons id
    dcids_to_land_area = fetch_datacommons_land_areas(dcids)
    dcids_to_stats = fetch_datacommons_stats(dcids)

    # Get county diversity index
    df_diversity_index = _read_county_diversity_index()
    df_household_debt = _read_county_household_debt()
    df_commute_time = _read_county_average_commute_time()
    df_rucc = _read_county_rural_urban_continuum_codes()

    df_joined = (
        df_joined
        .join(
            dcids_to_land_area, on="dcid", how="left"
        )
        .join(
            dcids_to_stats, on=["dcid", "program_year"], how="left"
        )
        .join(
            df_diversity_index, on=["dcid"], how="left"
        )
        .join(
            df_household_debt, on=["dcid", "program_year"], how="left"
        )
        .join(
            df_commute_time, on=["dcid"], how="left"
        )
        .join(
            df_rucc, on=["dcid"], how="left"
        )
        .group_by([
            "program_year", 
            "state", 
            "workforce_board_code", 
            "jurisdiction",
            "jurisdiction_state"
        ]).agg(
            pl.col("dcid"),
            pl.col("land_area_sqm").sum(),
            pl.col("population").sum(),
            pl.col("median_age").mean(),
            pl.col("median_income").mean(),
            pl.col("unemployment_rate").mean(),
            pl.col("diversity_index").mean(),
            pl.col("household_debt_to_income_low").mean(),
            pl.col("household_debt_to_income_high").mean(),
            pl.col("mean_commuting_time_min").mean(),
            pl.col("rucc").mean()
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
            pl.col("land_area_sqm").mean(),
            pl.col("population").mean(),
            pl.col("median_age").mean(),
            pl.col("median_income").mean(),
            pl.col("unemployment_rate").mean(),
            pl.col("diversity_index").mean(),
            pl.col("household_debt_to_income_low").mean(),
            pl.col("household_debt_to_income_high").mean(),
            pl.col("mean_commuting_time_min").mean(),
            pl.col("rucc").mean(),
            pl.col("population_per_sqkm").mean(),
            pl.col("jurisdiction").count().alias("jurisdiction_count"),
            pl.col("jurisdiction").alias("jurisdictions")
        ])
        .with_columns(
            pl.col("workforce_board_code").cast(pl.String)
        )
        .sort(pl.col("program_year"), descending=True)
        .unique()
    )

    return df_grouped