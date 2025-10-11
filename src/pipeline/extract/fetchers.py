import polars as pl
from datacommons_client.client import DataCommonsClient
from typing import Any

client = DataCommonsClient(dc_instance="datacommons.one.org")

def fetch_datacommons_ids(names: str | list[str]) -> dict[str, Any]:
    """Fetch Data Commons ids by name."""
    results = client.resolve.fetch_dcids_by_name(names).to_flat_dict()

    # Filter for entries that only have 1 dcid.
    ids = {name: results[name] for name in names if isinstance(results[name], str)}
    
    return ids

def fetch_datacommons_land_areas(dcids: list[str]) -> pl.DataFrame:

    land_areas = client.node.fetch_property_values(node_dcids=dcids, properties="landArea").data

    dcids_to_land_area = { "dcid": [], "land_area_sqm": [] }

    for place_dcid, arc_obj in land_areas.items():
        dcids_to_land_area["dcid"].append(place_dcid)
        dcids_to_land_area["land_area_sqm"].append(_get_landarea_dcid(arc_obj))

    return pl.DataFrame(dcids_to_land_area)

def fetch_datacommons_stats(dcids: list[str]) -> pl.DataFrame:
    stats = client.observations_dataframe(
        entity_dcids=dcids, 
        variable_dcids=['Count_Person', 'Median_Age_Person', 'Median_Income_Person'], 
        date='latest'
    )

    # Because we can get multiple "latest" rows for a single entity, 
    # we will take the value which is the most recent date (year).
    stats = (
        stats
        .sort_values(by="date", ascending=False)
        .drop_duplicates(subset=["entity", "variable"])
        .pivot_table(values=["value"], index=["entity"], columns=["variable"])
        .reset_index()
        .droplevel(level=0, axis=1)
        .rename(columns={
            "": "dcid",
            "Count_Person": "population",
            "Median_Age_Person": "median_age",
            "Median_Income_Person": "median_income"
        })
    )

    return pl.from_pandas(stats)


# HELPER FUNCTIONS

def _get_landarea_dcid(arcs_obj):
    """
    Given an Arcs object like:
      Arcs(arcs={'landArea': NodeGroup(nodes=[Node(dcid='SquareMeter90056657', ...)])})
    return the dcid string, e.g. 'SquareMeter90056657'.
    """
    try:
        # Grab the first NodeGroup under 'landArea'
        nodes = arcs_obj.arcs.get("landArea").nodes
        if nodes:
            return _parse_landarea_dcid(nodes[0].dcid)
    except Exception as e:
        print(f"Failed to parse dcid: {e}")
    return None

def _parse_landarea_dcid(s):
    if (type(s) != str):
        return None

    split_s = s.split("SquareMeter")
    if len(split_s) > 1:
        return int(split_s[1])
    
    return None