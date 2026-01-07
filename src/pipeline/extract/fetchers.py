import polars as pl
import pandas as pd
from datacommons_client.client import DataCommonsClient
from typing import Any, List, Dict
import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("DATA_COMMONS_API_KEY")
API_BATCH_SIZE = 200

client = DataCommonsClient(api_key=API_KEY)

def fetch_datacommons_ids(names: str | List[str]) -> Dict[str, Any]:
    """
    Resolves names to County DCIDs. 
    If a name resolves to a City (7-digit), it fetches the parent County.
    If a name resolves to a County (5-digit), it keeps it.
    """
    # Resolve names broadly (don't limit to County yet, or you might miss cities)
    # to_flat_dict() automatically picks the best single candidate.
    resolved = client.resolve.fetch_dcids_by_name(names).to_flat_dict()

    final_ids = {}
    cities_to_lookup = {} # Store {city_dcid: original_name} for mapping back later

    # Sort results into "Is County" vs "Is City" based on ID length
    for name, dcid in resolved.items():
        if not dcid:
            continue
        
        if isinstance(dcid, list):
            if len(dcid) > 0:
                dcid = dcid[0]
            continue

        # Extract numeric part: "geoId/06037" -> "06037"
        code_part = dcid.split('/')[-1]

        # US Logic: Counties are 5 digits, States 2, Cities/others usually 7
        if len(code_part) == 5 and code_part.isdigit():
            final_ids[name] = dcid
        else:
            # It's likely a city, queue it for parent lookup
            cities_to_lookup[dcid] = name

    # Batch fetch parents for the identified cities
    if cities_to_lookup:
        parent_response = client.node.fetch(
            node_dcids=list(cities_to_lookup.keys()),
            expression="->containedInPlace"
        )

        if parent_response.data:
            for city_dcid, node in parent_response.data.items():
                
                # Safely get the containedInPlace list
                arcs_dict = node.arcs if hasattr(node, 'arcs') else {}
                target_arc = arcs_dict.get('containedInPlace')
                
                found_county_dcid = None
                
                # ITERATE through all parents to find the specific County type
                if target_arc and target_arc.nodes:
                    for parent in target_arc.nodes:
                        # Check if 'County' is in the list of types for this parent
                        if "County" in parent.types:
                            found_county_dcid = parent.dcid
                            break
                
                # Map back to original name
                original_name = cities_to_lookup[city_dcid]
                
                if found_county_dcid:
                    final_ids[original_name] = found_county_dcid
                else:
                    print(f"Warning: {original_name} ({city_dcid}) has parents, but none are Counties.")

    return final_ids

def fetch_datacommons_land_areas(dcids: list[str]) -> pl.DataFrame:

    land_areas = client.node.fetch_property_values(node_dcids=dcids, properties="landArea").data

    dcids_to_land_area = { "dcid": [], "land_area_sqm": [] }

    for place_dcid, arc_obj in land_areas.items():
        dcids_to_land_area["dcid"].append(place_dcid)
        dcids_to_land_area["land_area_sqm"].append(_get_landarea_dcid(arc_obj))

    return pl.DataFrame(dcids_to_land_area)

def fetch_datacommons_stats(dcids: list[str]) -> pl.DataFrame:

    stats_chunks = []

    for i in range(0, len(dcids), API_BATCH_SIZE):
        # Slice the list to get the current chunk
        chunk = dcids[i : i + API_BATCH_SIZE]
        
        print(f"Processing batch {i} to {i + len(chunk)}...")
        
        try:
            # Call the API for just this chunk
            stats_chunk = client.observations_dataframe(
                entity_dcids=chunk, 
                variable_dcids=['Count_Person', 'Median_Age_Person', 'Median_Income_Person', 'UnemploymentRate_Person'], 
                property_filters={"importName": ["BLS_LAUS", "CensusACS5YearSurvey"]},
                date='all'
            )
            
            # If data was found, add it to our list
            if not stats_chunk.empty:
                stats_chunks.append(stats_chunk)
                
        except Exception as e:
            print(f"Error in batch starting at index {i}: {e}")
            # Optional: continue to next batch even if one fails
            pass

    if not stats_chunks:
        print("No data found.")
        return
    
    stats = pd.concat(stats_chunks)

    # Format all measurement dates as just the corresponding year
    stats["date"] = pd.to_datetime(stats["date"], format='ISO8601').dt.year

    # Because we can get multiple "latest" rows for a single entity, 
    # we will take the value which is the most recent date (year).

    groupby_columns = [
        'date',
        'entity',
        'variable',
    ]

    stats = (
        stats
        .groupby(groupby_columns)["value"].mean()
        .reset_index()
        .pivot_table(values=["value"], index=["date", "entity"], columns=["variable"])
        .droplevel(level=0, axis=1)
        .reset_index()
        .rename(columns={
            "date": "program_year",
            "entity": "dcid",
            "Count_Person": "population",
            "Median_Age_Person": "median_age",
            "Median_Income_Person": "median_income",
            "UnemploymentRate_Person": "unemployment_rate"
        })
        .reset_index(drop=True)
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