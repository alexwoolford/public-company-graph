"""
Neo4j data loaders for Domain and Technology nodes.

This module provides functions to load data structures into Neo4j.
"""

from typing import List


def load_domains(driver, domains: List[dict], batch_size: int = 1000, database: str = None):
    """
    Load Domain nodes into Neo4j.

    Args:
        driver: Neo4j driver instance
        domains: List of domain dictionaries from read_domains()
        batch_size: Number of domains to process per batch
        database: Neo4j database name
    """
    with driver.session(database=database) as session:
        for i in range(0, len(domains), batch_size):
            batch = domains[i : i + batch_size]

            query = """
            UNWIND $batch AS row
            MERGE (d:Domain {final_domain: row.final_domain})
            SET d.domain = row.domain,
                d.status = row.status,
                d.status_description = row.status_description,
                d.response_time = row.response_time,
                d.timestamp = row.timestamp,
                d.is_mobile_friendly = row.is_mobile_friendly,
                d.spf_record = row.spf_record,
                d.dmarc_record = row.dmarc_record,
                d.title = row.title,
                d.keywords = row.keywords,
                d.description = row.description,
                d.creation_date = row.creation_date,
                d.expiration_date = row.expiration_date,
                d.registrar = row.registrar,
                d.registrant_country = row.registrant_country,
                d.registrant_org = row.registrant_org,
                d.loaded_at = datetime()
            """

            session.run(query, batch=batch)

            if (i // batch_size + 1) % 10 == 0:
                print(f"  Processed {i + len(batch)}/{len(domains)} domains...")


def load_technologies(
    driver, tech_mappings: List[dict], batch_size: int = 1000, database: str = None
):
    """
    Load Technology nodes and USES relationships into Neo4j.

    Args:
        driver: Neo4j driver instance
        tech_mappings: List of technology mappings from read_technologies()
        batch_size: Number of relationships to process per batch
        database: Neo4j database name
    """
    # Extract unique technologies
    unique_techs = set(
        (row["technology_name"], row["technology_category"]) for row in tech_mappings
    )

    with driver.session(database=database) as session:
        # Create Technology nodes
        query = """
        UNWIND $techs AS tech
        MERGE (t:Technology {name: tech.name})
        SET t.category = tech.category,
            t.loaded_at = datetime()
        """
        tech_data = [{"name": name, "category": category} for name, category in unique_techs]
        session.run(query, techs=tech_data)
        print(f"  ✓ Created {len(unique_techs)} Technology nodes")

        # Create USES relationships
        for i in range(0, len(tech_mappings), batch_size):
            batch = tech_mappings[i : i + batch_size]

            query = """
            UNWIND $batch AS row
            MATCH (d:Domain {final_domain: row.final_domain})
            MATCH (t:Technology {name: row.technology_name})
            MERGE (d)-[r:USES]->(t)
            SET r.loaded_at = datetime()
            """

            session.run(query, batch=batch)

            if (i // batch_size + 1) % 10 == 0:
                print(f"  Processed {i + len(batch)}/{len(tech_mappings)} relationships...")
