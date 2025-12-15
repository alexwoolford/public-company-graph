#!/usr/bin/env python3
"""
Orchestration script to run all data pipelines in the correct order.

This script recreates the graph from scratch in the correct order:
1. Bootstrap Graph: Load Domain and Technology nodes from SQLite
2. Load Company Data: Collect domains, create embeddings, load Company nodes
3. Compute GDS Features: Technology adoption, affinity, and company similarity

All steps run in sequence to ensure the graph is complete and correct.

Usage:
    python scripts/run_all_pipelines.py          # Dry-run (plan only)
    python scripts/run_all_pipelines.py --execute  # Actually run all pipelines
"""

import argparse
import subprocess
import sys
from pathlib import Path

# Script paths
SCRIPT_DIR = Path(__file__).parent
BOOTSTRAP_SCRIPT = SCRIPT_DIR / "bootstrap_graph.py"
COMPUTE_GDS_SCRIPT = SCRIPT_DIR / "compute_gds_features.py"
COLLECT_DOMAINS_SCRIPT = SCRIPT_DIR / "collect_domains.py"
CREATE_COMPANY_EMBEDDINGS_SCRIPT = SCRIPT_DIR / "create_company_embeddings.py"
LOAD_COMPANY_DATA_SCRIPT = SCRIPT_DIR / "load_company_data.py"
CREATE_DOMAIN_EMBEDDINGS_SCRIPT = SCRIPT_DIR / "create_domain_embeddings.py"
COMPUTE_DOMAIN_SIMILARITY_SCRIPT = SCRIPT_DIR / "compute_domain_similarity.py"


def run_script(script_path: Path, execute: bool = False, description: str = ""):
    """Run a script and return success status."""
    if not script_path.exists():
        print(f"✗ ERROR: Script not found: {script_path}")
        return False

    print(f"\n{'=' * 70}")
    print(f"{description}")
    print(f"{'=' * 70}")
    print(f"Running: {script_path.name}")

    cmd = [sys.executable, str(script_path)]
    if execute:
        cmd.append("--execute")

    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        if result.returncode == 0:
            print(f"✓ {script_path.name} completed successfully")
            return True
        else:
            print(f"✗ {script_path.name} failed with return code {result.returncode}")
            return False
    except subprocess.CalledProcessError as e:
        print(f"✗ {script_path.name} failed: {e}")
        return False
    except KeyboardInterrupt:
        print("\n⚠ Interrupted by user")
        return False


def main():
    """Run main orchestration function."""
    parser = argparse.ArgumentParser(
        description="Run all data pipelines in the correct order"
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually execute the pipelines (default is dry-run)",
    )
    args = parser.parse_args()

    if not args.execute:
        print("=" * 70)
        print("PIPELINE ORCHESTRATION PLAN (Dry Run)")
        print("=" * 70)
        print()
        print("This script will recreate the graph from scratch:")
        print()
        print("Step 1: Bootstrap Graph")
        print("  - bootstrap_graph.py - Load Domain and Technology nodes from SQLite")
        print()
        print("Step 2: Load Company Data")
        print("  - collect_domains.py - Collect company domains (if needed)")
        print(
            "  - create_company_embeddings.py - Create embeddings for "
            "Company descriptions (if needed)"
        )
        print(
            "  - load_company_data.py - Load Company nodes and HAS_DOMAIN relationships"
        )
        print()
        print("Step 3: Domain Embeddings")
        print(
            "  - create_domain_embeddings.py - Create embeddings for Domain descriptions"
        )
        print("  - compute_domain_similarity.py - Compute Domain-Domain similarity")
        print()
        print("Step 4: Compute GDS Features")
        print("  - compute_gds_features.py - Compute all features:")
        print("    * Technology adoption predictions")
        print("    * Technology affinity/bundling")
        print("    * Company description similarity")
        print()
        print("=" * 70)
        print("To execute, run: python scripts/run_all_pipelines.py --execute")
        print("=" * 70)
        return

    # Execute mode
    print("=" * 70)
    print("RUNNING ALL PIPELINES")
    print("=" * 70)
    print()

    # Step 1: Bootstrap Graph (Domain + Technology nodes)
    print("\n" + "=" * 70)
    print("STEP 1: Bootstrap Graph")
    print("=" * 70)

    if not run_script(
        BOOTSTRAP_SCRIPT,
        execute=True,
        description="Loading Domain and Technology nodes from SQLite",
    ):
        print("\n✗ Failed at bootstrap step")
        return

    # Step 2: Company Data
    print("\n" + "=" * 70)
    print("STEP 2: Load Company Data")
    print("=" * 70)

    # Check if company data files exist
    companies_file = Path("data/public_company_domains.json")
    embeddings_file = Path("data/description_embeddings.json")

    if not companies_file.exists():
        print(f"⚠ Companies file not found: {companies_file}")
        print("  Running collect_domains.py...")
        if not run_script(
            COLLECT_DOMAINS_SCRIPT,
            execute=True,
            description="Step 2.1: Collect Company Domains",
        ):
            print("\n✗ Pipeline 2 failed at collect_domains step")
            return
    else:
        print(
            f"✓ Companies file exists: {companies_file} (skipping collect_domains.py)"
        )

    if not embeddings_file.exists():
        print(f"⚠ Embeddings file not found: {embeddings_file}")
        print("  Running create_company_embeddings.py...")
        if not run_script(
            CREATE_COMPANY_EMBEDDINGS_SCRIPT,
            execute=True,
            description="Step 2.2: Create Company Description Embeddings",
        ):
            print("\n✗ Pipeline 2 failed at create_embeddings step")
            return
    else:
        msg = f"✓ Embeddings file exists: {embeddings_file}"
        msg += " (skipping create_company_embeddings.py)"
        print(msg)

    if not run_script(
        LOAD_COMPANY_DATA_SCRIPT,
        execute=True,
        description="Loading Company nodes and HAS_DOMAIN relationships",
    ):
        print("\n✗ Failed at load_company_data step")
        return

    # Step 3: Domain Embeddings
    print("\n" + "=" * 70)
    print("STEP 3: Domain Embeddings")
    print("=" * 70)

    # Check if domain embeddings cache exists
    domain_embeddings_cache = Path("data/domain_embeddings_cache.json")
    if not domain_embeddings_cache.exists():
        print(f"⚠ Domain embeddings cache not found: {domain_embeddings_cache}")
        print("  Running create_domain_embeddings.py...")
        if not run_script(
            CREATE_DOMAIN_EMBEDDINGS_SCRIPT,
            execute=True,
            description="Step 3.1: Create Domain Description Embeddings",
        ):
            print("\n✗ Failed at create_domain_embeddings step")
            return
    else:
        print(f"✓ Domain embeddings cache exists: {domain_embeddings_cache}")
        print("  Running create_domain_embeddings.py to update Neo4j...")
        if not run_script(
            CREATE_DOMAIN_EMBEDDINGS_SCRIPT,
            execute=True,
            description="Step 3.1: Update Domain Embeddings in Neo4j",
        ):
            print("\n✗ Failed at create_domain_embeddings step")
            return

    if not run_script(
        COMPUTE_DOMAIN_SIMILARITY_SCRIPT,
        execute=True,
        description="Step 3.2: Compute Domain-Domain Similarity",
    ):
        print("\n✗ Failed at compute_domain_similarity step")
        return

    # Step 4: Compute all GDS features (tech + company similarity in one pass)
    print("\n" + "=" * 70)
    print("STEP 4: Compute GDS Features")
    print("=" * 70)
    print(
        "Computing all GDS features: Technology adoption, affinity, and company similarity"
    )

    if not run_script(
        COMPUTE_GDS_SCRIPT,
        execute=True,
        description="Computing all GDS features (tech adoption, affinity, company similarity)",
    ):
        print("\n✗ Failed at GDS computation step")
        return

    # Summary
    print("\n" + "=" * 70)
    print("ALL PIPELINES COMPLETE!")
    print("=" * 70)
    print()
    print("Graph is now ready for queries with:")
    print("  ✓ Domain nodes with title/keywords/description metadata")
    print("  ✓ Domain nodes with description embeddings")
    print("  ✓ Technology nodes")
    print("  ✓ USES relationships (Domain → Technology)")
    print("  ✓ LIKELY_TO_ADOPT relationships (Domain → Technology)")
    print("  ✓ CO_OCCURS_WITH relationships (Technology → Technology)")
    print("  ✓ SIMILAR_DESCRIPTION relationships (Domain → Domain)")
    print("  ✓ Company nodes with description embeddings")
    print("  ✓ HAS_DOMAIN relationships (Company → Domain)")
    print("  ✓ SIMILAR_DESCRIPTION relationships (Company → Company)")
    print()


if __name__ == "__main__":
    main()
