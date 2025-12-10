#!/usr/bin/env python3
"""
Orchestration script to run all data pipelines in the correct order.

This script runs:
1. Pipeline 1: Domain → Technology
   (bootstrap_graph.py + compute_gds_features.py)
2. Pipeline 2: Company Data
   (collect_domains.py + create_description_embeddings.py + load_company_data.py)
3. Pipeline 3: Company Similarity
   (compute_gds_features.py - company similarity step)

Dependencies:
- Pipeline 1 must run first (creates Domain and Technology nodes)
- Pipeline 2 depends on Pipeline 1 (needs Domain nodes for HAS_DOMAIN relationships)
- Pipeline 3 depends on Pipeline 2 (needs Company nodes with embeddings)

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
CREATE_EMBEDDINGS_SCRIPT = SCRIPT_DIR / "create_description_embeddings.py"
LOAD_COMPANY_DATA_SCRIPT = SCRIPT_DIR / "load_company_data.py"


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
    parser = argparse.ArgumentParser(description="Run all data pipelines in the correct order")
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
        print("This script will run the following pipelines in order:")
        print()
        print("Pipeline 1: Domain → Technology")
        print("  1. bootstrap_graph.py - Load Domain and Technology nodes from SQLite")
        print("  2. compute_gds_features.py - Compute Technology adoption and affinity")
        print()
        print("Pipeline 2: Company Data")
        print("  3. collect_domains.py - Collect company domains (if needed)")
        print("  4. create_description_embeddings.py - Create embeddings (if needed)")
        print("  5. load_company_data.py - Load Company nodes and HAS_DOMAIN relationships")
        print()
        print("Pipeline 3: Company Similarity")
        print("  6. compute_gds_features.py - Compute Company description similarity")
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

    # Pipeline 1: Domain → Technology
    print("\n" + "=" * 70)
    print("PIPELINE 1: Domain → Technology")
    print("=" * 70)

    if not run_script(
        BOOTSTRAP_SCRIPT,
        execute=True,
        description="Step 1.1: Bootstrap Graph (Domain + Technology nodes)",
    ):
        print("\n✗ Pipeline 1 failed at bootstrap step")
        return

    if not run_script(
        COMPUTE_GDS_SCRIPT,
        execute=True,
        description="Step 1.2: Compute GDS Features (Technology adoption + affinity)",
    ):
        print("\n✗ Pipeline 1 failed at GDS computation step")
        return

    # Pipeline 2: Company Data
    print("\n" + "=" * 70)
    print("PIPELINE 2: Company Data")
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
        print(f"✓ Companies file exists: {companies_file} (skipping collect_domains.py)")

    if not embeddings_file.exists():
        print(f"⚠ Embeddings file not found: {embeddings_file}")
        print("  Running create_description_embeddings.py...")
        if not run_script(
            CREATE_EMBEDDINGS_SCRIPT,
            execute=True,
            description="Step 2.2: Create Description Embeddings",
        ):
            print("\n✗ Pipeline 2 failed at create_embeddings step")
            return
    else:
        msg = f"✓ Embeddings file exists: {embeddings_file}"
        msg += " (skipping create_description_embeddings.py)"
        print(msg)

    if not run_script(
        LOAD_COMPANY_DATA_SCRIPT,
        execute=True,
        description="Step 2.3: Load Company Data (nodes + HAS_DOMAIN relationships)",
    ):
        print("\n✗ Pipeline 2 failed at load_company_data step")
        return

    # Pipeline 3: Compute Company Similarity (runs as part of compute_gds_features.py)
    print("\n" + "=" * 70)
    print("PIPELINE 3: Company Similarity")
    print("=" * 70)
    print("Note: Company similarity is computed as part of compute_gds_features.py")
    print("      Running it again to compute company similarity now that companies are loaded...")

    if not run_script(
        COMPUTE_GDS_SCRIPT,
        execute=True,
        description="Step 3: Compute Company Description Similarity",
    ):
        print("\n✗ Pipeline 3 failed at company similarity step")
        return

    # Summary
    print("\n" + "=" * 70)
    print("ALL PIPELINES COMPLETE!")
    print("=" * 70)
    print()
    print("Graph is now ready for queries with:")
    print("  ✓ Domain nodes and Technology nodes")
    print("  ✓ USES relationships (Domain → Technology)")
    print("  ✓ LIKELY_TO_ADOPT relationships (Domain → Technology)")
    print("  ✓ CO_OCCURS_WITH relationships (Technology → Technology)")
    print("  ✓ Company nodes with description embeddings")
    print("  ✓ HAS_DOMAIN relationships (Company → Domain)")
    print("  ✓ SIMILAR_DESCRIPTION relationships (Company → Company)")
    print()


if __name__ == "__main__":
    main()
