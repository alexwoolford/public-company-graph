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
ENRICH_COMPANY_PROPERTIES_SCRIPT = SCRIPT_DIR / "enrich_company_properties.py"
COMPUTE_COMPANY_SIMILARITY_SCRIPT = SCRIPT_DIR / "compute_company_similarity.py"
CREATE_DOMAIN_EMBEDDINGS_SCRIPT = SCRIPT_DIR / "create_domain_embeddings.py"
COMPUTE_DOMAIN_SIMILARITY_SCRIPT = SCRIPT_DIR / "compute_domain_similarity.py"
COMPUTE_KEYWORD_SIMILARITY_SCRIPT = SCRIPT_DIR / "compute_keyword_similarity.py"


def run_script(
    script_path: Path,
    execute: bool = False,
    description: str = "",
    extra_args: list = None,
):
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
    if extra_args:
        cmd.extend(extra_args)

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
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Fast mode: skip uncached companies in collect_domains.py",
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
        print("  - load_company_data.py - Load Company nodes and HAS_DOMAIN relationships")
        print("  - enrich_company_properties.py - Enrich Company nodes with properties")
        print("  - compute_company_similarity.py - Create SIMILAR_INDUSTRY and SIMILAR_SIZE")
        print()
        print("Step 3: Domain Embeddings & Similarity")
        print("  - create_domain_embeddings.py - Create embeddings for Domain descriptions")
        print("  - compute_domain_similarity.py - Compute Domain-Domain description similarity")
        print("  - compute_keyword_similarity.py - Compute Domain-Domain keyword similarity")
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

    # Check cache for company data
    from domain_status_graph.cache import get_cache

    cache = get_cache()
    cached_companies = cache.count("company_domains")

    # Run collect_domains.py if needed
    if cached_companies == 0 and not args.fast:
        print(f"⚠ No companies in cache")
        print("  Running collect_domains.py...")
        if not run_script(
            COLLECT_DOMAINS_SCRIPT,
            execute=True,
            description="Step 2.1: Collect Company Domains",
        ):
            print("\n✗ Pipeline 2 failed at collect_domains step")
            return
    elif cached_companies > 0:
        if args.fast:
            print(
                f"✓ {cached_companies} companies in cache (fast mode: skipping collect_domains.py)"
            )
        else:
            print(f"✓ {cached_companies} companies in cache")
            print("  Running collect_domains.py with --skip-uncached to check for new companies...")
            if not run_script(
                COLLECT_DOMAINS_SCRIPT,
                execute=True,
                description="Step 2.1: Collect Company Domains",
                extra_args=["--skip-uncached"],
            ):
                print("\n✗ Pipeline 2 failed at collect_domains step")
                return
    else:
        # cached_companies == 0 and args.fast
        print(f"⚠ No companies in cache, but --fast mode enabled")
        print("  Skipping collect_domains.py (use without --fast to fetch companies)")

    # Load Company nodes first (needed before creating embeddings)
    if not run_script(
        LOAD_COMPANY_DATA_SCRIPT,
        execute=True,
        description="Step 2.2: Loading Company nodes and HAS_DOMAIN relationships",
    ):
        print("\n✗ Failed at load_company_data step")
        return

    # Enrich Company nodes with properties (SEC, Yahoo Finance, etc.)
    if not run_script(
        ENRICH_COMPANY_PROPERTIES_SCRIPT,
        execute=True,
        description="Step 2.3: Enrich Company Properties (Industry, Size, etc.)",
    ):
        print("\n✗ Failed at enrich_company_properties step")
        return

    # Compute company similarity relationships
    if not run_script(
        COMPUTE_COMPANY_SIMILARITY_SCRIPT,
        execute=True,
        description="Step 2.4: Compute Company Similarity (Industry & Size)",
    ):
        print("\n✗ Failed at compute_company_similarity step")
        return

    # Then create embeddings for the Company nodes
    if not run_script(
        CREATE_COMPANY_EMBEDDINGS_SCRIPT,
        execute=True,
        description="Step 2.5: Create Company Description Embeddings",
    ):
        print("\n✗ Pipeline 2 failed at create_embeddings step")
        return

    # Step 3: Domain Embeddings
    print("\n" + "=" * 70)
    print("STEP 3: Domain Embeddings")
    print("=" * 70)

    # Always run create_domain_embeddings (it checks cache internally)
    if not run_script(
        CREATE_DOMAIN_EMBEDDINGS_SCRIPT,
        execute=True,
        description="Step 3.1: Create Domain Description Embeddings",
    ):
        print("\n✗ Failed at create_domain_embeddings step")
        return

    if not run_script(
        COMPUTE_DOMAIN_SIMILARITY_SCRIPT,
        execute=True,
        description="Step 3.2: Compute Domain-Domain Description Similarity",
    ):
        print("\n✗ Failed at compute_domain_similarity step")
        return

    if not run_script(
        COMPUTE_KEYWORD_SIMILARITY_SCRIPT,
        execute=True,
        description="Step 3.3: Compute Domain-Domain Keyword Similarity",
    ):
        print("\n✗ Failed at compute_keyword_similarity step")
        return

    # Step 4: Compute all GDS features (tech + company similarity in one pass)
    print("\n" + "=" * 70)
    print("STEP 4: Compute GDS Features")
    print("=" * 70)
    print("Computing all GDS features: Technology adoption, affinity, and company similarity")

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
    print("  ✓ SIMILAR_KEYWORD relationships (Domain → Domain)")
    print("  ✓ Company nodes with description embeddings")
    print("  ✓ HAS_DOMAIN relationships (Company → Domain)")
    print("  ✓ SIMILAR_DESCRIPTION relationships (Company → Company)")
    print()


if __name__ == "__main__":
    main()
