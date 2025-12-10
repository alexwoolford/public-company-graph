# Complete Setup Guide: End-to-End Process

This guide provides a **complete, repeatable process** to recreate the Domain Status Graph from scratch. Follow these steps in order.

## Prerequisites Checklist

Before starting, ensure you have:

- [ ] **Neo4j 5.x or later** installed and running
- [ ] **Neo4j Graph Data Science (GDS) library** installed and enabled
- [ ] **Python 3.13+** (or use conda environment)
- [ ] **SQLite database** at `data/domain_status.db` (source data - included in repository, or generate your own with [`domain_status`](https://github.com/alexwoolford/domain_status))
- [ ] **Network access** to Neo4j (default: `bolt://localhost:7687`)

### Verify Prerequisites

```bash
# Check Python version
python3 --version  # Should be 3.13+

# Check Neo4j is running
# In Neo4j Browser or via cypher-shell:
# RETURN 1 as test;

# Check GDS is installed
# In Neo4j Browser:
# CALL gds.version() YIELD version RETURN version;
```

---

## Step 1: Clone and Navigate

```bash
git clone <repository-url>
cd domain_status_graph
```

**Verify**: You should see `scripts/`, `docs/`, and `data/` directories.

---

## Step 2: Set Up Python Environment

### Option A: Use Conda (Recommended)

```bash
# Create and activate conda environment
conda create -n domain_status_graph python=3.13
conda activate domain_status_graph

# Install dependencies
pip install -r requirements.txt
```

### Option B: Use System Python

```bash
# Install dependencies
pip install -r requirements.txt
```

**Verify**:
```bash
python3 -c "import neo4j; import graphdatascience; import dotenv; print('✓ All packages installed')"
```

---

## Step 3: Configure Neo4j Connection

```bash
# Copy example environment file
cp .env.sample .env

# Edit .env with your Neo4j credentials
# Required variables:
# NEO4J_URI=bolt://localhost:7687
# NEO4J_USER=neo4j
# NEO4J_PASSWORD=your_password_here
# NEO4J_DATABASE=domain
```

**Verify**:
```bash
python3 -c "
from dotenv import load_dotenv
import os
load_dotenv()
uri = os.getenv('NEO4J_URI')
user = os.getenv('NEO4J_USER')
password = os.getenv('NEO4J_PASSWORD')
database = os.getenv('NEO4J_DATABASE', 'domain')
print(f'✓ URI: {uri}')
print(f'✓ User: {user}')
print(f'✓ Password: {\"*\" * len(password) if password else \"NOT SET\"}')
print(f'✓ Database: {database}')
"
```

---

## Step 4: Verify Source Data

```bash
# Check that SQLite database exists
ls -lh data/domain_status.db

# Verify it has data
python3 -c "
import sqlite3
conn = sqlite3.connect('data/domain_status.db')
cursor = conn.cursor()
cursor.execute('SELECT COUNT(*) FROM url_status')
domains = cursor.fetchone()[0]
cursor.execute('SELECT COUNT(*) FROM url_technologies')
techs = cursor.fetchone()[0]
print(f'✓ Domains in database: {domains}')
print(f'✓ Technology records: {techs}')
conn.close()
"
```

**Expected**: You should see domain and technology counts > 0.

---

## Step 5: Clean Neo4j Database (Fresh Start)

**⚠️ WARNING**: This will delete all existing data in the target database.

```bash
# Connect to Neo4j and run:
# MATCH (n) DETACH DELETE n;
# Or use cypher-shell:
echo "MATCH (n) DETACH DELETE n;" | cypher-shell -u neo4j -p your_password -d domain
```

**Verify**:
```bash
python3 -c "
from neo4j import GraphDatabase
from dotenv import load_dotenv
import os
load_dotenv()
driver = GraphDatabase.driver(
    os.getenv('NEO4J_URI', 'bolt://localhost:7687'),
    auth=(os.getenv('NEO4J_USER', 'neo4j'), os.getenv('NEO4J_PASSWORD', ''))
)
with driver.session(database=os.getenv('NEO4J_DATABASE', 'domain')) as session:
    result = session.run('MATCH (n) RETURN count(n) AS count')
    count = result.single()['count']
    print(f'✓ Nodes in database: {count} (should be 0 for fresh start)')
driver.close()
"
```

---

## Step 6: Bootstrap Graph from SQLite

### 6a. Dry Run (Review Plan)

```bash
python scripts/bootstrap_graph.py
```

**Expected Output**: You should see:
- Connection test
- Table counts from SQLite
- Plan showing what will be loaded
- Summary of nodes and relationships to create

**Verify**: Review the plan to ensure it matches expectations.

### 6b. Execute Bootstrap

```bash
python scripts/bootstrap_graph.py --execute
```

**Expected Output**: You should see:
- ✓ Connected to Neo4j
- Loading domains...
- Loading technologies...
- Creating USES relationships...
- Summary with counts

**Verify**:
```bash
python3 -c "
from neo4j import GraphDatabase
from dotenv import load_dotenv
import os
load_dotenv()
driver = GraphDatabase.driver(
    os.getenv('NEO4J_URI', 'bolt://localhost:7687'),
    auth=(os.getenv('NEO4J_USER', 'neo4j'), os.getenv('NEO4J_PASSWORD', ''))
)
with driver.session(database=os.getenv('NEO4J_DATABASE', 'domain')) as session:
    # Check Domain nodes
    result = session.run('MATCH (d:Domain) RETURN count(d) AS count')
    domains = result.single()['count']
    print(f'✓ Domain nodes: {domains}')

    # Check Technology nodes
    result = session.run('MATCH (t:Technology) RETURN count(t) AS count')
    techs = result.single()['count']
    print(f'✓ Technology nodes: {techs}')

    # Check USES relationships
    result = session.run('MATCH ()-[r:USES]->() RETURN count(r) AS count')
    uses = result.single()['count']
    print(f'✓ USES relationships: {uses}')

    # Check constraints exist
    result = session.run('SHOW CONSTRAINTS')
    constraints = list(result)
    print(f'✓ Constraints: {len(constraints)}')
driver.close()
"
```

**Expected**:
- Domain nodes: ~5,000-6,000
- Technology nodes: ~500-600
- USES relationships: ~30,000-40,000
- Constraints: 2 (Domain.final_domain, Technology.name)

---

## Step 7: Compute GDS Features

### 7a. Dry Run (Review Plan)

```bash
python scripts/compute_gds_features.py
```

**Expected Output**: You should see:
- GDS FEATURES PLAN (Dry Run)
- List of features to compute
- Instructions to run with --execute

**Verify**: Review the plan to ensure it matches expectations.

### 7b. Execute GDS Computation

```bash
python scripts/compute_gds_features.py --execute
```

**Expected Output**: You should see:
- ✓ Connected to Neo4j
- Cleaning up leftover graph projections...
- Computing GDS Features
- 1. Technology Adopter Prediction (Technology → Domain)
  - Creating graph...
  - Processing technologies...
  - Progress updates...
- 2. Technology Affinity and Bundling
  - Creating graph...
  - Computing Node Similarity...
- Summary with counts

**This may take 10-30 minutes** depending on your data size and hardware.

**Verify**:
```bash
python3 -c "
from neo4j import GraphDatabase
from dotenv import load_dotenv
import os
load_dotenv()
driver = GraphDatabase.driver(
    os.getenv('NEO4J_URI', 'bolt://localhost:7687'),
    auth=(os.getenv('NEO4J_USER', 'neo4j'), os.getenv('NEO4J_PASSWORD', ''))
)
with driver.session(database=os.getenv('NEO4J_DATABASE', 'domain')) as session:
    # Check LIKELY_TO_ADOPT relationships
    result = session.run('MATCH ()-[r:LIKELY_TO_ADOPT]->() RETURN count(r) AS count')
    adopt = result.single()['count']
    print(f'✓ LIKELY_TO_ADOPT relationships: {adopt}')

    # Check CO_OCCURS_WITH relationships
    result = session.run('MATCH ()-[r:CO_OCCURS_WITH]->() RETURN count(r) AS count')
    cooccurs = result.single()['count']
    print(f'✓ CO_OCCURS_WITH relationships: {cooccurs}')

    # Sample query: Find domains likely to adopt a technology
    result = session.run('''
        MATCH (t:Technology)<-[r:LIKELY_TO_ADOPT]-(d:Domain)
        WITH t, count(r) AS adopter_count
        WHERE adopter_count > 0
        RETURN t.name, adopter_count
        ORDER BY adopter_count DESC
        LIMIT 5
    ''')
    print('\\n✓ Sample: Technologies with most predicted adopters:')
    for record in result:
        print(f'  - {record[\"t.name\"]}: {record[\"adopter_count\"]} adopters')
driver.close()
"
```

**Expected**:
- LIKELY_TO_ADOPT relationships: > 0 (varies by data)
- CO_OCCURS_WITH relationships: > 0 (varies by data)
- Sample query should return technologies with adopters

---

## Step 8: Verify End-to-End

### 8a. Manual Verification Queries (Recommended)

Run these verification queries to ensure everything works:

### 8a. Basic Graph Structure

```cypher
// Check node counts
MATCH (d:Domain) RETURN count(d) AS domains;
MATCH (t:Technology) RETURN count(t) AS technologies;

// Check relationship counts
MATCH ()-[r:USES]->() RETURN count(r) AS uses;
MATCH ()-[r:LIKELY_TO_ADOPT]->() RETURN count(r) AS adoptions;
MATCH ()-[r:CO_OCCURS_WITH]->() RETURN count(r) AS co_occurs;
```

### 8c. Technology Adopter Prediction

```cypher
// Find domains likely to adopt a specific technology
MATCH (t:Technology {name: 'Shopify'})<-[r:LIKELY_TO_ADOPT]-(d:Domain)
RETURN d.final_domain, r.score
ORDER BY r.score DESC
LIMIT 10;
```

**Expected**: Should return domains with scores > 0.

### 8d. Technology Affinity

```cypher
// Find technologies that co-occur with a specific technology
MATCH (t1:Technology {name: 'WordPress'})-[r:CO_OCCURS_WITH]->(t2:Technology)
RETURN t2.name, r.similarity
ORDER BY r.similarity DESC
LIMIT 10;
```

**Expected**: Should return technologies with similarity scores > 0.

---

## Troubleshooting

### Issue: "Neo4j connection failed"

**Solution**:
1. Verify Neo4j is running: `neo4j status` (or check service)
2. Check `.env` file has correct credentials
3. Test connection: `cypher-shell -u neo4j -p password`

### Issue: "GDS library not found"

**Solution**:
1. Verify GDS is installed: `CALL gds.version()`
2. Check Neo4j version (GDS requires Neo4j 5.x+)
3. Install GDS plugin if missing

### Issue: "SQLite database not found"

**Solution**:
1. Verify `data/domain_status.db` exists
2. Check file permissions
3. Verify database has data (see Step 4)

### Issue: "No LIKELY_TO_ADOPT relationships created"

**Solution**:
1. Check that bootstrap completed successfully
2. Verify technologies exist in graph
3. Check GDS computation logs for errors
4. Ensure GDS library is enabled

### Issue: "Script fails with pandas error"

**Solution**:
1. Ensure you're using the correct Python environment
2. Reinstall dependencies: `pip install -r requirements.txt --force-reinstall`
3. Check Python version: `python3 --version` (should be 3.13+)

---

## Complete Command Sequence (Copy-Paste)

For a completely fresh start, run these commands in order:

```bash
# 1. Navigate to project
cd domain_status_graph

# 2. Set up environment (if using conda)
conda create -n domain_status_graph python=3.13
conda activate domain_status_graph

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure Neo4j
cp .env.sample .env
# Edit .env with your credentials

# 5. Clean Neo4j (if needed)
echo "MATCH (n) DETACH DELETE n;" | cypher-shell -u neo4j -p your_password -d domain

# 6. Bootstrap graph and compute GDS features
# Option A: Run individually
python scripts/bootstrap_graph.py --execute
python scripts/compute_gds_features.py --execute

# Option B: Use orchestration script (runs all pipelines in order)
# python scripts/run_all_pipelines.py --execute

# 8. Verify
python3 -c "
from neo4j import GraphDatabase
from dotenv import load_dotenv
import os
load_dotenv()
driver = GraphDatabase.driver(
    os.getenv('NEO4J_URI', 'bolt://localhost:7687'),
    auth=(os.getenv('NEO4J_USER', 'neo4j'), os.getenv('NEO4J_PASSWORD', ''))
)
with driver.session(database=os.getenv('NEO4J_DATABASE', 'domain')) as session:
    result = session.run('MATCH (d:Domain) RETURN count(d) AS count')
    print(f'Domains: {result.single()[\"count\"]}')
    result = session.run('MATCH ()-[r:LIKELY_TO_ADOPT]->() RETURN count(r) AS count')
    print(f'Adoption predictions: {result.single()[\"count\"]}')
driver.close()
"
```

---

## Success Criteria

You have successfully set up the graph if:

- [ ] Domain nodes exist (~5,000-6,000)
- [ ] Technology nodes exist (~500-600)
- [ ] USES relationships exist (~30,000-40,000)
- [ ] LIKELY_TO_ADOPT relationships exist (> 0)
- [ ] CO_OCCURS_WITH relationships exist (> 0)
- [ ] Sample queries return results

---

## Next Steps

Once setup is complete:

1. **Read**: `docs/money_queries.md` - Understand the 2 high-value GDS features
2. **Explore**: `docs/money_queries.md` - High-value business queries
3. **Query**: Use the example queries in `README.md` to explore the graph

**Verification**: The core scripts will fail fast if something is wrong, providing immediate feedback. You can also run the example queries in Step 8 to verify everything is working.

---

## Support

If you encounter issues:

1. Check the troubleshooting section above
2. Review script logs (GDS computation logs to `logs/` directory)
3. Verify all prerequisites are met
4. Check Neo4j logs for errors

---

*Last Updated: 2024-11-30*
