<agents_guidance>
  <global_rule>
    Fail fast on all critical preconditions (SQLite schema, Neo4j connectivity, constraints/indexes, dependency versions). No silent fallbacks.
  </global_rule>

  <neo4j_usage>
    <rule>Always use context managers for Driver/Session.</rule>
    <rule>Ensure constraints/indexes exist before any writes; abort if missing.</rule>
    <rule>Use MERGE for nodes/relationships; runs must be idempotent and re-runnable.</rule>
    <rule>No implicit data mutations; migrations/scripts are explicit and versioned.</rule>
  </neo4j_usage>

  <data_validation>
    <required_tables>url_status</required_tables>
    <recommended_tables>url_technologies, url_nameservers, url_mx_records, url_geoip</recommended_tables>
    <rule>Verify table presence + row counts before load; normalize key strings (lowercase hostnames/domains).</rule>
  </data_validation>

  <graph_schema>
    <nodes>
      Domain(key=final_domain)
      Technology(key=name, props=category)
      Nameserver(key=nameserver)
      MailServer(key=host)
      IP(key=address, props=geo fields)
      ASN(key=number, props=org)
      CertificateAuthority(key=name)
      SocialAccount(key=platform+identifier)
    </nodes>
    <relationships>
      (:Domain)-[:USES]->(:Technology)
      (:Domain)-[:HAS_NAMESERVER]->(:Nameserver)
      (:Domain)-[:HAS_MX {priority}]->(:MailServer)
      (:Domain)-[:HOSTED_ON]->(:IP)
      (:IP)-[:BELONGS_TO_ASN]->(:ASN)
      (:Domain)-[:SECURED_BY]->(:CertificateAuthority)
      (:Domain)-[:REDIRECTS_TO {order}]->(:Domain)
      (:Domain)-[:HAS_SOCIAL]->(:SocialAccount)
    </relationships>
    <properties>
      Keep raw page/TLS/DNS fields on Domain (status, tls_version, cipher_suite, key_algorithm, ssl_* dates, response_time, spf_record, dmarc_record, is_mobile_friendly).
      Stamp writes with loaded_at (UTC) and source="domain_status".
    </properties>
  </graph_schema>

  <constraints_and_indexes>
    CONSTRAINT unique_domain IF NOT EXISTS ON (d:Domain) ASSERT d.final_domain IS UNIQUE
    CONSTRAINT unique_tech IF NOT EXISTS ON (t:Technology) ASSERT t.name IS UNIQUE
    CONSTRAINT unique_ns IF NOT EXISTS ON (n:Nameserver) ASSERT n.nameserver IS UNIQUE
    CONSTRAINT unique_mx IF NOT EXISTS ON (m:MailServer) ASSERT m.host IS UNIQUE
    CONSTRAINT unique_ip IF NOT EXISTS ON (i:IP) ASSERT i.address IS UNIQUE
    CONSTRAINT unique_asn IF NOT EXISTS ON (a:ASN) ASSERT a.number IS UNIQUE
    CONSTRAINT unique_ca IF NOT EXISTS ON (c:CertificateAuthority) ASSERT c.name IS UNIQUE
    CONSTRAINT unique_social IF NOT EXISTS ON (s:SocialAccount) ASSERT (s.platform, s.identifier) IS UNIQUE
  </constraints_and_indexes>

  <ingest_pipeline>
    <stage0>Dry-run: connect, check constraints, count tables, print plan; exit unless --execute.</stage0>
    <stage1>Load core nodes with UNWIND batching (e.g., 1000); normalize keys; MERGE only.</stage1>
    <stage2>Load relationships (USES, HAS_NS, HAS_MX{priority}, HOSTED_ON, BELONGS_TO_ASN, SECURED_BY, REDIRECTS_TO{order}, HAS_SOCIAL).</stage2>
    <stage3>Post-load sanity checks (counts, degrees) and summary log.</stage3>
    <batching>Use APOC Core (apoc.periodic.iterate) for large batches; abort on errors &gt;0.</batching>
  </ingest_pipeline>

  <gds_usage>
    <allowed>WCC, Louvain, Node Similarity (Domain↔Technology projection), PageRank</allowed>
    <rules>Create named in-memory graphs (e.g., "ds_main"); check memory; drop graphs after use; log timings.</rules>
  </gds_usage>

  <development_setup>
    <env>
      Always use the dedicated conda env: domain_status_graph (Python 3.13)
      conda activate domain_status_graph
      pip install -e .[dev]
      pip install -r requirements.txt
      cp .env.sample .env  # set NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
    </env>
    <quality_tools>
      isort --profile black src/ tests/ scripts/
      black src/ tests/ scripts/
      flake8 src/ tests/ --max-line-length=100
      mypy src/ --ignore-missing-imports
      pytest -q
      pre-commit run --all-files  # must pass locally and in CI
    </quality_tools>
  </development_setup>

  <testing>
    <rule>Integration tests for constraints and ingest; unit tests for helpers.</rule>
    <rule>Idempotency test: running importer twice yields identical counts.</rule>
  </testing>

  <ci_zero_tolerance>
    <golden_rule>pre-commit run --all-files must show "Passed" for all checks.</golden_rule>
  </ci_zero_tolerance>

  <architecture_overview>
    <entry file="scripts/bootstrap_graph.py" desc="Entry point: dry-run plan, then execute ingest"/>
    <entry file="src/schema/constraints.py" desc="ensure_constraints_exist_or_fail()"/>
    <entry file="src/ingest/sqlite_readers.py" desc="Typed readers & normalizers for SQLite tables"/>
    <entry file="src/ingest/loaders.py" desc="Batch MERGE for nodes/relationships"/>
    <entry file="src/gds/pipelines.py" desc="Graph projections & algorithm helpers"/>
    <entry file="src/constants.py" desc="Batch sizes, property keys, graph names"/>
    <entry file="queries/money/" desc="Saved Cypher demos for the workshop"/>
  </architecture_overview>
</agents_guidance>
