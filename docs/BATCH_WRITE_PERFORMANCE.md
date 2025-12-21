# Batch Write Performance Guide

This document explains how large relationship writes are optimized for maximum performance and reliability.

## Overview

The project uses **optimized UNWIND batching** for large relationship writes (10M+ relationships). This approach is:
- **Reliable**: Works consistently across all systems
- **Performant**: Typically 1-2M relationships/minute
- **Simple**: No complex dependencies or configuration
- **Robust**: No transaction timeouts or connection issues

## Implementation

The system uses Neo4j's native `UNWIND` batching with:
- **Batch Size**: 5,000 relationships per transaction (configurable via `BATCH_SIZE_LARGE`)
- **Progress Reporting**: Updates every 100 batches with ETA
- **Single Session**: Reuses connection for better performance
- **MERGE Operations**: Idempotent writes (safe to re-run)

## Performance Expectations

**For 10M relationships:**
- **Time**: ~5-10 minutes (depending on system)
- **Throughput**: ~1-2M relationships/minute
- **Memory**: Low (processes in small batches)
- **Reliability**: High (no transaction timeouts)

## Configuration

The batch size is controlled by `BATCH_SIZE_LARGE` in `domain_status_graph/constants.py`:

```python
BATCH_SIZE_LARGE = 5000  # Relationships per transaction
```

**Tuning Guidelines:**
- **Smaller batches (1K-5K)**: More reliable, less memory, slower
- **Larger batches (5K-10K)**: Faster, more memory, may timeout on slower systems
- **Default (5K)**: Good balance for most systems

## Manual Override

To adjust batch size, modify `domain_status_graph/constants.py`:

```python
BATCH_SIZE_LARGE = 10000  # Increase for faster writes (if system can handle it)
```

## Troubleshooting

**If you see transaction timeouts:**
- Reduce `BATCH_SIZE_LARGE` (try 2,000 or 1,000)
- Check Neo4j transaction timeout settings

**If performance is slower than expected:**
- Check Neo4j heap memory settings (should be 4GB+ for large datasets)
- Verify indexes exist on `Company.cik` property
- Check for disk I/O bottlenecks (SSD recommended)

**If you see connection errors:**
- The code automatically retries with smaller batches
- Check Neo4j connection pool settings
- Verify network stability

## Why Not APOC?

We previously used APOC's `apoc.periodic.iterate` for parallel processing, but found it:
- **Unreliable**: Transaction timeouts on long-running operations
- **Complex**: Requires temporary nodes and cleanup logic
- **Fragile**: Connection issues cause partial failures
- **Unnecessary**: UNWIND batching is fast enough (1-2M/min) and more reliable

The current approach prioritizes **reliability and simplicity** over marginal performance gains.

## References

- [Neo4j UNWIND Performance](https://neo4j.com/docs/cypher-manual/current/clauses/unwind/)
- [Neo4j Performance Tuning Guide](https://neo4j.com/docs/operations-manual/current/performance/)
