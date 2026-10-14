#!/bin/sh
# The crawl's choices, one run after another (they would share the network if run together).
cd "$(dirname "$0")"
TILE=1024 TILES=5:8,2:5 CACHE=16 REUSE=1 node run_sql.mjs crawl.py > crawl_1024_reuse.log 2>&1
TILE=1024 TILES=5:8,2:5 CACHE=16 REUSE=0 node run_sql.mjs crawl.py > crawl_1024_noreuse.log 2>&1
TILE=512 TILES=5:7,2:4 CACHE=16 REUSE=1 node run_sql.mjs crawl.py > crawl_512.log 2>&1
TILE=1024 TILES=5:7,2:4 CACHE=2 REUSE=1 node run_sql.mjs crawl.py > crawl_1024_2x2_cache2.log 2>&1
echo all done > crawl_runs.done
