#!/bin/bash
# rf_dump with 4 recurrent rollback snapshots per sequence (n_rs_seq = 4), for L-B's harness --dump
exec "$(dirname "$0")/rf_dump_rs" "$@" --n-rs-seq 4
