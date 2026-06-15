# StockNetV2

`StockNetV2` is the new backend-first T1 mainline for theme discovery quality research.

## Purpose

- consume the existing `stocknet-research` data estate
- produce 5-minute theme-discovery snapshots
- persist graph, community, theme, semantic, lifecycle, and audit outputs into DuckDB
- power a read-only observer through Node query endpoints

## Runtime model

- Python: offline batch execution only
- DuckDB: system-of-record for T1 outputs
- Node: only online service
- Frontend: read-only observer

## Status

This project is being bootstrapped from the approved T1 design on 2026-06-15.
