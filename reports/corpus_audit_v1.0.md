# Canonical Corpus Audit v1.0

## 1. Purpose
This document provides the official audit for the repair-trajectory corpus used in NSP Paper 1.

## 5. Total discovered runs
**1184** runs were discovered in the raw results file.

## 6. Canonical runs
**1054** runs met all criteria and are included in the Canonical Core.

## 7. Excluded runs by reason
- Setup_Failure: 130

## 12. Patch availability
Completeness metrics are available for **480** runs.

## 16. Known limitations
- Trace files (JSONL) are currently stored on the remote execution server and are not present in the local Windows workspace. Trace validation was performed based on CSV metadata rather than raw file parsing.
