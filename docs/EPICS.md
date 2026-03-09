CAP-1 Dataset Governance

Milestone

CAP-1 Dataset Governance

EPICs

EPIC-1 Dataset Registry

Purpose

Store and manage evaluation datasets.

Scope

• dataset metadata
• dataset registry structure
• dataset discovery

EPIC-2 Dataset Manifest Schema

Purpose

Define dataset metadata format.

Scope

• dataset.yaml schema
• dataset metadata validation
• dataset version tracking

EPIC-3 Dataset Validation Engine

Purpose

Ensure datasets follow required structure.

Scope

• schema validation
• case format validation
• dataset integrity checks

EPIC-4 Dataset Version Resolution

Purpose

Resolve dataset versions deterministically.

Scope

• dataset version selection
• version pinning
• dataset resolution logic

EPIC-5 Dataset Loader System

Purpose

Load datasets into evaluation pipeline.

Scope

• dataset reading
• sampling support
• dataset filtering

EPIC-6 Dataset Governance CLI

Purpose

Manage datasets via CLI.

Scope

• dataset list
• dataset validate
• dataset inspect

CAP-2 Baseline Governance

Milestone

CAP-2 Baseline Governance

EPICs

EPIC-7 Baseline Snapshot System

Purpose

Create baseline snapshots from runs.

Scope

• baseline snapshot creation
• baseline metadata
• snapshot storage

EPIC-8 Baseline Registry

Purpose

Manage baseline versions.

Scope

• baseline listing
• baseline lookup
• baseline metadata

EPIC-9 Baseline Validation Engine

Purpose

Validate baseline integrity.

Scope

• artifact checks
• baseline compatibility
• snapshot validation

EPIC-10 Baseline Diff Engine

Purpose

Compare evaluation runs.

Scope

• decision flips
• score deltas
• flag differences

You already implemented part of this.

EPIC-11 Regression Detection System

Purpose

Detect evaluation regressions.

Scope

• metric drop detection
• regression rules
• failure policies

EPIC-12 CI Baseline Gate

Purpose

Fail CI if evaluation regresses.

Scope

• CI integration
• regression thresholds
• PR gate

CAP-3 Rule Governance

Milestone

CAP-3 Rule Governance

EPICs

EPIC-13 Rule Manifest System

Purpose

Track rule metadata.

Scope

• rules/manifest.yaml
• rule metadata
• rule ownership

EPIC-14 Rule Registry Validation

Purpose

Ensure rules exist in registry.

Scope

• registry validation
• missing rule detection

You implemented this.

EPIC-15 Rule Lifecycle Management

Purpose

Manage rule evolution.

Scope

• rule versioning
• rule lifecycle stages
• rule deprecation

EPIC-16 Rule Testing Framework

Purpose

Ensure rule correctness.

Scope

• rule test datasets
• rule regression tests

EPIC-17 Rule Drift Detection

Purpose

Detect rule behaviour changes.

Scope

• rule output monitoring
• rule drift alerts

EPIC-18 Rule Governance CLI

Purpose

Manage rules via CLI.

Scope

• rule validate
• rule list
• rule inspect

CAP-4 Artifact Governance

Milestone

CAP-4 Artifact Governance

EPICs

EPIC-19 Artifact Schema System

Purpose

Define evaluation artifact schemas.

Scope

• manifest schema
• metrics schema
• judgments schema

EPIC-20 Artifact Validation Engine

Purpose

Validate artifacts produced by runs.

Scope

• artifact structure validation
• missing artifact detection

EPIC-21 Artifact Compatibility Checks

Purpose

Ensure compatibility across versions.

Scope

• schema compatibility
• version compatibility

EPIC-22 Artifact Versioning System

Purpose

Track artifact format versions.

Scope

• artifact schema versions
• version upgrade path

EPIC-23 Artifact Storage Governance

Purpose

Manage run artifacts.

Scope

• artifact directory structure
• retention policy

EPIC-24 Artifact Inspection CLI

Purpose

Inspect evaluation artifacts.

Scope

• artifact summary
• artifact debugging tools

CAP-5 Drift Monitoring

Milestone

CAP-5 Drift Monitoring

EPICs

EPIC-25 Metric Drift Detection

Purpose

Detect metric changes across runs.

Scope

• metric trend detection
• metric change alerts

EPIC-26 Decision Flip Monitoring

Purpose

Detect decision changes.

Scope

• flip detection
• flip reporting

You already partially have this.

EPIC-27 Score Delta Analysis

Purpose

Analyze scoring changes.

Scope

• score deltas
• score variance

EPIC-28 Evaluation Stability Monitoring

Purpose

Ensure evaluation consistency.

Scope

• repeated evaluation stability
• variance tracking

EPIC-29 Evaluation Observability

Purpose

Expose evaluation insights.

Scope

• run statistics
• evaluation summaries

EPIC-30 Evaluation Dashboard

Purpose

Visualize evaluation results.

Scope
• run dashboards
• metric charts
• drift visualizations