# Task: Build a Comorbidity-Distance Evaluation Pipeline for Counterfactual Disease Perturbations

## Objective

Implement an evaluation pipeline that compares two top-`k` prediction sets for each intervention:

- `naive`
- `guided`

Each intervention has:

- one **perturbed disease term / ICD-10 class**
- one top-`k` predicted set from `naive`
- one top-`k` predicted set from `guided`

The pipeline should determine which method produces predictions that are **closer to empirical comorbidity structure**.

Closeness is defined using a **weighted disease graph** derived from public patient-level diagnosis data:

- more frequent co-occurrence => smaller distance
- less frequent co-occurrence => larger distance

The final output is an evaluation table and summary comparing `guided` vs `naive`.

---

## Inputs

### 1. Prediction results table
Expected columns or equivalent fields:

- `example_id`
- `perturbed_term` or `perturbed_icd10_class`
- `naive_topk_terms`
- `guided_topk_terms`

If predictions are already mapped to ICD-10 classes, use them directly. Otherwise map them first.

### 2. MIMIC diagnosis data
Use public MIMIC diagnosis tables as the source for empirical co-occurrence.

Expected usable fields:

- patient identifier
- ICD diagnosis code
- coding version if needed

### 3. Mapping resources
Need mapping artifacts for:

- ICD-10-CM code -> HCUP CCSR category
- ICD-10 code -> ICD-10 class
- prediction term -> ICD-10 code or ICD-10 class

---

## Outputs

### 1. Mapping artifacts
Produce reusable tables:

- `term_to_icd10_class`
- `icd10_code_to_icd10_class`
- `icd10_code_to_ccsr`
- `icd10_class_to_ccsr_distribution`

### 2. Comorbidity graph artifact
A weighted undirected graph on CCSR categories with:

- `ccsr_i`
- `ccsr_j`
- `patient_count_i`
- `patient_count_j`
- `cooccurrence_count_ij`
- `cooccurrence_prob_ij`
- `distance_ij`

### 3. Per-example evaluation table
For each intervention:

- `example_id`
- `perturbed_icd10_class`
- `naive_score`
- `guided_score`
- `delta_guided_minus_naive`
- `winner`
- `naive_effective_k`
- `guided_effective_k`

### 4. Aggregate summary
Include at minimum:

- mean `naive_score`
- mean `guided_score`
- mean paired delta
- median paired delta
- win rate of `guided`
- mapping coverage stats
- optional stratified summaries by ICD-10 chapter or class family

---

## Canonical Representation

### Graph layer
Use **HCUP CCSR categories** as graph nodes.

### Evaluation layer
Use **ICD-10 classes** in the prediction/evaluation interface.

### Bridge
Represent each ICD-10 class as a set or weighted distribution over CCSR categories.

---

## Required Processing Steps

## Step 1: Normalize the prediction results

### Goal
Convert the intervention table into a canonical format with ICD-10-class-level targets and predictions.

### Tasks
1. Parse the perturbation field.
2. Parse both top-`k` prediction fields.
3. Normalize strings if raw terms are provided.
4. Map all terms to ICD-10 codes or directly to ICD-10 classes.
5. Produce canonical columns:

- `perturbed_icd10_class`
- `naive_topk_icd10_classes`
- `guided_topk_icd10_classes`

### Rules
- Deduplicate repeated predicted classes within each top-`k` list.
- Preserve rank order if available.
- Drop unmappable predictions but record coverage.
- Compute effective `k` after filtering.

---

## Step 2: Build the ICD-10 / CCSR mapping layer

### Goal
Create reusable mappings from evaluation-layer classes into graph-layer nodes.

### Tasks

#### 2.1 ICD-10 code -> ICD-10 class
Implement or load a mapping that assigns each ICD-10 code to its class.

#### 2.2 ICD-10-CM code -> CCSR
Load the HCUP CCSR mapping and represent it as a many-to-many relation.

#### 2.3 ICD-10 class -> CCSR distribution
For each ICD-10 class:

1. collect all ICD-10 codes belonging to that class
2. map those codes to CCSR categories
3. aggregate into a class-level representation

Produce two variants:

- **set representation**: class -> unique CCSR set
- **weighted representation**: class -> CCSR weights

### Weighting rule
Default to prevalence-based weights from MIMIC:

\[
w(c \mid C) = \frac{\text{count of patients with CCSR } c \text{ among codes in class } C}{\sum_{c' \in S(C)} \text{count of patients with CCSR } c' \text{ among codes in class } C}
\]

Store normalized weights per class.

---

## Step 3: Build patient-level CCSR profiles from MIMIC

### Goal
Convert MIMIC diagnoses into patient-level disease sets in CCSR space.

### Tasks
1. Extract all diagnosis codes per patient.
2. Restrict to ICD-10 diagnoses if needed.
3. Map diagnosis codes to CCSR categories.
4. Deduplicate CCSR categories within patient.
5. Build a patient-by-CCSR binary representation.

### Canonical unit
Use **patient-level union** across diagnoses.

### Output
A binary incidence structure equivalent to:

- rows: patients
- columns: CCSR categories
- values: `0/1`

---

## Step 4: Estimate CCSR co-occurrence counts

### Goal
Compute empirical co-occurrence frequencies between all pairs of CCSR categories.

### Definitions

Let:

- `N` = total number of patients
- `n_i` = number of patients with CCSR `i`
- `n_j` = number of patients with CCSR `j`
- `n_ij` = number of patients with both `i` and `j`

### Tasks
1. Compute node prevalence counts `n_i`.
2. Compute pairwise co-occurrence counts `n_ij`.
3. Exclude self-edges from evaluation graph or store separately as zero-distance if needed.
4. Persist counts efficiently as a sparse edge list.

---

## Step 5: Convert co-occurrence into distances

### Goal
Construct the weighted graph used for scoring.

### Smoothed co-occurrence probability
For each pair `(i, j)`, compute:

\[
\tilde p(i,j) = \frac{n_{ij} + \alpha}{N + \beta}
\]

### Default smoothing
Use configurable small positive constants, e.g.:

- `alpha = 1`
- `beta = 2`

### Distance
Define:

\[
d(i,j) = -\log(\tilde p(i,j))
\]

### Required properties
- symmetric
- finite for all retained pairs
- smaller when co-occurrence is more frequent

### Output fields
Store at least:

- `ccsr_i`
- `ccsr_j`
- `n_i`
- `n_j`
- `n_ij`
- `p_ij_smoothed`
- `distance_ij`

---

## Step 6: Define distance between ICD-10 classes

### Goal
Map evaluation-layer classes into graph-layer distances.

Let:

- `C0` = perturbed ICD-10 class
- `C1` = predicted ICD-10 class
- `S(C)` = CCSR categories linked to class `C`

### Primary definition
Use **expected pairwise CCSR distance**:

\[
d(C_0, C_1) = \sum_{i \in S(C_0)} \sum_{j \in S(C_1)} w(i \mid C_0) \, w(j \mid C_1) \, d(i,j)
\]

Where:

- `w(i | C)` are the class-to-CCSR weights from Step 2

### Fallback
If weighted distributions are unavailable, use uniform weights over the class CCSR set.

### Edge cases
- If either class has no mapped CCSR nodes, return `NA`
- If a pairwise CCSR distance is missing, either:
  - compute from zero co-occurrence using smoothing
  - or use a precomputed dense distance matrix

---

## Step 7: Score the top-`k` prediction sets

### Goal
Assign one scalar score to each prediction set per intervention.

For each example:

- perturbed class: `C0`
- naive predictions: `P_naive = [C1, ..., Ck]`
- guided predictions: `P_guided = [C1', ..., Ck']`

### Score definition
For a prediction set `P`:

\[
S(P \mid C_0) = \frac{1}{|P|} \sum_{C \in P} d(C_0, C)
\]

Lower score is better.

### Required outputs per example
- `naive_score = S(P_naive | C0)`
- `guided_score = S(P_guided | C0)`
- `delta = guided_score - naive_score`
- `winner = guided if delta < 0 else naive if delta > 0 else tie`

### Rules
- Use effective `k` after deduplication and mapping.
- If effective `k = 0`, mark score as missing.
- Keep per-example diagnostics:
  - mapping coverage
  - dropped predictions
  - effective `k`

---

## Step 8: Produce aggregate evaluation summaries

### Goal
Compare `guided` vs `naive` over all interventions.

### Required summary metrics
Compute over rows with valid scores:

- mean `naive_score`
- mean `guided_score`
- mean paired delta
- median paired delta
- standard deviation of delta
- fraction with `guided_score < naive_score`
- fraction ties
- count of valid rows
- count of invalid rows
- mean effective `k` by method

### Optional summaries
- stratify by perturbed ICD-10 chapter
- stratify by class frequency
- top improved classes
- top degraded classes

---

## Step 9: Quality-control checks

### Mapping QC
- number of unique perturbation terms mapped
- number of unique predicted terms mapped
- number of ICD-10 classes with non-empty CCSR mapping
- number of classes dropped due to missing mapping

### Graph QC
- number of CCSR nodes
- number of retained edges
- density or sparsity stats
- distribution of node prevalence
- distribution of edge distances

### Evaluation QC
- number of examples scored
- mean effective `k`
- rows with all predictions unmappable
- rows where perturbed class has no CCSR representation

---

## File / Module Structure

Suggested modules:

### `mappings/`
- `build_icd10_class_mapping.py`
- `build_ccsr_mapping.py`
- `build_term_mapping.py`
- `build_class_to_ccsr_distribution.py`

### `graph/`
- `extract_patient_ccsr_profiles.py`
- `compute_ccsr_cooccurrence.py`
- `build_ccsr_distance_graph.py`

### `evaluation/`
- `score_prediction_sets.py`
- `aggregate_results.py`

### `artifacts/`
- mapping tables
- graph edge list / matrix
- scored evaluation table
- summary report

---

## Data Contracts

## Prediction input contract
Each row must resolve to:

- `example_id: str`
- `perturbed_icd10_class: str`
- `naive_topk_icd10_classes: list[str]`
- `guided_topk_icd10_classes: list[str]`

## Graph edge contract
Each row must contain:

- `ccsr_i: str`
- `ccsr_j: str`
- `cooccurrence_count_ij: int`
- `cooccurrence_prob_ij: float`
- `distance_ij: float`

## Class-to-CCSR contract
Each row must contain:

- `icd10_class: str`
- `ccsr: str`
- `weight: float`

with weights summing to `1.0` per class.

---

## Default Configuration

```yaml
graph_layer: ccsr
evaluation_layer: icd10_class
patient_unit: patient_level_union
smoothing_alpha: 1.0
smoothing_beta: 2.0
class_distance: expected_pairwise
prediction_set_score: mean_distance
deduplicate_predictions: true
drop_unmappable_predictions: true
min_effective_k: 1
