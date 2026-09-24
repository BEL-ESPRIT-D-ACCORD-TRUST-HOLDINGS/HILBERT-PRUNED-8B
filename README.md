# Hilbert

## Pruned 4B Decision Model · CUDA Backend · Alloy Governance

**Hilbert** is an experimental pruned 4B semantic-decision model and systems runtime derived from the decision-native architecture demonstrated by SemIf.

The project focuses on a narrow execution path:

```text
state + question + declared options
              │
              ▼
       Hilbert 4B Model
              │
              ▼
       typed option scores
              │
       ┌──────┴──────┐
       ▼             ▼
    CUDA          Governance
   Backend          Layer
       │             │
       ▼             ▼
   Kernels         Alloy
       │          constraints
       └──────┬──────┘
              ▼
        Auditable Result
```

> **Status:** Experimental research implementation.
> Model pruning, distillation, CUDA kernels, and Alloy governance specifications must be independently benchmarked and verified before production claims are made.

---

## 1. What Hilbert Is

Hilbert treats many AI workloads as **typed decisions rather than text-generation problems**.

Instead of asking a model to generate:

```text
"The request should be routed to account-access support because..."
```

Hilbert evaluates declared alternatives directly:

```json
{
  "question": "Which queue should handle this request?",
  "options": [
    {"id": "access", "description": "Account access support."},
    {"id": "billing", "description": "Billing support."}
  ]
}
```

The resulting computation is conceptually:

```text
P(access | state, question, options)
P(billing | state, question, options)
```

This follows the decision-native pattern documented by the upstream SemIf experiment, where declared option logits are read directly without sampling an answer sentence or running a JSON-repair loop.

---

# 2. Hilbert 4B

Hilbert targets a **pruned/distilled 4B-class model**.

The intended transformation is:

```text
Reference semantic model
          │
          ▼
   distillation data
          │
          ▼
   structured decisions
          │
          ▼
       pruning
          │
          ▼
    Hilbert 4B
          │
          ▼
   CUDA execution
```

The supplied SemIf baseline demonstrates Qwen3.5-4B as a frozen 4B decision model and reports direct-logit evaluation across multiple decision workloads.

Hilbert's pruning/distillation pipeline is a separate experimental layer and should not be interpreted as evidence that the upstream SemIf model itself was distilled or pruned.

---

# 3. Decision-Native Runtime

Hilbert removes unnecessary text-generation work from decision workloads.

### Conventional generation

```text
prompt
  ↓
model
  ↓
tokens
  ↓
JSON/text
  ↓
parser
  ↓
decision
```

### Hilbert

```text
state
  +
question
  +
options
  ↓
model forward pass
  ↓
typed option scores
  ↓
decision
```

The source experiment reports that direct typed logits produced zero output tokens for its measured decision path, while a compact autoregressive JSON baseline generated 111 tokens for the same 21 binary decisions.

---

# 4. CUDA Backend

Hilbert adds a dedicated CUDA execution backend.

```text
Hilbert Runtime
      │
      ▼
CUDA Backend
      │
 ┌────┼────────────────────┐
 ▼    ▼                    ▼
GEMM  Attention            Norm
 │      │                  │
 ▼      ▼                  ▼
Tensor memory        Reduction kernels
      │
      ▼
Typed decision logits
```

The CUDA layer is intended to provide:

* GPU-resident model execution
* fused tensor operations
* reduced intermediate-memory movement
* optimized matrix multiplication
* attention kernels
* normalization kernels
* reduction kernels
* option-score extraction
* deterministic execution modes where supported by the hardware/runtime

CUDA performance numbers should be treated as **unverified until benchmark artifacts are committed**.

---

# 5. CUDA Kernel Boundary

The kernel layer should remain explicit rather than hiding GPU execution behind an opaque abstraction.

Example structure:

```text
cuda/
├── attention/
├── matmul/
├── normalization/
├── reduction/
├── embedding/
├── logits/
├── memory/
└── dispatch/
```

Each kernel should have:

```text
kernel specification
      │
      ├── input dimensions
      ├── output dimensions
      ├── dtype
      ├── memory contract
      ├── synchronization requirements
      ├── numerical tolerance
      └── test vector
```

A kernel is not considered verified merely because it compiles or produces plausible output.

---

# 6. Alloy Governance Layer

Alloy is used as a **relational specification and counterexample engine** around the governance model.

It does not prove the neural network's semantic correctness.

The intended separation is:

```text
Neural computation
       │
       ▼
candidate decision
       │
       ▼
governance specification
       │
       ▼
Alloy analysis
       │
 ┌─────┴─────┐
 ▼           ▼
SAT         UNSAT
 │           │
 ▼           ▼
counter-    constraint
example     boundary
```

Alloy models should specify relationships such as:

```text
Node
Decision
Policy
Authority
Grant
Permission
Resource
Evidence
Revision
```

Example conceptual invariant:

```text
No administrative permission exists
unless an authorized grant relates:

    Node × Grant × Authority × Permission
```

The Alloy model should be used to search for counterexamples to the governance invariants.

---

# 7. Governance Boundary

Hilbert separates **model output** from **authority**.

A model can produce:

```text
decision = ALLOW
```

without possessing authority to authorize the corresponding operation.

The governance layer therefore follows:

```text
MODEL OUTPUT
     │
     ▼
PROPOSED DECISION
     │
     ▼
POLICY EVALUATION
     │
     ▼
AUTHORITY CHECK
     │
     ▼
GOVERNANCE RESULT
```

This prevents:

```text
model confidence
      ≠
administrative authority
```

and:

```text
prediction
      ≠
authorization
```

---

# 8. Auditable Results

Every decision should preserve sufficient metadata to reproduce the computation.

Recommended result structure:

```json
{
  "decision_id": "example-001",
  "model": "hilbert-4b",
  "model_revision": "<PINNED_REVISION>",
  "prompt_sha256": "<HASH>",
  "options": [
    {
      "id": "access",
      "probability": 0.91
    },
    {
      "id": "billing",
      "probability": 0.09
    }
  ],
  "selected": "access",
  "backend": "cuda",
  "kernel_revision": "<PINNED_REVISION>",
  "governance_revision": "<PINNED_REVISION>"
}
```

The upstream experiment similarly records typed scores, timing, model revision, and prompt hashes as part of its audit-oriented result structure.

---

# 9. Reproducibility

Hilbert should pin:

* model revision
* tokenizer revision
* pruning configuration
* distillation dataset revision
* CUDA version
* GPU architecture
* kernel revision
* compiler version
* numerical precision
* benchmark fixture
* governance specification revision
* Alloy model revision

A result without its corresponding revision metadata should be treated as incomplete evidence.

---

# 10. Benchmark Structure

```text
benchmarks/
├── decisions/
├── pruning/
├── distillation/
├── cuda/
├── numerical/
├── governance/
├── alloy/
└── reproduction/
```

Each benchmark should distinguish:

```text
correctness
performance
numerical agreement
decision agreement
governance validity
```

These properties must not be collapsed into one score.

---

# 11. Model Quality

The upstream SemIf experiment provides a useful reference methodology: fixed workloads, frozen models, explicit metrics, prompt hashes, timing measurements, and committed raw results.

Hilbert should therefore report at minimum:

| Metric                 | Purpose                                 |
| ---------------------- | --------------------------------------- |
| Decision accuracy      | Semantic decision quality               |
| Balanced accuracy      | Class-balanced evaluation               |
| Calibration error      | Confidence reliability                  |
| Decision agreement     | Comparison against reference model      |
| Perturbation stability | Sensitivity to controlled input changes |
| Kernel numerical error | CUDA correctness                        |
| Throughput             | Runtime performance                     |
| Latency                | Per-decision response                   |
| Memory footprint       | Deployment requirements                 |
| Governance violations  | Policy-model boundary testing           |

---

# 12. CUDA Correctness

Performance cannot substitute for numerical verification.

For each optimized kernel:

```text
Reference implementation
          │
          ▼
High-precision / trusted baseline
          │
          ▼
CUDA implementation
          │
          ▼
comparison
```

Record:

```text
absolute error
relative error
maximum error
mean error
ULP difference where applicable
NaN/Inf behavior
edge-case behavior
```

A faster kernel that changes the decision boundary must be treated as a correctness issue, not merely a performance optimization.

---

# 13. Alloy Counterexample Testing

Governance specifications should deliberately attempt to break themselves.

Example challenge classes:

```text
duplicate authority
revoked grant
expired grant
conflicting policy
orphaned node
unauthorized node
circular authority
missing evidence
stale revision
conflicting permissions
```

The expected workflow is:

```text
Invariant
   ↓
Alloy model
   ↓
bounded analysis
   ↓
counterexample
   ↓
repair specification
   ↓
rerun analysis
```

An Alloy model finding no counterexample within a finite scope is **not equivalent to an unrestricted mathematical proof**.

---

# 14. Negative-Gate Principle

Hilbert adopts an explicit negative gate:

```text
UNVERIFIED
    ↓
DO NOT PROMOTE TO VERIFIED
```

Therefore:

```text
compiles
    ≠
correct

passes test
    ≠
proven

high confidence
    ≠
authority

CUDA speedup
    ≠
semantic correctness

Alloy SAT result
    ≠
system correctness

absence of a bounded counterexample
    ≠
universal proof
```

This distinction is central to the project.

---

# 15. Repository Composition

Current repository composition:

```text
Python       44.1%
C            43.0%
CUDA          6.5%
JavaScript    3.5%
HTML          1.7%
Alloy         0.8%
Makefile      0.4%
```

The stack deliberately keeps the CUDA and Alloy layers visible in the repository rather than hiding them behind a single runtime abstraction.

---

# 16. Directory Layout

```text
hilbert/
├── model/
│   ├── config/
│   ├── tokenizer/
│   ├── pruning/
│   └── distillation/
│
├── runtime/
│   ├── python/
│   └── c/
│
├── cuda/
│   ├── attention/
│   ├── matmul/
│   ├── normalization/
│   ├── reduction/
│   ├── logits/
│   └── dispatch/
│
├── alloy/
│   ├── governance/
│   ├── authority/
│   ├── permissions/
│   └── counterexamples/
│
├── benchmarks/
│   ├── decisions/
│   ├── cuda/
│   ├── model/
│   └── governance/
│
├── web/
│   ├── demo/
│   └── assets/
│
├── docs/
│   ├── METHOD.md
│   ├── RESULTS.md
│   ├── REPRODUCE.md
│   ├── CUDA.md
│   └── GOVERNANCE.md
│
├── Makefile
└── README.md
```

---

# 17. Relationship to SemIf

Hilbert is inspired by the architecture documented in the supplied SemIf project:

* runtime-defined criteria
* decision-native scoring
* direct option-logit evaluation
* shared-state execution
* reproducible fixtures
* row-level results
* pinned model revisions
* prompt hashing
* explicit claim boundaries

These characteristics are documented in the supplied project material.

Hilbert adds its own experimental layers:

```text
SemIf decision architecture
            │
            ▼
       Hilbert 4B
            │
      ┌─────┴─────┐
      ▼           ▼
   CUDA         Alloy
   backend      governance
      │           │
      └─────┬─────┘
            ▼
      Hilbert Runtime
```

Hilbert should not represent upstream SemIf measurements as measurements of Hilbert.

---

# 18. Claim Discipline

The project distinguishes four states:

```text
PROPOSED
    ↓
IMPLEMENTED
    ↓
BENCHMARKED
    ↓
INDEPENDENTLY VERIFIED
```

A feature remains **PROPOSED** until implementation evidence exists.

A feature remains **IMPLEMENTED** until benchmark evidence exists.

A benchmark result remains **BENCHMARKED** until its reproduction procedure is independently confirmed.

This prevents README claims from becoming stronger than the underlying evidence.

---

# 19. License

This repository should include an explicit root license and identify the license applicable to each third-party dependency and model.

Model weights and upstream components retain their respective licenses unless their licenses explicitly permit redistribution.

The supplied SemIf project, for comparison, states that its project code is MIT licensed while upstream model weights retain their own licenses.

---

# 20. Research Status

Hilbert is an experimental systems research project combining:

```text
4B model compression
+
decision-native inference
+
CUDA kernel engineering
+
GPU execution
+
formal governance modeling
+
Alloy counterexample analysis
+
reproducible benchmarking
```

The central engineering question is:

> **How small can a decision-oriented model become while retaining measurable decision quality, efficient GPU execution, reproducibility, and an explicit machine-checkable governance boundary?**

The answer must come from the benchmarks, kernel tests, model evaluations, and Alloy counterexamples—not from the README.

---

## Hilbert

```text
PRUNE THE MODEL.
EXECUTE THE DECISION.
VERIFY THE BOUNDARY.
```
