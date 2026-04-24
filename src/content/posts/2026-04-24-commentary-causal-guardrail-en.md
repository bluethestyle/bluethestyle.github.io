---
title: "[Commentary] A reliability flag on every prediction — Causal Guardrail and Mahalanobis distance"
date: 2026-04-24 12:00:00 +0900
categories: [Commentary]
tags: [commentary, causal, guardrail, mrm, paper-3, financial-ai]
lang: en
excerpt: "If MRM Ep 3's audit log guarantees *record integrity*, who judges whether *each individual prediction* is trustworthy? A prediction-level guardrail that detects OOD signals via Mahalanobis distance on the Causal Expert's latent space, and how it pairs with CEH attribution to fold into the audit trail."
alt_lang: /2026/04/24/commentary-causal-guardrail-ko/
source_url: https://doi.org/10.5281/zenodo.19622052
source_label: "Paper 2 v2 § CEH / Paper 3 Findings 10-11"
---

*If MRM Thread Ep 3 covered "the audit log persists without
tampering", this Commentary goes one layer down — the layer that
structurally judges whether *a single prediction* is
trustworthy.*

## Log integrity and prediction reliability are different questions

In MRM Ep 3, what the seven audit tables and the HMAC hash chain
guarantee is *aggregate-level integrity*. "At 14:37 on 2026-04-15,
model v143 recommended which product for customer X with which
feature combination" can be reconstructed 15 months later. That
the chain wasn't broken is what AuditAgent verifies daily.

But when a reconstructed record is actually examined, the
supervisor's natural follow-up is "was this prediction
**trustworthy**?"

At the aggregate level the best answer we can give is "model v143
had an overall AUC of 0.82". For the single prediction under
dispute, that number is meaningless. Whether that prediction sat
inside the model's training distribution, or whether the model
was making an uncertain extrapolation outside that distribution
(OOD), is a separate judgment.

Causal Guardrail is the prediction-level indicator that answers
this question. Its design is covered in Paper 3's Findings 10 and
11, and it runs on the latent space of the Causal expert — one
of the seven experts introduced in Ep 4.

## Mahalanobis distance — the thin principle behind OOD detection

The guardrail's core is simple. During training, the Causal
expert's latent distribution gets summarized by its mean vector
(μ) and covariance matrix (Σ). At inference time, each incoming
prediction's latent vector z is scored with the Mahalanobis
distance:

```
d_M(z) = sqrt( (z - μ)^T · Σ^(-1) · (z - μ) )
```

The difference from Euclidean distance is that *the axes are
normalized by the covariance*. If the training latent
distribution is elliptical, "closeness" is measured along the
ellipsoid; if spherical, along the sphere. The yardstick for
"close" adapts to the distribution's own geometry.

Predictions inside the training distribution come out with small
d_M (a ±2σ region is typically d_M ≈ 1-2). Predictions from
*outside* the distribution — unseen feature combinations, rare
segments, regions requiring extrapolation — spike in d_M. Any
prediction over threshold gets a **reliability flag** and lands
in the audit log with that flag attached.

## Synthetic-probe target — 100% TPR at 5% FPR

We checked how useful this layer actually is by mixing
in-distribution and out-of-distribution samples in a synthetic
probe. A small number of OOD samples — deliberately constructed
outside the training latent distribution — were injected, and we
measured how well the guardrail flagged them.

The separation came out nearly clean. At a 5% FPR threshold
(False Positive Rate — the rate at which in-distribution samples
get misflagged), TPR was 100% (True Positive Rate — the rate at
which actual OOD samples were caught). Whether these numbers
hold on real production distributions is something to verify as
traffic accumulates after 2026-04-30, but on the narrow task of
*geometric separation of the training distribution* the
Mahalanobis-based detector has strong synthetic-level evidence.

## Paired with CEH Attribution — the complete regulatory answer

The guardrail alone is only half of the answer. The real
regulatory value appears when it's paired with **CEH (Causal
Explainability Head) attribution**. Introduced in Paper 3 Finding
9 and Paper 2 v2, this attribution layer decomposes each
prediction into *which features drove it, and how*, from a
causal standpoint.

When both layers land in the audit log, the shape of the answer
to a dispute query changes.

- **CEH attribution** → "why was this recommendation made for
  this customer?"
- **Causal guardrail** → "can we trust that recommendation?"

CEH without guardrail — there's an explanation, but it may be
hiding the fact that the explanation is grounded in an
extrapolation *outside* the training distribution. Guardrail
without CEH — there's a "low reliability" flag but no answer to
the *why*. The point of this design is that both layers run on
the same Causal expert's same forward pass and both feed the
audit log in the same step.

## Folding into the audit log

On the implementation side, the guardrail output doesn't create
a new table — it writes into Ep 3's `log_guardrail`. The entry
schema is already defined for Safety Gate's regulatory-keyword
checks, and the guardrail output (distance value, pass/fail vs
threshold, short latent-stats summary) is attached as additional
fields. HMAC signatures link it to the same chain as other `log_*`
entries.

When a supervisor asks for the reliability history of a specific
prediction 15 months later, a one-line SQL query against
`log_guardrail` filtered by prediction ID pulls the guardrail
result and the CEH attribution together. No separate "reliability
reporting system" is built — every prediction auto-deposits both
entries, and those entries accumulate.

## Limits and the next question

Mahalanobis-based OOD detection uses only the *first and second
moments* of the training distribution (mean and covariance).
When the distribution is multi-modal or strongly non-Gaussian —
for instance, completely different feature patterns per segment
— a single μ, Σ pair can blur the boundary. In that regime the
extension is per-segment Mahalanobis or density-based OOD (e.g.,
normalizing flows).

Threshold selection is an operator-judgment call. Whether 5% FPR
is "reasonable" depends on domain and risk tolerance — tightening
to 2% FPR reduces true positives in trade. The threshold is
managed in config, and changes route through the PR path and
leave an audit-log entry (same policy as Ep 6's threshold
management).

As real-traffic metrics accumulate, the 100% / 5% numbers from
the synthetic probe will be re-measured on production
distributions, and the threshold — or the detection structure
itself — may need adjustment. That process is itself a quarterly
MRM review item.

## A new line on the committee's review sheet

Introducing the guardrail adds one line to the quarterly MRM
review — *"predictions flagged this quarter, and the fraction
that turned out to be actual issues"*. If flags are frequent but
follow-up issues are rare, the threshold is too strict. If flags
are rare but customer disputes are common, the threshold is too
loose — or the detection structure itself is misfit. Operational
metrics roll naturally into audit metrics.

The audit log stays tamper-evident, and each prediction carries
its own reliability signal. Only when both are stored together
does the question "15 months later, why this prediction and how
reliable was it?" collapse to a single query. Aggregate integrity
and prediction-level reliability answer different questions; the
point of this Commentary is that both sit inside the same audit
chain.

---

Source:
[Paper 2 (Zenodo)](https://doi.org/10.5281/zenodo.19622052) v2 § CEH,
Paper 3 Findings 10-11 (WIP);
implementation lives in the
[open-source repo](https://github.com/bluethestyle/aws_ple_for_financial).
