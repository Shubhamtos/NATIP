# Investor-inspired research overlays

NATIP uses a small curated subset of the supplied Investor Skills library,
not all 63 personas and not their allocation or trading thresholds. Inspiration:
`minervini-vcp`, `darvas-box`, and the library's earnings-quality/forensic
research principles. The original MIT attribution is retained in
`docs/licenses/investor_skills_LICENSE.txt` (Bob Xu, 2026). No Questflow
subscription, skill installation, automatic browsing or broker access is added.

## Existing screens

- Buying Agent VCP/DarvaX results: choose a stock, then Explain selected setup
  with AI. The relevant research perspective is automatic. US scanners now
  expose this same optional review. Institutional motives remain unknown;
  Indian holding requirements cannot be assumed applicable to US companies.
- Quarterly Results: open Company detail, then Review earnings quality with
  Gemini under Earnings quality. Deterministic calculations remain available
  even if AI is disabled, not configured or unavailable.

Each review is one combined request rather than one request per investor or
factor. Existing provider retry/backoff can make additional transport attempts.
API credentials remain in NATIP's existing secret configuration. No new API
key or market-data source is introduced. Quarterly review currently uses the
configured Gemini model; pattern reviews preserve Gemini/OpenAI provider choice.

Reviews must supply a strongest counterargument, dated evidence references,
missing inputs, conditional action and invalidation. Missing or invented
evidence identifiers are rejected, not silently displayed as sourced research.
Version 2 makes citation IDs and a counterargument mandatory in Gemini's
framework-response schema, restricted to supplied IDs. Brackets/case and exact
unambiguous source URLs are normalized to known IDs. Missing IDs, unknown
sources and ambiguous URLs are still rejected; no citation is invented.
Source references establish traceability, not automatic proof that every AI
claim is true. Documents and excerpts remain untrusted data, not instructions.
No retrieval augmentation, numerical fact-checking engine or institutional
causality inference is claimed.

The curated checklist has a version and SHA256 content hash. Cache signatures
include evidence content, framework version/hash and model. Pattern signatures
also include provider and request rules. Changing evidence, document revisions
or model invalidates cached output. Identical earnings requests reuse session
cache; identical pattern reviews are displayed without another paid request.
Caches persist across Streamlit reruns, not arbitrary new browser sessions.

This overlay never updates scanner scores/ranks, deterministic result analyses,
shortlists, position sizes, frozen model features/weights/thresholds or Kronos
forecast paths. AI confidence is not calibrated investment-success probability.
Publication dates are not inferred and current excerpts are not used as
point-in-time ML features. No unattended action or live order is possible.

Tests cover framework routing, unsupported-perspective rejection, counterargument
and citation validation, preservation of missing publication dates, PDF page
references, input immutability, hash invalidation and on-demand cache reuse.
