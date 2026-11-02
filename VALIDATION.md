# Publication validation

Validated on 2026-11-01 (America/Los_Angeles), using the selected public snapshot.

- 692 Python files parsed successfully with Python 3.14; this checks syntax, not all runtime dependencies or older Python compatibility.
- 60 Shell files passed `bash -n`.
- 417 JSON files parsed successfully, including the publication manifest.
- 11 prompt/evaluator contract tests passed using pytest in an isolated local virtual environment. The exact selected commands are in the README. No full historical test-suite pass is claimed: some tests require omitted private evidence or infrastructure.
- The 64-pair fresh-task analysis was recomputed with the standard library. Its complete resulting JSON exactly matched the published analysis JSON (including all rows, bootstrap settings and numerical outputs).
- Gitleaks scanned the public files. The initial 49 generic-key detections were reviewed as hexadecimal model revisions / file hashes. Allowlisting only complete 40/64-character hexadecimal values for that rule produced zero unresolved detections. No broad file exclusions were used for the credential scan.
- Private communication reports, original repository history, server receipts, raw task/episode payloads, credentials and model weights were not exported. Known infrastructure paths, tracking accounts and device identifiers were generalized; source and public file hashes identify transformed files.

This validates the publication and the selected CPU contracts. It does not certify every historical result, reproduce GPU training, or establish compatibility with an arbitrary current OpenEvo release. No scientific workload was launched.
