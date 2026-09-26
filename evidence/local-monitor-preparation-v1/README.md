# Retained first monitor replay failure

The first Windows fixture export completed, but direct comparison of the in-memory Python record with decoded JSON failed because tuples become lists in JSON. No model or GPU calls occurred. The fixture, manifest and original verifier bytes were retained before correction in `source-monitor-preparation-v1.json.gz`.

The follow-up normalized the replay through JSON. A separate subsequent Linux check then found last-bit differences in intermediate log statistics. The final [evidence and replay](../local-monitor-v1/README.md) keep exact decisions and record structure while allowing a declared `1e-12` finite-float tolerance. Neither correction changes recorded fixture outcomes, alarm decisions or the learned policy.
