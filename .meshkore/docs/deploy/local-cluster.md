---
title: "Local cluster"
category: deploy
updated: 2026-08-16
owner: hyverk-lead
status: active
---

# Local multi-Mac cluster

Canonical runbook: `scripts/LOCAL_CLUSTER.md`.

```bash
bash scripts/prepare-model.sh
bash scripts/run-coordinator-local.sh          # :17000
bash scripts/run-node-local.sh http://<ip>:17000
bash scripts/smoke-coordinator-local.sh       # no GPU / no model download
```

Override model dir with `HYVERK_MODEL_DIR`. Gate public-ish LAN inference with `HYVERK_API_KEY`.

