# Agent instructions — Hyverk

This repo applies the [MeshKore Standard](https://meshkore.com/standard).
Normative layout and schemas: https://api.meshkore.com/v1/standard.md

## Product

Hyverk is a distributed network for **training and serving** open-source coding models.
Contributors run `hyverk-node` against a **coordinator you host** (local LAN or any cloud).

## Hard rules

- Never commit `.meshkore/credentials/`, `.meshkore.local`, API keys, or invite URLs.
- Never invent a live public coordinator URL. Default local: `http://127.0.0.1:17000`.
- Prefer docs under `.meshkore/docs/` and tasks under `.meshkore/modules/<id>/tasks/`.
- Link to https://meshkore.com/standard instead of copying normative MeshKore text.
- Do not push to `origin` unless the operator asks.

## Modules

See `.meshkore/public/cluster.yaml` → `modules`.
