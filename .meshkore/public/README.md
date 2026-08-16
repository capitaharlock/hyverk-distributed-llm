# Hyverk cluster

Distributed training and inference for open-source coding models.

## Join (developers)

1. Clone the repo and read the root `README.md`.
2. Adopt MeshKore locally: secrets go in `.meshkore/credentials/` and `.meshkore.local` (never commit).
3. To join the MeshKore channel, ask a lead for an invite URL, then:

```bash
MESHKORE_INVITE='https://hub.meshkore.com/agents/invites/<nonce>/join' \
  MESHKORE_HUB_URL=https://meshkore-relay.fly.dev \
  bash scripts/meshkore-join.sh
```

## Run Hyverk (demo)

There is no required public coordinator. Start a local cluster:

```bash
bash scripts/prepare-model.sh
bash scripts/run-coordinator-local.sh
bash scripts/run-node-local.sh http://<coordinator-lan-ip>:17000
```

See `scripts/LOCAL_CLUSTER.md` and `.meshkore/docs/`.
