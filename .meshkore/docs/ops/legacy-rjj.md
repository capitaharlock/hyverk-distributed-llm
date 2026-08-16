---
title: "Legacy _rjj archive"
category: ops
updated: 2026-08-16
owner: hyverk-lead
status: active
---

# Legacy `_rjj` archive

The old private `_rjj/` tree was git-crypt encrypted. It was removed from the working tree when this repo adopted the MeshKore Standard.

A **ciphertext** tarball is kept locally (gitignored) at:

`.meshkore/_archive/_rjj-encrypted-*.tar.gz`

Core developers with the historical git-crypt key can unlock that archive offline and migrate any remaining notes into `.meshkore/docs/` or `.meshkore/credentials/`. Public contributors do not need it.

