# Licensing

FlightStack is MIT licensed.  It currently contains FlightStack-authored source
and data definitions plus package manifests/lockfiles; it does not vendor
third-party source, assets, datasets, or model checkpoints.

Direct declared runtime/build dependencies are recorded in
[`THIRD_PARTY.md`](../../THIRD_PARTY.md).  The pre-researched upstream
repositories in [`SOURCE_MANIFEST.md`](SOURCE_MANIFEST.md) are references, not
an implicit license to copy their code or assets.

Before adding any dependency, copied source, model checkpoint, mesh, texture,
or dataset, verify and record:

1. the exact upstream version or artifact hash;
2. the license and any notice/attribution requirements;
3. whether that license is compatible with the intended MIT distribution;
4. the source path/artifact provenance and the precise copied or generated
   scope; and
5. any separate asset, dataset, model-weight, or trademark restrictions.

GPL-only or unlicensed implementation is not copied into the MIT FlightStack
core.  A permissive repository license does not automatically cover every
asset, pretrained weight, or dataset it happens to reference.
