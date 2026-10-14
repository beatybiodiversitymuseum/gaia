# Pinned biodiversity-standard data

`gbif_extension_registry.json` pins the latest production definitions published
by GBIF at the review date recorded in its `provenance` object. Runtime ingestion
uses this local file and does not depend on the availability or current contents
of the live registry.

To review an upstream registry update, run this command from the repository root:

```bash
make refresh-gbif
```

The target records the current date. To supply a different review date, run
`make refresh-gbif SNAPSHOT_DATE=YYYY-MM-DD`.

The command downloads GBIF's production `extensions.json`, retains entries marked
`isLatest`, downloads each referenced XML definition, and records its allowed
`qualName` property URIs. It validates unique row types and matching XML
definitions before atomically replacing the snapshot. Review the resulting diff
before committing it, especially added or removed row types and terms.

## Darwin Core terms

`darwin_core_sources.json` pins exact commits and downloaded archive hashes for
the official `tdwg/dwc` and `tdwg/rs.tdwg.org` repositories.
`darwin_core_terms.json` is the deterministic, normalized runtime artifact built
from those sources. It also contains the ordered Simple Darwin Core property
profile used to validate archive fields and build occurrence-table columns.
Runtime ingestion and lookup read only the local artifact and never use the
network.

To verify both sources and rebuild the artifact, run:

```bash
make refresh-darwin-core
```

The builder selects the recommended version of each term when one exists, and
otherwise the latest historical version. It cross-checks every compiled term
version against the pinned source tables in `rs.tdwg.org`, records hashes for
all inputs, and atomically replaces the artifact. Updating the standard is an
explicit pin update and review, not an unpinned refresh.
