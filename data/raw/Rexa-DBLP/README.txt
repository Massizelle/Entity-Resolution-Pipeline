Expected raw layout for the `rexa_dblp` benchmark
===============================================

Put the benchmark files for Rexa-DBLP in this directory, or point
`REXA_DBLP_DIR` to another directory with the same structure.

Supported source files:
- one RDF file for DBLP:
  - `dblp.rdf`, `dblp.owl`, `dblp.xml`, `dblp.ttl`, `dblp.nt`, `dblp.n3`
  - or common alternatives such as `swetodblp_april_2008.rdf.gz`
- one RDF file for Rexa:
  - `rexa.rdf`, `rexa.owl`, `rexa.xml`, `rexa.ttl`, `rexa.nt`, `rexa.n3`

Optional ground truth:
- OAEI alignment RDF/XML:
  - `refalign.rdf`, `alignment.rdf`, `reference.xml`, `gold_standard.rdf`, ...
- or a CSV mapping file:
  - two ID columns, for example `idDBLP,idRexa`

The ingestion layer will:
- detect the two source RDF files
- flatten literals and URI object labels into tabular rows
- orient alignment pairs automatically to `idDBLP,idRexa`
