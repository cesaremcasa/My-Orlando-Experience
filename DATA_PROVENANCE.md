# Data provenance and redistribution boundary

The public HEAD intentionally contains source code, tests, documentation, and
small synthetic fixtures only. The previously tracked PDFs, processed JSONL,
FAISS indexes, embedding arrays, and source-derived metadata were removed from
HEAD because this repository has no explicit redistribution permission for
those materials. Git history was not rewritten; removal from HEAD is not a
claim that history is clean.

The removed assets must be obtained or rebuilt by an operator with the
appropriate rights:

```bash
python 01_parse_pdfs.py       # local, user-supplied PDFs under data/raw_pdfs/
python 04_build_auxiliary_layers.py
python 06_rebuild_core_factual_index.py
```

Those commands are not run by CI and no downloaded source content is included
in release artifacts. The repository-authored code is MIT-licensed by
`LICENSE`; external PDFs, guides, maps, provider responses, and derived data
retain their original terms and are not relicensed here.

`tests/fixtures/synthetic_context.json` is newly generated test-only material,
dedicated to the public domain under CC0 1.0. It contains no real travel
source content and is safe to redistribute.
