# Documentation tools

The VitePress website, API map and build tools live in `docs/` on the code branches. This directory contains the website build entry points and Julia documentation checks. Source docstrings remain available to Julia help and IDEs.

- `contract_check.jl`: compare complete documented signatures with parsed source without evaluating default expressions; optionally compare executable syntax after removing documentation.
- `export_api.jl`: export explicitly mapped Base.Docs bindings as Markdown fragments; reject missing/duplicate selections, unsupported syntax and unresolved references. Mapping `parts` select type/method docs; `<untyped>` selects an argument with no type annotation. Dependency-owned docstrings are excluded only when the map explicitly requests it, and exclusions are recorded.
- `test_contract_check.jl`, `test_export_api.jl`: tool regression checks. Set `DOCS_API_MAP` to the website checkout's `docs/api-map.toml` for exporter tests.
- `check_examples.jl`: CPU checks for the selected complete website examples. Set `DOCS_SITE_PATH` to the website's `docs` directory and hide GPU devices. Use one Julia thread for this bounded CPU check; the plotting example runs in its own process so Python objects are not carried into subsequent threaded numerical examples.
- `check-version.jl`: run the contract, exporter and CPU example checks from the selected code checkout, using the supplied website input directory.
- `prepare-api.mjs`: verify the selected code checkout and invoke its Julia API exporter.
- `build-site.mjs`, `build-dev.mjs`, `versions.mjs`: prepare and build the versioned VitePress site from explicit code and documentation revisions.
- `check_site.jl`: compatibility CLI for generated-page checks; delegates to `docs/publish/PublicationTools.jl`. Run it with `--project=docs/publish`.
- `*.test.mjs`: website build, exporter-selection and historical-path regression checks.

Formal website builds require a clean code checkout whose full SHA matches the selected version's `code_sha`, with its exporter and code-side checks included in that commit. Both API preparation and example checks prefer `docs/scripts/` in that checkout and accept the historical `scripts/docs/` layout there. They never borrow an exporter or code-side tests from a different checkout. A labelled API preview can use an isolated source copy with uncommitted docstrings; the complete-site builder still requires the selected clean code commits. Build output and API fragments are disposable and not committed.

From the repository root:

```bash
npm ci --prefix docs --no-audit --no-fund
npm run docs:check-tools --prefix docs
CUDA_VISIBLE_DEVICES='' MPLBACKEND=Agg JULIA_NUM_THREADS=1 JULIA_LOAD_PATH='@:@stdlib' julia --startup-file=no --project=. docs/scripts/check-version.jl . docs
```

Publishing and deployment use the independent Julia environment in `docs/publish/`. Pushes to `dev_jinyuan` publish `dev`; pushes to `main` validate a candidate without publishing. Formal releases add their version directory and update `stable`. Generated sites and historical versions are archived on `gh-pages`.

The source mechanism uses [Julia documentation](https://docs.julialang.org/en/v1.12/manual/documentation/); the website uses [VitePress file inclusion](https://vitepress.dev/guide/markdown#markdown-file-inclusion). Neither generation nor validation installs or depends on Documenter. A rendered page is not evidence that all Julia examples or GPU paths ran: test results must identify their actual scope.
