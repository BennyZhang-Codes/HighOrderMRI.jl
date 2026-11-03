# Documentation tools

The VitePress website and API map live on the `docs` branch. This directory contains the Julia exporter and checks that must belong to the selected code commit. Source docstrings remain available to Julia help and IDEs.

- `contract_check.jl`: compare complete documented signatures with parsed source without evaluating default expressions; optionally compare executable syntax after removing documentation.
- `export_api.jl`: export explicitly mapped Base.Docs bindings as Markdown fragments; reject missing/duplicate selections, unsupported syntax and unresolved references. Mapping `parts` select type/method docs; `<untyped>` selects an argument with no type annotation. Dependency-owned docstrings are excluded only when the map explicitly requests it, and exclusions are recorded.
- `test_contract_check.jl`, `test_export_api.jl`: tool regression checks. Set `DOCS_API_MAP` to the website checkout's `docs/api-map.toml` for exporter tests.
- `check_examples.jl`: CPU checks for the selected complete website examples. Set `DOCS_SITE_PATH` to the website's `docs` directory and hide GPU devices. Use one Julia thread for this bounded CPU check; the plotting example runs in its own process so Python objects are not carried into subsequent threaded numerical examples.
- `check_site.py`: Python 3.11+ standard-library check of generated pages, API anchors, links, source/edit identity and math errors.

Formal website builds require a clean code checkout whose full SHA matches `docs/code-version.json`, with these tools included in that commit. A labelled local preview can use an isolated source copy with uncommitted docstrings. Build output and API fragments are disposable and not committed. The current workflow produces dev only.

The source mechanism uses [Julia documentation](https://docs.julialang.org/en/v1.12/manual/documentation/); the website uses [VitePress file inclusion](https://vitepress.dev/guide/markdown#markdown-file-inclusion). Neither generation nor validation installs or depends on Documenter. A rendered page is not evidence that all Julia examples or GPU paths ran: test results must identify their actual scope.
