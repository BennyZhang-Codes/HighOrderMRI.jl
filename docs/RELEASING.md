# Versioned documentation releases

Documentation source belongs to the package's `docs/` directory on `main`
and `dev_jinyuan`. VitePress remains the renderer. `gh-pages` contains the
checked generated versions and is updated by workflow; no separate source
branch is needed.

## Branches and releases

- A push to `dev_jinyuan` updates `dev/` from that code commit's own `docs/`.
- A push to `main` updates `main/` from that code commit's own `docs/`.
- Publishing a non-prerelease GitHub Release with a `vMAJOR.MINOR.PATCH` tag
  adds that version and selects the newest formal release as `stable/`.
- The root URL opens stable. Drafts, prereleases and unpublished tags do not
  advance stable. Pull requests build preview artifacts without deployment.

Review the code and its `docs/` content together before tagging a release.
`Project.toml` must match the tag's version and the tag must belong to main's
history. Do not move a published tag. New releases use the tagged code and
its tagged documentation; old releases keep their recorded code/docs SHAs.
The original `v0.1.0` documentation snapshot predates the in-tree migration
and remains available through the preserved Git history and version registry.

## Build and publication checks

The build resolves branch heads, published release metadata and existing
archive provenance into a temporary publication plan. Each target uses its
exact clean code checkout, source Markdown, public figures and API mapping.
The shared VitePress theme supplies the complete version menu. Release pages
show their frozen commits and omit the edit-current-page action; branch pages
link edits to the matching code branch.

API generation, contract checks, CPU examples, site links, math rendering and
figures run before producing a complete versioned artifact. A failed build
retains the existing published site. Archive publication and Pages deployment
are serialized together. The publisher rechecks branch and tag identities,
preserves historical releases, writes a normal fast-forward archive commit,
and deploys that exact tree. No force push is part of routine publication.
Automated archive commits use the maintainer's configured author/committer
identity; GitHub retains its own workflow execution audit identity.

## Rebuild an existing formal release

Run the Documentation workflow from `main` or `dev_jinyuan` and set
`release_tag` to an existing published formal tag, such as `v0.1.0`. The tag
and its recorded documentation source are not changed. This is also the
recovery entry point for rebuilding the complete archive after a failed run.
The generated `publication-plan.json`, per-version `version.json` and root
`versions.json` record the exact inputs and preview status.

## Local fixed-snapshot builds

`code-version.json` and `versions.json` preserve the original local snapshot
pins; automated publication resolves its own plan instead of editing these
files. Prepare one clean code worktree per required full SHA under
`DOCS_CODE_ROOT`, or use `DOCS_CODE_PATH` when every target shares one code
commit. Use Node.js 22 and compatible Julia. The independent `docs/publish`
environment supplies JSON, HTML/XML and HTTP support for publication and
validation; it does not change the package's runtime dependencies or plotting
backend.

```bash
julia --startup-file=no --project=docs/publish -e 'using Pkg; Pkg.instantiate()'
npm ci --prefix docs
export DOCS_CODE_ROOT=/absolute/path/to/prepared-code-checkouts
export DOCS_CHECK_EXAMPLES=1
npm run docs:build-site --prefix docs -- --preview
npm run docs:preview --prefix docs
```

The candidate label remains visible. Preview output is never a formal Pages
artifact. README, package dependencies and numerical defaults are independent
of this publishing setup.
