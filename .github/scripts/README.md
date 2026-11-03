# Pages deployment checks and recovery

Normal documentation builds still generate the API, run the CPU examples and
build VitePress. After Pages reports a successful deployment, the live-site check
verifies the homepage, notation, pinned API source links and both figure themes.

If Pages stalls after a successful build, `artifact_run_id` can republish that
checked artifact without repeating the Julia build:

```bash
gh workflow run documentation.yml --ref docs -f artifact_run_id=SOURCE_RUN_ID
```

Recovery requires a **new docs commit SHA** and an unchanged `docs/` Git tree.
It refuses to reuse the source deployment's commit ID. The source must be a
completed `docs`-branch run of this workflow in the same repository, with a
successful checked build and exactly one unexpired Pages artifact. A failed
deployment does not invalidate its successful build.

The recovery helper checks these identities and the configured code revision,
then safely extracts the static files and uploads them under the new workflow
run. Changed website inputs require a normal build. Recovery does not change
Pages settings, skip source-build validation or guarantee that the Pages service
will accept a new deployment.

Run the helper tests with Python 3.11 or later:

```bash
python3 -m unittest discover -s .github/scripts -p 'test_*.py'
```
