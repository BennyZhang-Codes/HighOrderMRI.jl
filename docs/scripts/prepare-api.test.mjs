import assert from 'node:assert/strict'
import { mkdtempSync, mkdirSync, readFileSync, rmSync, writeFileSync, existsSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import test from 'node:test'
import { prepareApi } from './prepare-api.mjs'

const sha = 'd141e7dc48e8d18380050aaa970ee4ae364d916d'

function fixture(t, { rootIsCheckout = true, head = sha, branch = null, dirty = '', missingGenerator = false, exportStatus = 0 } = {}) {
  const dir = mkdtempSync(join(tmpdir(), 'highordermri-docs-entry-'))
  t.after(() => rmSync(dir, { recursive: true, force: true }))
  const code = join(dir, 'code')
  const docs = join(dir, 'docs')
  const output = join(docs, '.generated', 'api')
  mkdirSync(join(code, 'src'), { recursive: true })
  mkdirSync(join(code, 'scripts', 'docs'), { recursive: true })
  mkdirSync(output, { recursive: true })
  writeFileSync(join(code, 'Project.toml'), 'name = "HighOrderMRI"\nuuid = "065cb43a-79ab-4c63-92af-604ac7366864"\n\n[deps]\n')
  writeFileSync(join(code, 'src', 'HighOrderMRI.jl'), 'module HighOrderMRI\nend\n')
  if (!missingGenerator) writeFileSync(join(code, 'scripts', 'docs', 'export_api.jl'), '# fixture only; never executed\n')
  writeFileSync(join(docs, 'package.json'), JSON.stringify({ name: 'highordermri-docs', devDependencies: { vitepress: '2.0.0-alpha.19' } }))
  writeFileSync(join(docs, 'code-version.json'), JSON.stringify({ schema_version: 1, code_sha: sha }))
  writeFileSync(join(docs, 'api-map.toml'), 'schema_version = 1\n')
  writeFileSync(join(output, 'build-meta.json'), '{"stale":true}')
  writeFileSync(join(output, 'stale.md'), 'old fragment')
  const calls = []
  const run = (command, args, options) => {
    calls.push({ command, args, options })
    if (command === 'git') {
      if (args.join(' ') === 'rev-parse --show-toplevel') return { status: 0, stdout: `${rootIsCheckout ? code : dir}\n` }
      if (args.join(' ') === 'rev-parse HEAD') return { status: 0, stdout: `${head}\n` }
      if (args[0] === 'symbolic-ref') return { status: branch ? 0 : 1, stdout: branch ? `${branch}\n` : '' }
      if (args[0] === 'status') return { status: 0, stdout: dirty }
      assert.fail(`Unexpected Git command: ${args.join(' ')}`)
    }
    assert.equal(command, 'fixture-julia')
    const generated = args[args.indexOf('--output') + 1]
    writeFileSync(join(generated, 'find-delay.md'), 'fixture fragment\n')
    writeFileSync(join(generated, 'index.toml'), 'schema_version = 1\n')
    return { status: exportStatus }
  }
  const env = {
    DOCS_CODE_PATH: code, JULIA: 'fixture-julia', JULIA_DEPOT_PATH: '/private/depot',
    JULIA_LOAD_PATH: 'unsafe-global-fallback', CUDA_VISIBLE_DEVICES: '0,1', MPLBACKEND: 'QtAgg',
  }
  const invoke = (argv = [], overrideEnv = env) => prepareApi({ argv, env: overrideEnv, docsRoot: docs, run, log: () => {} })
  return { dir, code, docs, output, calls, env, invoke }
}

test('requires explicit source path and removes stale generated files on failure', (t) => {
  const f = fixture(t)
  assert.throws(() => f.invoke(['--preview'], {}), /DOCS_CODE_PATH is required/)
  assert.equal(f.calls.length, 0)
  assert.equal(existsSync(f.output), false)
})

test('only is forbidden in a formal build', (t) => {
  const f = fixture(t)
  assert.throws(() => f.invoke(['--only', 'find-delay']), /only with --preview/)
  assert.equal(f.calls.length, 0)
})

test('formal build refuses copied code nested in an ancestor checkout', (t) => {
  const f = fixture(t, { rootIsCheckout: false })
  assert.throws(() => f.invoke(), /itself to be a Git checkout root/)
  assert.equal(f.calls.filter((x) => x.command !== 'git').length, 0)
})

test('formal build refuses a different HEAD', (t) => {
  const f = fixture(t, { head: 'a'.repeat(40) })
  assert.throws(() => f.invoke(), /differs from configured code_sha/)
})

test('formal build refuses changed docstrings', (t) => {
  const f = fixture(t, { dirty: ' M src/Synch/FindDelay.jl\n' })
  assert.throws(() => f.invoke(), /require a clean code checkout/)
})

test('formal build refuses untracked source and tools', (t) => {
  const f = fixture(t, { dirty: '?? scripts/docs/export_api.jl\n' })
  assert.throws(() => f.invoke(), /require a clean code checkout/)
})

test('a website branch cannot provide implementation source even in preview', (t) => {
  const f = fixture(t, { branch: 'docs' })
  assert.throws(() => f.invoke(['--preview']), /website branch/)
})

test('missing exporter requires a code commit containing the tool', (t) => {
  const f = fixture(t, { missingGenerator: true })
  assert.throws(() => f.invoke(), /code commit containing this tool/)
  assert.equal(f.calls.filter((x) => x.command !== 'git').length, 0)
})

test('validates full code SHA and package identity', (t) => {
  const f = fixture(t)
  writeFileSync(join(f.docs, 'code-version.json'), JSON.stringify({ schema_version: 1, code_sha: 'd141e7d' }))
  assert.throws(() => f.invoke(['--preview']), /full lowercase 40-character/)
  writeFileSync(join(f.docs, 'code-version.json'), JSON.stringify({ schema_version: 1, code_sha: sha }))
  writeFileSync(join(f.code, 'Project.toml'), 'name = "Other"\nuuid = "065cb43a-79ab-4c63-92af-604ac7366864"\n')
  assert.throws(() => f.invoke(['--preview']), /matching name and UUID/)
})

test('preview passes safe argv/env, does not claim ancestor HEAD, and is deterministic', (t) => {
  const f = fixture(t, { rootIsCheckout: false })
  const first = f.invoke(['--preview', '--only', 'find-delay'])
  const firstBytes = readFileSync(join(f.output, 'build-meta.json'), 'utf8')
  assert.equal(existsSync(join(f.output, 'stale.md')), false)
  assert.equal(first.actualHead, null)
  assert.equal(first.sourceClean, null)
  assert.equal(first.preview, true)
  assert.equal(first.codeRef, sha)
  const call = f.calls.find((x) => x.command === 'fixture-julia')
  assert.equal(call.options.shell, false)
  assert.equal(call.options.cwd, f.code)
  assert.equal(call.options.env.CUDA_VISIBLE_DEVICES, '')
  assert.equal(call.options.env.MPLBACKEND, 'Agg')
  assert.equal(call.options.env.JULIA_LOAD_PATH, '@:@stdlib')
  assert.equal(call.options.env.JULIA_DEPOT_PATH, '/private/depot')
  assert.equal(call.options.env.JULIA_PKG_PRECOMPILE_AUTO, '0')
  assert.deepEqual(call.args, [
    '--startup-file=no', `--project=${f.code}`, join(f.code, 'scripts', 'docs', 'export_api.jl'),
    '--map', join(f.docs, 'api-map.toml'), '--output', f.output, '--code-root', f.code,
    '--code-sha', sha, '--mode', 'preview', '--only', 'find-delay',
  ])
  f.invoke(['--preview', '--only', 'find-delay'])
  assert.equal(readFileSync(join(f.output, 'build-meta.json'), 'utf8'), firstBytes)
})

test('formal entry delegates approved-manifest validation in build mode', (t) => {
  const f = fixture(t)
  const result = f.invoke()
  assert.equal(result.preview, false)
  assert.equal(result.actualHead, sha)
  assert.equal(result.sourceClean, true)
  const call = f.calls.find((x) => x.command === 'fixture-julia')
  assert.equal(call.args[call.args.indexOf('--mode') + 1], 'build')
  assert.equal(call.args.includes('--only'), false)
})

test('export failure removes both old and partially generated fragments', (t) => {
  const f = fixture(t, { exportStatus: 2 })
  assert.throws(() => f.invoke(['--preview', '--only', 'find-delay']), /exit 2/)
  assert.equal(existsSync(f.output), false)
})


test('preview accepts multiple distinct API ids and rejects ambiguous selections', (t) => {
  const f = fixture(t)
  const result = f.invoke(['--preview', '--only', 'find-delay,highorderlowrankop'])
  assert.equal(result.only, 'find-delay,highorderlowrankop')
  const call = f.calls.find(x => x.command === 'fixture-julia')
  assert.equal(call.args[call.args.indexOf('--only') + 1], 'find-delay,highorderlowrankop')
  for (const ids of ['find-delay,find-delay', 'find-delay,', ',find-delay', '../outside', 'find-delay, highorderlowrankop']) {
    assert.throws(() => f.invoke(['--preview', '--only', ids]), /unique comma-separated/)
  }
})
