import assert from 'node:assert/strict'
import { spawnSync } from 'node:child_process'
import { cpSync, existsSync, mkdirSync, mkdtempSync, readFileSync, readdirSync, renameSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import test from 'node:test'
import { buildSite } from './build-site.mjs'
import { compareVersions, loadVersions } from './versions.mjs'

const actualDocs = join(dirname(fileURLToPath(import.meta.url)), '..')

function fixture(t) {
  const repo = mkdtempSync(join(tmpdir(), 'highordermri-versions-test-'))
  t.after(() => rmSync(repo, { recursive: true, force: true }))
  const docs = join(repo, 'docs')
  mkdirSync(join(docs, 'src', '.vitepress'), { recursive: true })
  mkdirSync(join(docs, 'figures'))
  writeFileSync(join(docs, 'figures', 'source.tex'), '% frozen figure source')
  writeFileSync(join(docs, 'src', 'index.md'), 'frozen release prose')
  writeFileSync(join(docs, 'src', '.vitepress', 'config.mts'), 'export default {}')
  writeFileSync(join(docs, 'api-map.toml'), 'schema_version = 1')
  cpSync(join(actualDocs, 'package.json'), join(docs, 'package.json'))
  // The mocked renderer never loads packages. Its CLI path is resolved from
  // this real dependency installation to exercise the normal entry point.
  const link = spawnSync('ln', ['-s', join(actualDocs, 'node_modules'), join(docs, 'node_modules')])
  assert.equal(link.status, 0)
  function git(...args) {
    const result = spawnSync('git', args, { cwd: repo, encoding: 'utf8' })
    assert.equal(result.status, 0, result.stderr)
    return result.stdout.trim()
  }
  git('init', '--quiet')
  git('add', 'docs/src', 'docs/api-map.toml', 'docs/figures', 'docs/package.json')
  git('-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid', 'commit', '--quiet', '-m', 'Frozen website content')
  const sha = git('rev-parse', 'HEAD')
  const config = { schema_version: 1, stable: 'v0.1.0', releases: [{ version: 'v0.1.0', code_sha: sha, docs_sha: sha }] }
  writeFileSync(join(docs, 'code-version.json'), JSON.stringify({ schema_version: 1, code_sha: sha }))
  const save = () => writeFileSync(join(docs, 'versions.json'), JSON.stringify(config))
  save()
  writeFileSync(join(docs, 'src', 'index.md'), 'new dev prose')
  let failVersion = null
  const calls = []
  const run = (command, args, options) => {
    if (command === 'git' || command === 'tar') return spawnSync(command, args, options)
    calls.push({ command, args, options })
    if (command === process.execPath) {
      const version = options.env.DOCS_VERSION
      if (version === failVersion) return { status: 1 }
      const source = args[args.indexOf('build') + 1]
      const out = args[args.indexOf('--outDir') + 1]
      mkdirSync(out, { recursive: true })
      writeFileSync(join(out, 'index.html'), readFileSync(join(source, 'index.md')))
    }
    return { status: 0 }
  }
  const prepare = ({ docsRoot, argv }) => {
    const current = JSON.parse(readFileSync(join(docsRoot, 'code-version.json')))
    mkdirSync(join(docsRoot, '.generated', 'api'), { recursive: true })
    writeFileSync(join(docsRoot, '.generated', 'api', 'index.toml'), 'fixture API')
    return { codeRef: current.code_sha, actualHead: current.code_sha, sourceClean: true, preview: argv.includes('--preview'), apiMapSha256: 'fixture' }
  }
  const invoke = (argv = ['--preview'], extraEnv = {}) => buildSite({ argv, docsRoot: docs, env: { DOCS_CODE_PATH: '/explicit/source', ...extraEnv }, run, prepare, log: () => {} })
  return { repo, docs, sha, config, save, git, invoke, calls, setFailure: version => { failVersion = version } }
}

test('candidate builds dev, frozen release and stable together', t => {
  const f = fixture(t)
  const result = f.invoke()
  const pages = join(f.docs, '.pages')
  assert.deepEqual(readdirSync(pages).sort(), ['dev', 'index.html', 'stable', 'v0.1.0', 'versions.json'])
  assert.equal(readFileSync(join(pages, 'dev/index.html'), 'utf8'), 'new dev prose')
  assert.equal(readFileSync(join(pages, 'v0.1.0/index.html'), 'utf8'), 'frozen release prose')
  assert.equal(readFileSync(join(pages, 'stable/index.html'), 'utf8'), 'frozen release prose')
  assert.match(readFileSync(join(pages, 'index.html'), 'utf8'), /\.\/stable\//)
  assert.equal(result.versions.length, 3)
  assert.ok(result.versions.every(version => version.preview))
  assert.deepEqual(f.calls.filter(call => call.command === process.execPath).map(call => call.options.env.DOCS_BASE), ['/HighOrderMRI.jl/dev/', '/HighOrderMRI.jl/v0.1.0/', '/HighOrderMRI.jl/stable/'])
})

test('adding a release preserves older versions and advances stable', t => {
  const f = fixture(t)
  f.invoke()
  f.config.releases.push({ version: 'v0.2.0', code_sha: f.sha, docs_sha: f.sha })
  f.config.stable = 'v0.2.0'
  f.save()
  f.invoke()
  const manifest = JSON.parse(readFileSync(join(f.docs, '.pages/versions.json')))
  assert.equal(manifest.stable, 'v0.2.0')
  assert.deepEqual(manifest.versions.map(version => version.version), ['dev', 'v0.2.0', 'v0.1.0', 'stable'])
  assert.ok(existsSync(join(f.docs, '.pages/v0.1.0/index.html')))
})

test('failed release build preserves the previous complete artifact', t => {
  const f = fixture(t)
  f.invoke()
  const before = readFileSync(join(f.docs, '.pages/versions.json'), 'utf8')
  f.setFailure('v0.1.0')
  assert.throws(() => f.invoke(), /failed/)
  assert.equal(readFileSync(join(f.docs, '.pages/versions.json'), 'utf8'), before)
  assert.equal(readFileSync(join(f.docs, '.pages/dev/index.html'), 'utf8'), 'new dev prose')
  assert.equal(existsSync(join(f.docs, '.generated/version-build')), false)
})

test('formal builds require matching registered tags; unpublished tags are not releases', t => {
  const f = fixture(t)
  assert.throws(() => f.invoke([]), /git failed/)
  f.git('tag', 'v0.1.0')
  assert.equal(f.invoke([]).preview, false)
  f.git('tag', 'v0.0.9')
  assert.equal(f.invoke([]).preview, false)
})

test('only dev, stable and formal versions appear in navigation', t => {
  const f = fixture(t)
  const result = f.invoke()
  assert.deepEqual(result.versions.map(version => version.version), ['dev', 'v0.1.0', 'stable'])
  assert.throws(() => readFileSync(join(f.docs, '.pages/main/index.html')), /ENOENT/)
  const call = f.calls.find(call => call.command === process.execPath && call.options.env.DOCS_VERSION === 'dev')
  assert.ok(JSON.parse(call.options.env.DOCS_VERSION_ITEMS).every(item => item.text !== 'main'))
  writeFileSync(join(f.docs, 'code-version.json'), JSON.stringify({ schema_version: 1, code_sha: f.sha, main_sha: 'b'.repeat(40) }))
  assert.throws(() => f.invoke(), /main_sha/)
})

test('resolved publication plans bind each branch to committed in-tree prose', t => {
  const f = fixture(t)
  const release = f.config.releases[0]
  const plan = { schema_version: 1, source_sha: f.sha, stable: 'v0.1.0', releases: [release], targets: [
    { version: 'dev', code_sha: f.sha, docs_sha: f.sha },
    release, { ...release, version: 'stable', release: release.version },
  ] }
  const path = join(f.repo, 'plan.json')
  writeFileSync(path, JSON.stringify(plan))
  const result = f.invoke(['--preview'], { DOCS_BUILD_PLAN: path })
  assert.equal(result.versions.length, 3)
  assert.equal(readFileSync(join(f.docs, '.pages/dev/index.html'), 'utf8'), 'frozen release prose')
  assert.deepEqual(JSON.parse(readFileSync(join(f.docs, '.pages/publication-plan.json'))), plan)
  const cli = spawnSync(process.execPath, [join(actualDocs, 'scripts/versions.mjs'), f.docs, '--code-shas'], {
    env: { ...process.env, DOCS_BUILD_PLAN: path }, encoding: 'utf8',
  })
  assert.equal(cli.status, 0, cli.stderr)
  assert.equal(cli.stdout.trim(), f.sha)
  const withMain = { ...plan, targets: [...plan.targets, { version: 'main', code_sha: f.sha, docs_sha: f.sha }] }
  writeFileSync(path, JSON.stringify(withMain))
  assert.throws(() => f.invoke(['--preview'], { DOCS_BUILD_PLAN: path }), /Invalid release version/)
  plan.targets[0].docs_sha = 'invalid'
  writeFileSync(path, JSON.stringify(plan))
  assert.throws(() => f.invoke(['--preview'], { DOCS_BUILD_PLAN: path }), /publication target/)
})

test('retry restores a completed artifact after an interrupted replacement', t => {
  const f = fixture(t)
  f.invoke()
  const oldManifest = readFileSync(join(f.docs, '.pages/versions.json'), 'utf8')
  renameSync(join(f.docs, '.pages'), join(f.docs, '.generated/previous-pages'))
  f.setFailure('dev')
  assert.throws(() => f.invoke(), /failed/)
  assert.equal(readFileSync(join(f.docs, '.pages/versions.json'), 'utf8'), oldManifest)
})

test('version manifest rejects duplicates, unsafe paths and dangling stable', t => {
  const f = fixture(t)
  f.config.releases.push({ ...f.config.releases[0] })
  f.save()
  assert.throws(() => loadVersions(f.docs), /Duplicate/)
  f.config.releases.pop()
  f.config.releases[0].version = '../outside'
  f.save()
  assert.throws(() => loadVersions(f.docs), /Invalid release/)
  f.config.releases[0].version = 'v0.1.0'
  f.config.stable = 'v0.2.0'
  f.save()
  assert.throws(() => loadVersions(f.docs), /Stable/)
  assert.ok(compareVersions('v0.10.0', 'v0.9.0') > 0)
})
