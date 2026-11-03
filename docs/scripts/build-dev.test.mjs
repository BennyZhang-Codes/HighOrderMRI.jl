import assert from 'node:assert/strict'
import { existsSync, mkdirSync, mkdtempSync, readFileSync, readdirSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import test from 'node:test'
import { buildDev } from './build-dev.mjs'

const sha = 'd141e7dc48e8d18380050aaa970ee4ae364d916d'

function fixture(t, { preparationError = null, processError = null, status = 0, produceIndex = true, checkStatus = 0 } = {}) {
  const docs = mkdtempSync(join(tmpdir(), 'highordermri-dev-build-'))
  t.after(() => rmSync(docs, { recursive: true, force: true }))
  const pages = join(docs, '.pages')
  const temporary = join(docs, '.generated', 'dev-site')
  mkdirSync(join(pages, 'stable'), { recursive: true })
  mkdirSync(join(pages, 'dev'), { recursive: true })
  mkdirSync(temporary, { recursive: true })
  writeFileSync(join(pages, 'index.html'), 'stale redirect')
  writeFileSync(join(pages, 'stable', 'index.html'), 'stale release')
  writeFileSync(join(pages, 'dev', 'index.html'), 'stale dev')
  writeFileSync(join(temporary, 'stale.html'), 'incomplete previous build')
  const code = join(docs, 'fixture-code')
  mkdirSync(join(code, 'scripts', 'docs'), { recursive: true })
  writeFileSync(join(code, 'scripts', 'docs', 'check_site.py'), '# test fixture')
  const calls = []
  const entry = join(docs, 'fake-vitepress.js')
  const prepare = (options) => {
    calls.push({ step: 'prepare', options })
    if (preparationError) throw new Error(preparationError)
    return { codeRoot: code, codeRef: sha, preview: options.argv.includes('--preview') }
  }
  const resolveVitepress = (root) => {
    calls.push({ step: 'resolve', root })
    return entry
  }
  const run = (command, args, options) => {
    if (args[0].endsWith('check_site.py')) {
      calls.push({ step: 'check', command, args, options })
      return { status: checkStatus }
    }
    calls.push({ step: 'build', command, args, options })
    const destination = args[args.indexOf('--outDir') + 1]
    if (produceIndex) writeFileSync(join(destination, 'index.html'), '<html>new dev</html>')
    writeFileSync(join(destination, 'partial.html'), 'generated content')
    return { status, error: processError }
  }
  const env = {
    DOCS_CODE_PATH: '/prepared/code', JULIA_DEPOT_PATH: '/private/depot',
    DOCS_VERSION: 'stable', DOCS_BASE: '/wrong/', DOCS_SOURCE_REF: 'wrong',
    DOCS_EDIT_REF: 'main', DOCS_STABLE_VERSION: 'v1', DOCS_RELEASE_VERSIONS: 'v1,v2',
  }
  const invoke = (argv = []) => buildDev({ argv, env, docsRoot: docs, prepare, resolveVitepress, run, log: () => {} })
  return { docs, pages, temporary, entry, env, calls, invoke }
}

test('preview arguments reach preparation before a dev-only, shell-free build', (t) => {
  const f = fixture(t)
  const argv = ['--preview', '--only', 'find-delay']
  const result = f.invoke(argv)
  assert.deepEqual(f.calls.map((x) => x.step), ['prepare', 'resolve', 'build', 'check'])
  assert.deepEqual(f.calls[0].options.argv, argv)
  assert.equal(f.calls[0].options.env.DOCS_CODE_PATH, '/prepared/code')
  const call = f.calls[2]
  assert.equal(call.command, process.execPath)
  assert.deepEqual(call.args, [f.entry, 'build', join(f.docs, 'src'), '--outDir', f.temporary])
  assert.equal(call.options.shell, false)
  assert.equal(call.options.env.DOCS_VERSION, 'dev')
  assert.equal(call.options.env.DOCS_BASE, '/HighOrderMRI.jl/dev/')
  assert.equal(call.options.env.DOCS_SOURCE_REF, sha)
  assert.equal(call.options.env.DOCS_EDIT_REF, 'docs')
  assert.equal(call.options.env.DOCS_STABLE_VERSION, '')
  assert.equal(call.options.env.DOCS_RELEASE_VERSIONS, '')
  assert.equal(call.options.env.JULIA_DEPOT_PATH, '/private/depot')
  assert.deepEqual(readdirSync(f.pages).sort(), ['dev', 'index.html'])
  assert.equal(readFileSync(join(f.pages, 'dev', 'index.html'), 'utf8'), '<html>new dev</html>')
  assert.equal(existsSync(join(f.pages, 'dev', 'stale.html')), false)
  assert.equal(existsSync(f.temporary), false)
  assert.match(readFileSync(join(f.pages, 'index.html'), 'utf8'), /url=\.\/dev\//)
  assert.equal(result.preview, true)
})

test('formal build forwards no preview override to the hard gates', (t) => {
  const f = fixture(t)
  assert.equal(f.invoke().preview, false)
  assert.deepEqual(f.calls[0].options.argv, [])
})

test('preparation rejection prevents VitePress and discards previous output', (t) => {
  const f = fixture(t, { preparationError: 'manifest is not approved' })
  assert.throws(() => f.invoke(), /manifest is not approved/)
  assert.deepEqual(f.calls.map((x) => x.step), ['prepare'])
  assert.equal(existsSync(f.pages), false)
  assert.equal(existsSync(f.temporary), false)
})

test('failed VitePress process leaves no old or partial site', (t) => {
  const f = fixture(t, { status: 1 })
  assert.throws(() => f.invoke(['--preview']), /VitePress failed \(exit 1\)/)
  assert.equal(existsSync(f.pages), false)
  assert.equal(existsSync(f.temporary), false)
})

test('process launch error leaves no generated site', (t) => {
  const f = fixture(t, { processError: new Error('fixture launch failure') })
  assert.throws(() => f.invoke(['--preview']), /Cannot run VitePress/)
  assert.equal(existsSync(f.pages), false)
  assert.equal(existsSync(f.temporary), false)
})

test('success exit without index.html is not treated as a finished build', (t) => {
  const f = fixture(t, { produceIndex: false })
  assert.throws(() => f.invoke(['--preview']), /did not produce the dev index/)
  assert.equal(existsSync(f.pages), false)
  assert.equal(existsSync(f.temporary), false)
})

test('help does not prepare, build, or remove artifacts', (t) => {
  const f = fixture(t)
  assert.equal(f.invoke(['--help']), null)
  assert.equal(f.calls.length, 0)
  assert.equal(readFileSync(join(f.pages, 'index.html'), 'utf8'), 'stale redirect')
})

test('link or API check failure discards the site', (t) => {
  const f = fixture(t, { checkStatus: 1 })
  assert.throws(() => f.invoke(['--preview']), /Built-site checks failed/)
  assert.equal(existsSync(f.pages), false)
})
