import assert from 'node:assert/strict'
import { spawnSync } from 'node:child_process'
import { mkdtempSync, mkdirSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { fileURLToPath } from 'node:url'
import test from 'node:test'

const runner = fileURLToPath(new URL('./check-version.jl', import.meta.url))
const checks = ['test_contract_check.jl', 'test_export_api.jl', 'check_examples.jl']

function fixture(t) {
  const dir = mkdtempSync(join(tmpdir(), 'highordermri-docs-check-paths-'))
  t.after(() => rmSync(dir, { recursive: true, force: true }))
  const code = join(dir, 'code')
  const docs = join(dir, 'website')
  mkdirSync(code)
  mkdirSync(docs)
  const addTools = (layout, missing = null) => {
    const tools = layout === 'current' ? join(code, 'docs', 'scripts') : join(code, 'scripts', 'docs')
    mkdirSync(tools, { recursive: true })
    writeFileSync(join(tools, 'export_api.jl'), '# fixture exporter\n')
    for (const name of checks) {
      if (name !== missing) writeFileSync(join(tools, name), `println("${layout}:${name}")\n`)
    }
  }
  const invoke = () => spawnSync(process.env.JULIA || 'julia', ['--startup-file=no', '--history-file=no', runner, code, docs], {
    encoding: 'utf8', timeout: 30000,
    env: { ...process.env, CUDA_VISIBLE_DEVICES: '', JULIA_NUM_THREADS: '1', JULIA_LOAD_PATH: '@:@stdlib' },
  })
  return { code, addTools, invoke }
}

test('version checks use consolidated tools from the selected code checkout', (t) => {
  const f = fixture(t)
  f.addTools('current')
  const result = f.invoke()
  assert.ifError(result.error)
  assert.equal(result.status, 0, result.stderr)
  assert.equal(result.stdout, checks.map(name => `current:${name}\n`).join(''))
})

test('historical version checks accept frontend scripts without a consolidated exporter', (t) => {
  const f = fixture(t)
  f.addTools('legacy')
  mkdirSync(join(f.code, 'docs', 'scripts'), { recursive: true })
  writeFileSync(join(f.code, 'docs', 'scripts', 'prepare-api.mjs'), '// frontend only\n')
  const result = f.invoke()
  assert.ifError(result.error)
  assert.equal(result.status, 0, result.stderr)
  assert.equal(result.stdout, checks.map(name => `legacy:${name}\n`).join(''))
})

test('version checks prefer consolidated tools when both layouts exist', (t) => {
  const f = fixture(t)
  f.addTools('current')
  f.addTools('legacy')
  const result = f.invoke()
  assert.ifError(result.error)
  assert.equal(result.status, 0, result.stderr)
  assert.equal(result.stdout, checks.map(name => `current:${name}\n`).join(''))
})

test('an incomplete consolidated suite fails before running or mixing historical checks', (t) => {
  const f = fixture(t)
  f.addTools('current', 'check_examples.jl')
  f.addTools('legacy')
  const result = f.invoke()
  assert.ifError(result.error)
  assert.notEqual(result.status, 0)
  assert.equal(result.stdout, '')
  assert.match(result.stderr, /Missing documentation tool in selected code checkout: check_examples\.jl/)
})

test('missing code-side documentation tools fail before running checks', (t) => {
  const f = fixture(t)
  const result = f.invoke()
  assert.ifError(result.error)
  assert.notEqual(result.status, 0)
  assert.equal(result.stdout, '')
  assert.match(result.stderr, /Missing documentation tool in selected code checkout: export_api\.jl/)
})
