import { createHash } from 'node:crypto'
import { spawnSync } from 'node:child_process'
import { existsSync, mkdirSync, readFileSync, realpathSync, rmSync, statSync, writeFileSync } from 'node:fs'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath, pathToFileURL } from 'node:url'

const defaultDocsRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..')
const packageUuid = '065cb43a-79ab-4c63-92af-604ac7366864'
const usage = `Usage: node docs/scripts/prepare-api.mjs [--preview [--only <api-id[,api-id...]>]]

DOCS_CODE_PATH must name a separately prepared HighOrderMRI package root.
No checkout, package installation, or source changes are performed here.
CI must checkout code-version.json's exact code_sha before invoking this tool.
Preview accepts an explicit source copy; a formal build requires a clean Git
checkout at code_sha and all API entries approved (checked by the exporter).
JULIA may name the Julia executable. JULIA_DEPOT_PATH is passed through.
`

function parseArguments(argv) {
  let preview = false
  let only = null
  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i]
    if (arg === '--preview' && !preview) preview = true
    else if (arg === '--only' && only === null) {
      only = argv[++i]
      const ids = only?.split(',') || []
      if (!ids.length || !ids.every(id => /^[a-z0-9]+(?:-[a-z0-9]+)*$/.test(id)) || new Set(ids).size !== ids.length) {
        throw new Error('--only requires unique comma-separated API ids such as find-delay,highorderlowrankop.')
      }
    } else throw new Error(`Unknown or duplicate argument: ${arg}`)
  }
  if (only && !preview) throw new Error('--only is allowed only with --preview; formal builds must cover the full manifest.')
  return { preview, only }
}

function requireFile(path, description) {
  if (!existsSync(path) || !statSync(path).isFile()) throw new Error(`Missing ${description}: ${path}`)
}

function jsonObject(path, description) {
  requireFile(path, description)
  let value
  try { value = JSON.parse(readFileSync(path, 'utf8')) }
  catch (error) { throw new Error(`Invalid ${description}: ${error.message}`) }
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    throw new Error(`${description} must contain a JSON object.`)
  }
  return value
}

function fingerprint(path) {
  return createHash('sha256').update(readFileSync(path)).digest('hex')
}

function gitSourceState(codeRoot, env, run) {
  const options = { cwd: codeRoot, env, encoding: 'utf8', shell: false }
  const top = run('git', ['rev-parse', '--show-toplevel'], options)
  if (top.error || top.status !== 0) return { kind: 'directory-copy', head: null, branch: null, clean: null }
  let repositoryRoot
  try { repositoryRoot = realpathSync(top.stdout.trim()) }
  catch { return { kind: 'directory-copy', head: null, branch: null, clean: null } }
  // A copied package inside some other checkout is not that checkout's source.
  if (repositoryRoot !== codeRoot) return { kind: 'directory-copy', head: null, branch: null, clean: null }
  const head = run('git', ['rev-parse', 'HEAD'], options)
  const branch = run('git', ['symbolic-ref', '--quiet', '--short', 'HEAD'], options)
  const state = run('git', ['status', '--porcelain=v1', '--untracked-files=all'], options)
  if (head.error || head.status !== 0 || !/^[0-9a-f]{40}$/.test(head.stdout.trim()) || state.error || state.status !== 0) {
    throw new Error('Cannot verify the code checkout HEAD and clean status.')
  }
  if (branch.error || ![0, 1].includes(branch.status)) throw new Error('Cannot verify the code checkout branch.')
  return {
    kind: 'git-checkout',
    head: head.stdout.trim(),
    branch: branch.status === 0 ? branch.stdout.trim() : null,
    clean: state.stdout.length === 0,
  }
}

/** Prepare API artifacts only; never mutate the selected code or website pages. */
export function prepareApi({ argv = process.argv.slice(2), env = process.env, docsRoot = defaultDocsRoot, run = spawnSync, log = console.log } = {}) {
  if (argv.length === 1 && ['--help', '-h'].includes(argv[0])) {
    log(usage)
    return null
  }
  const root = realpathSync(docsRoot)
  const packageInfo = jsonObject(join(root, 'package.json'), 'VitePress package.json')
  if (packageInfo.name !== 'highordermri-docs' || !packageInfo.devDependencies?.vitepress) {
    throw new Error('The docs path must be the HighOrderMRI VitePress package root.')
  }
  const output = join(root, '.generated', 'api')
  // Never let a failed or newly partial run leave previously usable fragments.
  rmSync(output, { recursive: true, force: true })
  try {
    const { preview, only } = parseArguments(argv)
    const config = jsonObject(join(root, 'code-version.json'), 'code-version.json')
    if (config.schema_version !== 1 || typeof config.code_sha !== 'string' || !/^[0-9a-f]{40}$/.test(config.code_sha)) {
      throw new Error('code-version.json requires schema_version=1 and a full lowercase 40-character code_sha.')
    }
    if (!env.DOCS_CODE_PATH?.trim()) {
      throw new Error('DOCS_CODE_PATH is required. Prepare the code checkout explicitly (CI must checkout code_sha); this tool never falls back to code on the docs branch.')
    }
    const codeRoot = realpathSync(resolve(env.DOCS_CODE_PATH))
    if (!statSync(codeRoot).isDirectory()) throw new Error('DOCS_CODE_PATH must be a package directory.')
    const project = join(codeRoot, 'Project.toml')
    const moduleFile = join(codeRoot, 'src', 'HighOrderMRI.jl')
    requireFile(project, 'HighOrderMRI Project.toml')
    requireFile(moduleFile, 'HighOrderMRI source module')
    // Read only top-level identity fields; authoritative TOML/API checks are Julia's.
    const projectHeader = readFileSync(project, 'utf8').split(/^\s*\[/m, 1)[0]
    if (!/^name\s*=\s*"HighOrderMRI"\s*(?:#.*)?$/m.test(projectHeader) ||
        !new RegExp(`^uuid\\s*=\\s*"${packageUuid}"\\s*(?:#.*)?$`, 'm').test(projectHeader)) {
      throw new Error('DOCS_CODE_PATH must point directly to the HighOrderMRI package root (matching name and UUID).')
    }
    const source = gitSourceState(codeRoot, env, run)
    if (['docs', 'docs-modern-ui'].includes(source.branch)) {
      throw new Error('The selected checkout is a website branch. Supply the explicit code checkout or preview code copy instead.')
    }
    if (!preview) {
      if (source.kind !== 'git-checkout') throw new Error('Formal builds require DOCS_CODE_PATH itself to be a Git checkout root, not a copied directory or its ancestor repository.')
      if (source.head !== config.code_sha) throw new Error(`Code HEAD ${source.head} differs from configured code_sha ${config.code_sha}.`)
      if (!source.clean) throw new Error('Formal builds require a clean code checkout, including tracked code/docstrings and untracked files.')
    }
    const generator = join(codeRoot, 'scripts', 'docs', 'export_api.jl')
    if (!existsSync(generator) || !statSync(generator).isFile()) {
      throw new Error(`Missing API exporter at ${generator}. Formal code_sha must refer to a code commit containing this tool; prepare and approve that commit before updating code-version.json. No fallback exporter will be used.`)
    }
    const map = join(root, 'api-map.toml')
    requireFile(map, 'API migration manifest')
    const mode = preview ? 'preview' : 'build'
    const args = [
      '--startup-file=no', `--project=${codeRoot}`, generator,
      '--map', map, '--output', output, '--code-root', codeRoot,
      '--code-sha', config.code_sha, '--mode', mode,
    ]
    if (only) args.push('--only', only)
    mkdirSync(output, { recursive: true })
    const result = run(env.JULIA || 'julia', args, {
      cwd: codeRoot,
      shell: false,
      stdio: 'inherit',
      env: {
        ...env,
        CUDA_VISIBLE_DEVICES: '',
        MPLBACKEND: 'Agg',
        JULIA_LOAD_PATH: '@:@stdlib',
        JULIA_PKG_PRECOMPILE_AUTO: '0',
      },
    })
    if (result.error) throw new Error(`Cannot run Julia API exporter: ${result.error.message}`)
    if (result.status !== 0) throw new Error(`API exporter failed (${result.signal ? `signal ${result.signal}` : `exit ${result.status}`}); generated fragments were discarded.`)
    const index = join(output, 'index.toml')
    requireFile(index, 'API exporter index.toml')
    const metadata = {
      schema_version: 1,
      codeRef: config.code_sha,
      preview,
      mode,
      only,
      codeRoot,
      sourceKind: source.kind,
      actualHead: source.head,
      sourceBranch: source.branch,
      sourceClean: source.clean,
      apiMapSha256: fingerprint(map),
      projectSha256: fingerprint(project),
      generatorSha256: fingerprint(generator),
      generatedIndexSha256: fingerprint(index),
    }
    writeFileSync(join(output, 'build-meta.json'), `${JSON.stringify(metadata, null, 2)}\n`)
    log(`API ${mode} prepared: ${output}; configured code ref ${config.code_sha}${preview ? ' (candidate source; not a release build)' : ''}`)
    return metadata
  } catch (error) {
    rmSync(output, { recursive: true, force: true })
    throw error
  }
}

if (process.argv[1] && pathToFileURL(resolve(process.argv[1])).href === import.meta.url) {
  try { prepareApi() }
  catch (error) {
    console.error(`API preparation failed: ${error.message}`)
    process.exitCode = 1
  }
}
