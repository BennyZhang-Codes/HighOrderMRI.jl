import { spawnSync } from 'node:child_process'
import { createHash } from 'node:crypto'
import { cpSync, existsSync, mkdirSync, readFileSync, readdirSync, realpathSync, renameSync, rmSync, symlinkSync, writeFileSync } from 'node:fs'
import { createRequire } from 'node:module'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath, pathToFileURL } from 'node:url'
import { prepareApi } from './prepare-api.mjs'
import { loadVersions } from './versions.mjs'

const defaultDocsRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..')
const sitePath = '/HighOrderMRI.jl/'

function runChecked(run, command, args, options) {
  const result = run(command, args, { shell: false, ...options })
  if (result.error || result.status !== 0) throw new Error(`${command} failed: ${result.error?.message || result.stderr?.toString() || result.status}`)
  return result.stdout
}

function prepareInput(root, target, destination, run, env) {
  mkdirSync(destination, { recursive: true })
  if (target.docs_sha) {
    const archive = runChecked(run, 'git', ['archive', '--format=tar', target.docs_sha, 'docs/src', 'docs/api-map.toml', 'docs/figures'], {
      cwd: dirname(root), env, maxBuffer: 32 * 1024 * 1024,
    })
    runChecked(run, 'tar', ['-xf', '-', '-C', destination, '--strip-components=1'], { input: archive, env })
  } else {
    for (const name of ['src', 'api-map.toml', 'figures']) cpSync(join(root, name), join(destination, name), { recursive: true })
  }
  // Freeze release prose, figures and API mapping. Use the current theme for
  // every version so the navigation can include releases published later.
  rmSync(join(destination, 'src', '.vitepress'), { recursive: true, force: true })
  cpSync(join(root, 'src', '.vitepress'), join(destination, 'src', '.vitepress'), {
    recursive: true, filter: path => !['dist', 'cache'].includes(path.split('/').at(-1)),
  })
  cpSync(join(root, 'package.json'), join(destination, 'package.json'))
  symlinkSync(realpathSync(join(root, 'node_modules')), join(destination, 'node_modules'), 'dir')
  writeFileSync(join(destination, 'code-version.json'), JSON.stringify({ schema_version: 1, code_sha: target.code_sha }))
}

function rootRedirect() {
  return `<!doctype html><html lang="en"><head><meta charset="utf-8"><meta http-equiv="refresh" content="0; url=./stable/"><link rel="canonical" href="https://bennyzhang-codes.github.io/HighOrderMRI.jl/stable/"><title>HighOrderMRI.jl documentation</title></head><body><p><a href="./stable/">HighOrderMRI.jl stable documentation</a></p><script>location.replace('./stable/' + location.search + location.hash)</script></body></html>\n`
}

function pageFingerprint(directory) {
  const hash = createHash('sha256')
  for (const name of readdirSync(directory, { recursive: true }).sort()) {
    if (name.endsWith('.md')) hash.update(name).update(readFileSync(join(directory, name)))
  }
  return hash.digest('hex')
}

export function buildSite({ argv = process.argv.slice(2), env = process.env, docsRoot = defaultDocsRoot, run = spawnSync, prepare = prepareApi, log = console.log } = {}) {
  if (argv.length === 1 && ['--help', '-h'].includes(argv[0])) {
    log('Build dev, stable and all releases in versions.json. Use --preview for an unpublished candidate. DOCS_CODE_PATH or DOCS_CODE_ROOT is required.')
    return null
  }
  if (argv.length > 1 || (argv.length === 1 && argv[0] !== '--preview')) throw new Error('Only --preview is supported by the complete-site builder.')
  const preview = argv.includes('--preview')
  const root = realpathSync(docsRoot)
  const plan = loadVersions(root, env.DOCS_BUILD_PLAN)
  if (!preview) {
    for (const release of plan.releases) {
      const commit = runChecked(run, 'git', ['rev-parse', '--verify', `refs/tags/${release.version}^{commit}`], { cwd: dirname(root), env, encoding: 'utf8' }).trim()
      if (commit !== release.code_sha) throw new Error(`Tag ${release.version} does not match its code_sha.`)
    }
  }
  const temporary = join(root, '.generated', 'version-build')
  const stage = join(temporary, 'pages')
  const previous = join(root, '.generated', 'previous-pages')
  const pages = join(root, '.pages')
  if (existsSync(previous) && !existsSync(pages)) renameSync(previous, pages)
  rmSync(temporary, { recursive: true, force: true })
  mkdirSync(stage, { recursive: true })
  const require = createRequire(join(root, 'package.json'))
  const pkg = require.resolve('vitepress/package.json')
  const entry = resolve(dirname(pkg), JSON.parse(readFileSync(pkg, 'utf8')).bin.vitepress)
  const siteUrl = (env.DOCS_SITE_URL || 'https://bennyzhang-codes.github.io/HighOrderMRI.jl').replace(/\/$/, '')
  const items = [
    { text: `stable · ${plan.stable}`, link: `${siteUrl}/stable/` },
    { text: 'dev', link: `${siteUrl}/dev/` },
    ...plan.releases.map(release => ({ text: release.version, link: `${siteUrl}/${release.version}/` })),
  ]
  const records = []
  const preparedApis = new Map()
  const checkedExamples = new Set()
  try {
    for (const target of plan.targets) {
      const input = join(temporary, target.version)
      prepareInput(root, target, input, run, env)
      const codePath = env.DOCS_CODE_ROOT ? join(resolve(env.DOCS_CODE_ROOT), target.code_sha) : env.DOCS_CODE_PATH
      if (!codePath) throw new Error(`No separately prepared code checkout for ${target.version}.`)
      const mapHash = createHash('sha256').update(readFileSync(join(input, 'api-map.toml'))).digest('hex')
      const key = `${target.code_sha}:${codePath}:${mapHash}`
      const cached = preparedApis.get(key)
      let api
      if (cached) {
        mkdirSync(join(input, '.generated'), { recursive: true })
        cpSync(cached.directory, join(input, '.generated', 'api'), { recursive: true })
        api = cached.metadata
      } else {
        api = prepare({ argv: preview ? ['--preview'] : [], env: { ...env, DOCS_CODE_PATH: codePath }, docsRoot: input, run, log })
        preparedApis.set(key, { metadata: api, directory: join(input, '.generated', 'api') })
      }
      if (api.codeRef !== target.code_sha || api.actualHead !== target.code_sha || api.sourceClean !== true) {
        throw new Error(`Version ${target.version} requires its exact clean code commit, including in candidate builds.`)
      }
      const exampleKey = key + ':' + pageFingerprint(join(input, 'src'))
      if (env.DOCS_CHECK_EXAMPLES === '1' && !checkedExamples.has(exampleKey)) {
        runChecked(run, env.JULIA || 'julia', ['--startup-file=no', `--project=${codePath}`, join(root, 'scripts', 'check-version.jl'), codePath, input], {
          cwd: codePath, stdio: 'inherit', env: {
            ...env, CUDA_VISIBLE_DEVICES: '', MPLBACKEND: 'Agg', JULIA_LOAD_PATH: '@:@stdlib',
            DOCS_API_MAP: join(input, 'api-map.toml'), DOCS_SITE_PATH: input,
          },
        })
        checkedExamples.add(exampleKey)
      }
      const output = join(stage, target.version)
      runChecked(run, process.execPath, [entry, 'build', join(input, 'src'), '--outDir', output], {
        cwd: input, stdio: 'inherit', env: {
          ...env, DOCS_VERSION: target.version, DOCS_BASE: `${sitePath}${target.version}/`,
          DOCS_RELEASE_VERSION: target.release || target.version,
          DOCS_VERSION_ITEMS: JSON.stringify(items), DOCS_DOCS_SHA: target.docs_sha || '',
        },
      })
      if (!existsSync(join(output, 'index.html'))) throw new Error(`Missing ${target.version} index.html.`)
      const record = { ...target, preview, api_map_sha256: api.apiMapSha256 }
      writeFileSync(join(output, 'version.json'), JSON.stringify(record, null, 2) + '\n')
      records.push(record)
      runChecked(run, env.JULIA || 'julia', ['--startup-file=no', `--project=${join(root, 'publish')}`, join(root, 'publish', 'run.jl'), 'check-site', input, output, `${sitePath}${target.version}/`, target.version], { stdio: 'inherit', env })
    }
    writeFileSync(join(stage, 'versions.json'), JSON.stringify({ schema_version: 1, stable: plan.stable, preview, versions: records }, null, 2) + '\n')
    if (plan.source_sha) writeFileSync(join(stage, 'publication-plan.json'), JSON.stringify(plan, null, 2) + '\n')
    writeFileSync(join(stage, 'index.html'), rootRedirect())
    // Publish one complete artifact after every version passed. Preserve the
    // previous artifact if a build or replacement fails.
    if (existsSync(previous)) {
      if (existsSync(pages)) rmSync(previous, { recursive: true, force: true })
      else renameSync(previous, pages)
    }
    if (existsSync(pages)) renameSync(pages, previous)
    try { renameSync(stage, pages) }
    catch (error) { if (existsSync(previous)) renameSync(previous, pages); throw error }
    rmSync(previous, { recursive: true, force: true })
    log(`Complete documentation built at ${pages}: ${records.map(record => record.version).join(', ')}`)
    return { directory: pages, preview, stable: plan.stable, versions: records }
  } finally {
    rmSync(temporary, { recursive: true, force: true })
  }
}

if (process.argv[1] && pathToFileURL(resolve(process.argv[1])).href === import.meta.url) {
  try { buildSite() }
  catch (error) { console.error(`Documentation build failed: ${error.message}`); process.exitCode = 1 }
}
