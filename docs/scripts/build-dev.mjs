import { spawnSync } from 'node:child_process'
import { existsSync, mkdirSync, readFileSync, realpathSync, renameSync, rmSync, statSync, writeFileSync } from 'node:fs'
import { createRequire } from 'node:module'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath, pathToFileURL } from 'node:url'
import { prepareApi } from './prepare-api.mjs'

const defaultDocsRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..')
const usage = `Usage: node docs/scripts/build-dev.mjs [--preview [--only <api-id[,api-id...]>]]

Prepare API fragments, then build only /HighOrderMRI.jl/dev/ into docs/.pages.
Source selection and approval checks are enforced by prepare-api.mjs.
DOCS_CODE_PATH is required. This command never checks out tags or builds stable.
`

function vitepressEntry(docsRoot) {
  const require = createRequire(join(docsRoot, 'package.json'))
  let packagePath
  try { packagePath = require.resolve('vitepress/package.json') }
  catch { throw new Error('VitePress is unavailable. Install the documentation dependencies before building.') }
  const packageInfo = JSON.parse(readFileSync(packagePath, 'utf8'))
  const relativeBin = packageInfo.bin?.vitepress
  if (typeof relativeBin !== 'string') throw new Error('Installed VitePress package has no vitepress CLI entry.')
  const entry = resolve(dirname(packagePath), relativeBin)
  if (!existsSync(entry) || !statSync(entry).isFile()) throw new Error(`Missing VitePress CLI: ${entry}`)
  return entry
}

const redirect = `<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <meta http-equiv="refresh" content="0; url=./dev/">
    <link rel="canonical" href="https://bennyzhang-codes.github.io/HighOrderMRI.jl/dev/">
    <title>HighOrderMRI.jl documentation</title>
  </head>
  <body>
    <p>Redirecting to <a href="./dev/">HighOrderMRI.jl dev documentation</a>.</p>
    <script>location.replace('./dev/' + location.search + location.hash)</script>
  </body>
</html>
`

/** Build artifacts only. Approval/source gates remain in prepareApi. */
export function buildDev({ argv = process.argv.slice(2), env = process.env, docsRoot = defaultDocsRoot, run = spawnSync, prepare = prepareApi, resolveVitepress = vitepressEntry, log = console.log } = {}) {
  if (argv.length === 1 && ['--help', '-h'].includes(argv[0])) {
    log(usage)
    return null
  }
  const root = realpathSync(docsRoot)
  const pages = join(root, '.pages')
  const temporary = join(root, '.generated', 'dev-site')
  // .pages is disposable build output: do not publish stale versions or pages.
  rmSync(pages, { recursive: true, force: true })
  rmSync(temporary, { recursive: true, force: true })
  try {
    const api = prepare({ argv, env, docsRoot: root, log })
    if (!api || !/^[0-9a-f]{40}$/.test(api.codeRef) || typeof api.preview !== 'boolean') {
      throw new Error('API preparation did not return valid source metadata.')
    }
    const entry = resolveVitepress(root)
    mkdirSync(temporary, { recursive: true })
    const result = run(process.execPath, [entry, 'build', join(root, 'src'), '--outDir', temporary], {
      cwd: root,
      shell: false,
      stdio: 'inherit',
      env: {
        ...env,
        DOCS_VERSION: 'dev',
        DOCS_BASE: '/HighOrderMRI.jl/dev/',
        DOCS_SOURCE_REF: api.codeRef,
        DOCS_EDIT_REF: 'docs',
        DOCS_STABLE_VERSION: '',
        DOCS_RELEASE_VERSIONS: '',
      },
    })
    if (result.error) throw new Error(`Cannot run VitePress: ${result.error.message}`)
    if (result.status !== 0) throw new Error(`VitePress failed (${result.signal ? `signal ${result.signal}` : `exit ${result.status}`}).`)
    const index = join(temporary, 'index.html')
    if (!existsSync(index) || !statSync(index).isFile()) throw new Error('VitePress did not produce the dev index.html.')
    mkdirSync(pages, { recursive: true })
    renameSync(temporary, join(pages, 'dev'))
    writeFileSync(join(pages, 'index.html'), redirect)
    const checker = join(api.codeRoot, 'scripts', 'docs', 'check_site.py')
    if (!existsSync(checker)) throw new Error(`Missing site checker: ${checker}`)
    const checked = run(env.PYTHON_FOR_DOCS || 'python3', [checker, root], {
      cwd: root, shell: false, stdio: 'inherit', env,
    })
    if (checked.error || checked.status !== 0) throw new Error('Built-site checks failed; output was discarded.')
    log(`Dev documentation built: ${pages}; code ref ${api.codeRef}${api.preview ? ' (preview)' : ''}`)
    return { directory: pages, version: 'dev', codeRef: api.codeRef, preview: api.preview }
  } catch (error) {
    rmSync(pages, { recursive: true, force: true })
    throw error
  } finally {
    rmSync(temporary, { recursive: true, force: true })
  }
}

if (process.argv[1] && pathToFileURL(resolve(process.argv[1])).href === import.meta.url) {
  try { buildDev() }
  catch (error) {
    console.error(`Dev documentation build failed: ${error.message}`)
    process.exitCode = 1
  }
}
