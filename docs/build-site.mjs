import { execFileSync } from 'node:child_process'
import { cpSync, mkdirSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..')
const output = join(root, 'docs', '.pages')
const base = '/HighOrderMRI.jl'
const git = (...args) => execFileSync('git', args, { cwd: root, encoding: 'utf8' }).trim()

// Only advertise versions whose tagged source can actually build VitePress.
const tags = git('tag', '--list', 'v*', '--sort=-v:refname').split('\n').filter(Boolean)
const versions = tags.filter((tag) => {
  let text
  try {
    text = git('show', `${tag}:docs/package.json`)
  } catch {
    console.log(`Skipping ${tag}: no VitePress documentation source`)
    return false
  }
  const pkg = JSON.parse(text)
  return Boolean(pkg.devDependencies?.vitepress && pkg.scripts?.['docs:build'])
}).slice(0, 8)
const stable = versions[0] || ''

function build(source, version, sourceRef) {
  execFileSync('npm', ['run', 'docs:build', '--prefix', 'docs'], {
    cwd: source,
    stdio: 'inherit',
    env: {
      ...process.env,
      DOCS_VERSION: version,
      DOCS_BASE: `${base}/${version}/`,
      DOCS_SOURCE_REF: sourceRef,
      DOCS_STABLE_VERSION: stable,
      DOCS_RELEASE_VERSIONS: versions.join(','),
    },
  })
  cpSync(join(source, 'docs', 'src', '.vitepress', 'dist'), join(output, version), {
    recursive: true,
  })
}

rmSync(output, { recursive: true, force: true })
mkdirSync(output, { recursive: true })
build(root, 'dev', 'docs')

for (const tag of versions) {
  const temporary = mkdtempSync(join(tmpdir(), 'highordermri-docs-'))
  const source = join(temporary, 'source')
  try {
    execFileSync('git', ['worktree', 'add', '--detach', source, tag], {
      cwd: root,
      stdio: 'inherit',
    })
    execFileSync('npm', ['install', '--prefix', 'docs', '--no-audit', '--no-fund'], {
      cwd: source,
      stdio: 'inherit',
    })
    build(source, tag, tag)
    if (tag === stable) build(source, 'stable', tag)
  } finally {
    if (git('worktree', 'list', '--porcelain').includes(`worktree ${source}\n`)) {
      execFileSync('git', ['worktree', 'remove', '--force', source], { cwd: root })
    }
    rmSync(temporary, { recursive: true, force: true })
  }
}

const target = stable ? 'stable' : 'dev'
writeFileSync(join(output, 'index.html'), `<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <meta http-equiv="refresh" content="0; url=./${target}/">
    <link rel="canonical" href="https://bennyzhang-codes.github.io${base}/${target}/">
    <title>HighOrderMRI.jl documentation</title>
  </head>
  <body>
    <p>Redirecting to <a href="./${target}/">HighOrderMRI.jl ${target} documentation</a>.</p>
    <script>location.replace('./${target}/' + location.search + location.hash)</script>
  </body>
</html>
`)
console.log(`Complete Pages site: ${output}; versions: dev${versions.length ? `, stable, ${versions.join(', ')}` : ''}`)
