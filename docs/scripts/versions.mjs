import { readFileSync } from 'node:fs'
import { join, resolve } from 'node:path'
import { pathToFileURL } from 'node:url'

const shaPattern = /^[0-9a-f]{40}$/
const versionPattern = /^v(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$/

export function versionNumber(version) {
  const match = versionPattern.exec(version)
  if (!match || match.slice(1).some(part => !Number.isSafeInteger(Number(part)))) throw new Error(`Invalid release version: ${version}`)
  return match.slice(1).map(Number)
}

export function compareVersions(left, right) {
  const a = versionNumber(left)
  const b = versionNumber(right)
  for (let i = 0; i < 3; i += 1) if (a[i] !== b[i]) return a[i] - b[i]
  return 0
}

export function loadVersions(docsRoot, planFile = null) {
  if (planFile) {
    const plan = JSON.parse(readFileSync(planFile, 'utf8'))
    if (plan.schema_version !== 1 || !shaPattern.test(plan.source_sha) || !Array.isArray(plan.releases) || !Array.isArray(plan.targets)) throw new Error('Invalid publication plan.')
    const names = new Set()
    for (const target of plan.targets) {
      if (names.has(target.version) || !shaPattern.test(target.code_sha) || !shaPattern.test(target.docs_sha)) throw new Error('Invalid or duplicate publication target.')
      if (!['dev', 'stable'].includes(target.version)) versionNumber(target.version)
      names.add(target.version)
    }
    const releases = [...plan.releases].sort((a, b) => compareVersions(b.version, a.version))
    if (!releases.length || releases[0].version !== plan.stable || names.size !== releases.length + 2 || !['dev', 'stable', ...releases.map(r => r.version)].every(v => names.has(v))) throw new Error('Incomplete publication version set.')
    for (const release of releases) {
      for (const target of plan.targets.filter(t => t.version === release.version || t.version === 'stable' && release.version === plan.stable)) {
        if (target.code_sha !== release.code_sha || target.docs_sha !== release.docs_sha) throw new Error('Release source identity mismatch.')
      }
    }
    return plan
  }
  const code = JSON.parse(readFileSync(join(docsRoot, 'code-version.json'), 'utf8'))
  const config = JSON.parse(readFileSync(join(docsRoot, 'versions.json'), 'utf8'))
  if (code.schema_version !== 1 || !shaPattern.test(code.code_sha)) throw new Error('Invalid dev code_sha.')
  if (code.main_sha !== undefined) throw new Error('The main documentation channel is not published; remove main_sha.')
  if (config.schema_version !== 1 || !Array.isArray(config.releases)) throw new Error('Invalid versions.json.')
  const seen = new Set()
  const releases = config.releases.map(release => {
    versionNumber(release.version)
    if (seen.has(release.version)) throw new Error(`Duplicate release: ${release.version}`)
    seen.add(release.version)
    if (!shaPattern.test(release.code_sha) || !shaPattern.test(release.docs_sha)) throw new Error(`Release ${release.version} requires full code_sha and docs_sha.`)
    return { version: release.version, code_sha: release.code_sha, docs_sha: release.docs_sha }
  }).sort((a, b) => compareVersions(b.version, a.version))
  const stable = releases.find(release => release.version === config.stable)
  if (!stable || releases[0].version !== stable.version) throw new Error('Stable must select the newest configured formal release.')
  const targets = [
    { version: 'dev', code_sha: code.code_sha, docs_sha: null },
    ...releases, { ...stable, version: 'stable', release: stable.version },
  ]
  return { schema_version: 1, stable: stable.version, releases, targets }
}

if (process.argv[1] && pathToFileURL(resolve(process.argv[1])).href === import.meta.url) {
  try {
    const plan = loadVersions(resolve(process.argv[2] || 'docs'), process.env.DOCS_BUILD_PLAN)
    if (process.argv[3] === '--code-shas') console.log([...new Set(plan.targets.map(target => target.code_sha))].join('\n'))
    else console.log(JSON.stringify(plan, null, 2))
  } catch (error) { console.error(error.message); process.exitCode = 1 }
}
