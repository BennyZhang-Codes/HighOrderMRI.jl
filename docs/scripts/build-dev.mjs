// Compatibility entry. Always retain every configured release in the artifact.
import { buildSite } from './build-site.mjs'
export { buildSite as buildDev } from './build-site.mjs'
import { resolve } from 'node:path'
import { pathToFileURL } from 'node:url'
if (process.argv[1] && pathToFileURL(resolve(process.argv[1])).href === import.meta.url) {
  try { buildSite() }
  catch (error) { console.error(error.message); process.exitCode = 1 }
}
