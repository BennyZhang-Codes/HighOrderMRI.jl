// Every production build includes all configured documentation versions.
import { buildSite } from './scripts/build-site.mjs'
try { buildSite() }
catch (error) {
  console.error(`Documentation build failed: ${error.message}`)
  process.exitCode = 1
}
