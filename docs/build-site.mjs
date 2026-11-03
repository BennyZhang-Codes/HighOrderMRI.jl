// Compatibility entry: every production build uses the checked dev pipeline.
import { buildDev } from './scripts/build-dev.mjs'
try { buildDev() }
catch (error) {
  console.error(`Documentation build failed: ${error.message}`)
  process.exitCode = 1
}
