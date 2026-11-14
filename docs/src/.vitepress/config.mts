import { defineConfig } from 'vitepress'
import { existsSync, readFileSync } from 'node:fs'
import { mermaidPlugin } from './plugins/vitepress-mermaid/index'

const siteUrl = 'https://bennyzhang-codes.github.io/HighOrderMRI.jl'
const docsVersion = process.env.DOCS_VERSION || 'dev'
const docsBase = process.env.DOCS_BASE || '/HighOrderMRI.jl/dev/'
const editRef = docsVersion === 'dev' ? 'dev_jinyuan' : 'main'
const jointPages = existsSync(new URL('../theory/joint-adaptive-low-rank.md', import.meta.url))
const lowRankItems = [
  { text: 'Shared low-rank', link: '/theory/low-rank' },
  ...(jointPages ? [{ text: 'Joint adaptive low-rank', link: '/theory/joint-adaptive-low-rank' }] : []),
]
const apiMeta = JSON.parse(readFileSync(new URL('../../.generated/api/build-meta.json', import.meta.url), 'utf8'))
const candidatePreview = apiMeta.preview === true
const codeLabel = apiMeta.codeRef.slice(0, 7)
const versionItems = JSON.parse(process.env.DOCS_VERSION_ITEMS || JSON.stringify([{ text: 'dev', link: `${siteUrl}/dev/` }])).map(item => ({ ...item, target: '_self' }))
const versionLabel = docsVersion === 'stable' ? `stable · ${process.env.DOCS_RELEASE_VERSION}` : docsVersion
const frozenDocs = process.env.DOCS_DOCS_SHA || ''

export default defineConfig({
  title: 'HighOrderMRI.jl',
  description: 'GPU-accelerated Cartesian and non-Cartesian MRI reconstruction with dynamic high-order field encoding.',
  base: docsBase,
  cleanUrls: true,
  outDir: '../.pages/dev',
  vite: { define: { __DOCS_CANDIDATE__: JSON.stringify(candidatePreview), __DOCS_CODE_LABEL__: JSON.stringify(codeLabel) } },
  lastUpdated: !candidatePreview,
  head: [
    ['meta', { name: 'theme-color', content: '#6366f1' }],
    ['link', { rel: 'icon', href: 'data:,' }],
  ],
  markdown: {
    math: true,
    config(md) {
      md.use(mermaidPlugin)
      const originalLink = md.renderer.rules.link_open
      md.renderer.rules.link_open = (tokens, index, options, env, self) => {
        const token = tokens[index]
        const href = token.attrGet('href') || ''
        const match = href.match(/^(https:\/\/github\.com\/BennyZhang-Codes\/HighOrderMRI\.jl\/(?:blob|tree)\/)(main|dev_jinyuan|docs|[0-9a-f]{40})(\/(?:src|MRIGeometry|scripts)\/.*)$/)
        if (match) {
          let suffix = match[3]
          if (candidatePreview || match[2] !== apiMeta.codeRef) suffix = suffix.replace(/#L\d+(?:-L\d+)?$/, '')
          token.attrSet('href', match[1] + apiMeta.codeRef + suffix)
        }
        return originalLink ? originalLink(tokens, index, options, env, self) : self.renderToken(tokens, index, options)
      }
    },
  },
  themeConfig: {
    siteTitle: 'HighOrderMRI.jl',
    nav: [
      {
        text: 'Guide',
        items: [
          { text: 'Getting started', link: '/getting-started' },
          { text: 'Architecture', link: '/concepts-overview' },
        ],
      },
      {
        text: 'Theory',
        items: [
          { text: 'Symbols & notation', link: '/theory/symbols' },
          { text: 'Expanded encoding model', link: '/theory/encoding-model' },
          ...lowRankItems,
        ],
      },
      {
        text: 'Reconstruction',
        items: [
          { text: 'Encoding operators', link: '/guide/operators' },
          { text: 'Field preprocessing', link: '/guide/field-preprocessing' },
          { text: 'Coil compression', link: '/guide/coil-compression' },
          { text: 'Reconstruction workflow', link: '/guide/reconstruction' },
          { text: 'Multi-GPU execution', link: '/guide/multi-gpu' },
        ],
      },
      {
        text: 'Validation',
        items: [
          { text: 'Scientific validation', link: '/guide/validation' },
          { text: 'Reconstruction conventions', link: '/guide/reconstruction-protocol' },
          { text: 'Performance & benchmarking', link: '/guide/performance' },
        ],
      },
      {
        text: 'Reference',
        items: [
          { text: 'API overview', link: '/api' },
          { text: 'Encoding operators', link: '/reference/highorderop' },
          { text: 'Reconstruction', link: '/reference/recon-hoop' },
          { text: 'Coil compression', link: '/reference/coil-compression-transform' },
          { text: 'Field & synchronization', link: '/reference/girf-model' },
          { text: 'Metrics & utilities', link: '/reference/image-metrics' },
          { text: 'Literature references', link: '/references' },
        ],
      },
      {
        text: 'Support',
        items: [
          { text: 'Troubleshooting', link: '/guide/troubleshooting' },
          { text: 'GitHub repository', link: 'https://github.com/BennyZhang-Codes/HighOrderMRI.jl' },
        ],
      },
      {
        text: versionLabel,
        items: versionItems,
      },
    ],
    sidebar: [
      {
        text: 'Getting Started',
        items: [
          { text: 'Overview', link: '/getting-started' },
          { text: 'Architecture', link: '/concepts-overview' },
        ],
      },
      {
        text: 'Theory',
        items: [
          { text: 'Symbols & notation', link: '/theory/symbols' },
          { text: 'Expanded encoding model', link: '/theory/encoding-model' },
          ...lowRankItems,
        ],
      },
      {
        text: 'Reconstruction',
        items: [
          { text: 'Encoding operators', link: '/guide/operators' },
          { text: 'Field preprocessing', link: '/guide/field-preprocessing' },
          { text: 'Coil compression', link: '/guide/coil-compression' },
          { text: 'Reconstruction workflow', link: '/guide/reconstruction' },
          { text: 'Multi-GPU execution', link: '/guide/multi-gpu' },
        ],
      },
      {
        text: 'Validation & Performance',
        items: [
          { text: 'Scientific validation', link: '/guide/validation' },
          { text: 'Reconstruction conventions', link: '/guide/reconstruction-protocol' },
          { text: 'Performance & benchmarking', link: '/guide/performance' },
        ],
      },
      {
        text: 'Reference',
        items: [
          { text: 'API overview', link: '/api' },
          { text: 'Grid & field basis', link: '/reference/grid-basis' },
          { text: 'Geometry & spatial I/O', link: '/reference/geometry' },
          {
            text: 'Encoding operators',
            collapsed: true,
            items: [
              { text: 'HighOrderOp', link: '/reference/highorderop' },
              { text: 'HighOrderKernelOp', link: '/reference/highorderkernelop' },
              { text: 'HighOrderLowRankOp', link: '/reference/highorderlowrankop' },
            ],
          },
          {
            text: 'Reconstruction',
            collapsed: true,
            items: [
              { text: 'recon_HOOp', link: '/reference/recon-hoop' },
              { text: 'samplingDensity', link: '/reference/sampling-density' },
              { text: 'CoilCombineSOS', link: '/reference/coil-combine-sos' },
            ],
          },
          {
            text: 'Coil compression',
            collapsed: true,
            items: [
              { text: 'CoilCompressionTransform', link: '/reference/coil-compression-transform' },
              { text: 'estimate_noise_covariance', link: '/reference/estimate-noise-covariance' },
              { text: 'noise_prewhitening_scale_factor', link: '/reference/noise-prewhitening-scale' },
              { text: 'fit_coil_compression', link: '/reference/fit-coil-compression' },
              { text: 'apply_coil_compression', link: '/reference/apply-coil-compression' },
              { text: 'compress_coils', link: '/reference/compress-coils' },
            ],
          },
          {
            text: 'Field & synchronization',
            collapsed: true,
            items: [
              { text: 'GIRFModel', link: '/reference/girf-model' },
              { text: 'apply_girf', link: '/reference/apply-girf' },
              { text: 'InterpTrajTime', link: '/reference/interp-traj-time' },
              { text: 'FindDelay', link: '/reference/find-delay' },
              { text: 'FindDelay_multishot', link: '/reference/find-delay-multishot' },
            ],
          },
          {
            text: 'Metrics & utilities',
            collapsed: true,
            items: [
              { text: 'Reconstruction metrics', link: '/reference/image-metrics' },
              { text: 'Array & signal utilities', link: '/reference/utilities' },
              { text: 'Plotting helpers', link: '/reference/plotting' },
              { text: 'Resource lifecycle', link: '/reference/resources' },
            ],
          },
          { text: 'Literature references', link: '/references' },
        ],
      },
      {
        text: 'Support',
        items: [
          { text: 'Troubleshooting', link: '/guide/troubleshooting' },
        ],
      },
    ],
    socialLinks: [
      { icon: 'github', link: 'https://github.com/BennyZhang-Codes/HighOrderMRI.jl' },
    ],
    search: {
      provider: 'local',
    },
    outline: {
      level: [2, 3],
      label: 'On this page',
    },
    editLink: ['dev', 'main'].includes(docsVersion) ? {
      pattern: `https://github.com/BennyZhang-Codes/HighOrderMRI.jl/edit/${editRef}/docs/src/:path`,
      text: 'Edit this page on GitHub',
    } : undefined,
    footer: {
      message: `HighOrderMRI.jl documentation · ${versionLabel} · code ${codeLabel}${frozenDocs ? ` · docs ${frozenDocs.slice(0, 7)}` : ""}${candidatePreview ? " + local candidate changes" : ""}`,
      copyright: 'HighOrderMRI.jl contributors',
    },
  },
})
