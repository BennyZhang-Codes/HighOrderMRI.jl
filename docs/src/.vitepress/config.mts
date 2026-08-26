import { defineConfig } from 'vitepress'
import { mermaidPlugin } from './plugins/vitepress-mermaid'

export default defineConfig({
  title: 'HighOrderMRI.jl',
  description: 'GPU-accelerated Cartesian and non-Cartesian MRI reconstruction with dynamic high-order field encoding.',
  base: '/HighOrderMRI.jl/',
  cleanUrls: true,
  lastUpdated: true,
  head: [
    ['meta', { name: 'theme-color', content: '#6366f1' }],
  ],
  markdown: {
    math: true,
    config(md) {
      md.use(mermaidPlugin)
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
          { text: 'Low-rank shared subspace', link: '/theory/low-rank' },
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
          { text: 'Reconstruction protocol', link: '/guide/reconstruction-protocol' },
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
        text: 'v0.1.0-dev',
        items: [
          { text: 'v0.1.0-dev · current', link: '/' },
          { text: 'Release history', link: 'https://github.com/BennyZhang-Codes/HighOrderMRI.jl/releases' },
          { text: 'Source: main', link: 'https://github.com/BennyZhang-Codes/HighOrderMRI.jl/tree/main' },
        ],
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
          { text: 'Low-rank shared subspace', link: '/theory/low-rank' },
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
          { text: 'Reconstruction protocol', link: '/guide/reconstruction-protocol' },
          { text: 'Performance & benchmarking', link: '/guide/performance' },
        ],
      },
      {
        text: 'Reference',
        items: [
          { text: 'API overview', link: '/api' },
          { text: 'Grid & field basis', link: '/reference/grid-basis' },
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
    editLink: {
      pattern: 'https://github.com/BennyZhang-Codes/HighOrderMRI.jl/edit/docs-modern-ui/docs/src/:path',
      text: 'Edit this page on GitHub',
    },
    footer: {
      message: 'Research software for field-aware Cartesian and non-Cartesian MRI reconstruction.',
      copyright: 'HighOrderMRI.jl contributors',
    },
  },
})
