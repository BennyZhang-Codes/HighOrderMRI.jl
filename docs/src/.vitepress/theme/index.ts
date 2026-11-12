import type { Theme } from 'vitepress'
import DefaultTheme from 'vitepress/theme'
import { h } from 'vue'
import CandidateNotice from './CandidateNotice.vue'
import MermaidDiagram from '../plugins/vitepress-mermaid/MermaidDiagram.vue'
import './style.css'

export default {
  extends: DefaultTheme,
  Layout: () => h(DefaultTheme.Layout, null, { 'doc-before': () => h(CandidateNotice) }),
  enhanceApp({ app }) {
    app.component('VitePressMermaid', MermaidDiagram)
  },
} satisfies Theme
