import type { Theme } from 'vitepress'
import DefaultTheme from 'vitepress/theme'
import MermaidDiagram from '../plugins/vitepress-mermaid/MermaidDiagram.vue'
import './style.css'

export default {
  extends: DefaultTheme,
  enhanceApp({ app }) {
    app.component('VitePressMermaid', MermaidDiagram)
  },
} satisfies Theme
