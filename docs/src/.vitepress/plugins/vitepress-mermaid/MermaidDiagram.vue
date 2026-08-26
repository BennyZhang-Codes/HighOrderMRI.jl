<template>
  <figure class="homri-mermaid">
    <div ref="diagramEl" class="homri-mermaid__canvas" />
    <pre v-if="error" class="homri-mermaid__error"><code>{{ source }}</code></pre>
  </figure>
</template>

<script lang="ts">
let globalMermaidRenderSerial = 0
</script>

<script setup lang="ts">
import { computed, nextTick, onMounted, ref, watch } from 'vue'
import { useData } from 'vitepress'
import mermaid from 'mermaid'

const props = defineProps<{
  code: string
}>()

const { isDark } = useData()
const diagramEl = ref<HTMLDivElement | null>(null)
const error = ref('')
const source = computed(() => decodeURIComponent(props.code))

// Mermaid HTML labels are intentionally kept free of mathematical markup.
// Exact mathematical notation is rendered with MathJax in the surrounding
// documentation, while overview diagrams use stable descriptive labels.
const stableLabelReplacements: ReadonlyArray<readonly [string, string]> = [
  ['Residual encoding H_d', 'Residual encoding matrix'],
  ['Temporal factor U_d', 'Temporal factor'],
  ['Spatial factor Ṽ_d', 'Retained spatial factor'],
  ['Shared basis S_d + coefficients C̄_d', 'Shared basis + coefficients'],
  ['3. Sample-domain factor q̂_d', '3. Sample-domain coefficients'],
  ['Residual model H_d ≈ q̂_d Sᴴ', 'Low-rank residual representation'],
]

function stabilizeQuotedLabels(code: string) {
  return code.replace(/"([^"\n]*)"/g, (quoted, label: string) => {
    let stable = label
    for (const [token, replacement] of stableLabelReplacements) {
      if (stable === token) {
        stable = replacement
        break
      }
    }
    return `"${stable}"`
  })
}

async function renderDiagram() {
  if (!diagramEl.value) return

  await nextTick()
  error.value = ''
  diagramEl.value.innerHTML = ''

  const dark = isDark.value

  mermaid.initialize({
    startOnLoad: false,
    securityLevel: 'strict',
    theme: 'base',
    themeVariables: dark
      ? {
          background: 'transparent',
          primaryColor: '#24243c',
          primaryTextColor: '#eef2ff',
          primaryBorderColor: 'transparent',
          lineColor: '#a5b4fc',
          secondaryColor: '#2b2342',
          tertiaryColor: '#17313a',
          clusterBkg: 'transparent',
          clusterBorder: 'transparent',
          edgeLabelBackground: 'transparent',
        }
      : {
          background: 'transparent',
          primaryColor: '#f7f7ff',
          primaryTextColor: '#28233e',
          primaryBorderColor: 'transparent',
          lineColor: '#7c3aed',
          secondaryColor: '#f5f3ff',
          tertiaryColor: '#ecfeff',
          clusterBkg: 'transparent',
          clusterBorder: 'transparent',
          edgeLabelBackground: 'transparent',
        },
    themeCSS: `
      .node rect,
      .node polygon,
      .node path,
      .cluster rect {
        stroke: none !important;
        stroke-width: 0 !important;
      }
      .node rect {
        rx: 11px;
        ry: 11px;
      }
      .nodeLabel,
      .edgeLabel {
        font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      }
      .nodeLabel {
        font-size: 14px;
        line-height: 1.28;
      }
      .nodeLabel p {
        margin: 0 !important;
        line-height: 1.28 !important;
      }
      .edgeLabel {
        font-size: 13px;
      }
      .edgePath path {
        stroke-width: 1.35px;
      }
      .flowchart-link {
        stroke-linecap: round;
        stroke-linejoin: round;
      }
    `,
    flowchart: {
      curve: 'basis',
      htmlLabels: true,
      nodeSpacing: 40,
      rankSpacing: 46,
      padding: 14,
      wrappingWidth: 190,
    },
  })

  const id = `homri-mermaid-${++globalMermaidRenderSerial}`
  const renderedSource = stabilizeQuotedLabels(source.value)

  try {
    const { svg, bindFunctions } = await mermaid.render(id, renderedSource)
    if (!diagramEl.value) return
    diagramEl.value.innerHTML = svg
    bindFunctions?.(diagramEl.value)
  } catch (cause) {
    error.value = cause instanceof Error ? cause.message : String(cause)
  }
}

onMounted(renderDiagram)
watch([source, isDark], renderDiagram)
</script>

<style scoped>
.homri-mermaid {
  margin: 1.5rem 0 2rem;
  padding: .6rem 0;
  overflow-x: auto;
  border: 0;
  background: transparent;
}

.homri-mermaid__canvas {
  display: flex;
  width: max-content;
  min-width: 100%;
  justify-content: center;
  align-items: center;
  background: transparent;
}

.homri-mermaid__canvas :deep(svg) {
  display: block;
  width: auto !important;
  max-width: none !important;
  height: auto;
  overflow: visible;
  background: transparent !important;
}

.homri-mermaid__canvas :deep(.nodeLabel),
.homri-mermaid__canvas :deep(.nodeLabel p) {
  overflow: visible;
}

.homri-mermaid__error {
  margin: 0;
  white-space: pre-wrap;
  color: var(--vp-c-danger-1);
  background: transparent;
}

@media (max-width: 700px) {
  .homri-mermaid {
    padding: .45rem 0;
  }

  .homri-mermaid__canvas {
    justify-content: flex-start;
  }
}
</style>