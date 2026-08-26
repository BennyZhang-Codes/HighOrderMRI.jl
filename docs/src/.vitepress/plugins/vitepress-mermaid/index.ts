import type MarkdownIt from 'markdown-it'

export function mermaidPlugin(md: MarkdownIt) {
  const defaultFence = md.renderer.rules.fence

  md.renderer.rules.fence = (tokens, idx, options, env, self) => {
    const token = tokens[idx]
    const language = token.info.trim().split(/\s+/)[0]

    if (language === 'mermaid') {
      const encoded = encodeURIComponent(token.content.trim())
      return `<VitePressMermaid code="${encoded}" />\n`
    }

    if (defaultFence) {
      return defaultFence(tokens, idx, options, env, self)
    }

    return self.renderToken(tokens, idx, options)
  }
}
