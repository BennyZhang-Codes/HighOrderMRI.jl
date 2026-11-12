<script setup lang="ts">
declare const __DOCS_CANDIDATE__: boolean
declare const __DOCS_CODE_LABEL__: string
const candidate = __DOCS_CANDIDATE__
const codeLabel = __DOCS_CODE_LABEL__
</script>

<template>
  <aside v-if="candidate" class="candidate-notice" role="note">
    <strong>Local preview · unpublished</strong>
    <span>Code baseline {{ codeLabel }} with local docstring and website changes. This preview is not the published documentation.</span>
  </aside>
</template>

<style scoped>
.candidate-notice {
  margin: 0 0 24px;
  padding: 12px 16px;
  border: 1px solid var(--vp-c-divider);
  border-left: 3px solid var(--vp-c-brand-1);
  border-radius: 6px;
  background: var(--vp-c-bg-soft);
  color: var(--vp-c-text-2);
  font-size: 13px;
  line-height: 1.6;
}
.candidate-notice strong { display: block; color: var(--vp-c-text-1); }
</style>
