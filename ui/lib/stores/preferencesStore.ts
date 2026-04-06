'use client'

import { create } from 'zustand'
import { persist } from 'zustand/middleware'

export type GlossaryEntry = {
  id: string
  source: string
  target: string
}

export function buildSystemPromptWithGlossary(
  customSystemPrompt: string | undefined,
  glossary: GlossaryEntry[],
): string | undefined {
  const activeGlossary = glossary.filter(
    (e) => e.source.trim() && e.target.trim(),
  )
  if (activeGlossary.length === 0) return customSystemPrompt || undefined

  const glossarySection =
    'Use the following terminology glossary for consistent translations:\n' +
    activeGlossary.map((e) => `- ${e.source} → ${e.target}`).join('\n')

  if (!customSystemPrompt?.trim()) return glossarySection

  return `${customSystemPrompt.trim()}\n\n${glossarySection}`
}

type PreferencesState = {
  brushConfig: {
    size: number
    color: string
  }
  setBrushConfig: (config: Partial<PreferencesState['brushConfig']>) => void
  defaultFont?: string
  setDefaultFont: (font?: string) => void
  customSystemPrompt?: string
  setCustomSystemPrompt: (prompt?: string) => void
  glossary: GlossaryEntry[]
  addGlossaryEntry: (source: string, target: string) => void
  updateGlossaryEntry: (id: string, source: string, target: string) => void
  removeGlossaryEntry: (id: string) => void
  resetPreferences: () => void
}

const initialPreferences = {
  brushConfig: {
    size: 36,
    color: '#ffffff',
  },
  glossary: [] as GlossaryEntry[],
}

export const usePreferencesStore = create<PreferencesState>()(
  persist(
    (set) => ({
      ...initialPreferences,
      setBrushConfig: (config) =>
        set((state) => ({
          brushConfig: {
            ...state.brushConfig,
            ...config,
          },
        })),
      setDefaultFont: (font) => set({ defaultFont: font }),
      setCustomSystemPrompt: (prompt) => set({ customSystemPrompt: prompt }),
      addGlossaryEntry: (source, target) =>
        set((state) => ({
          glossary: [
            ...state.glossary,
            { id: crypto.randomUUID(), source, target },
          ],
        })),
      updateGlossaryEntry: (id, source, target) =>
        set((state) => ({
          glossary: state.glossary.map((entry) =>
            entry.id === id ? { ...entry, source, target } : entry,
          ),
        })),
      removeGlossaryEntry: (id) =>
        set((state) => ({
          glossary: state.glossary.filter((entry) => entry.id !== id),
        })),
      resetPreferences: () => set({ ...initialPreferences }),
    }),
    {
      name: 'koharu-config',
      version: 3,
      migrate: (persisted: any, version: number) => {
        if (version < 2 && persisted) {
          delete persisted.localLlm
          delete persisted.openAiCompatibleConfigVersion
        }
        if (version < 3 && persisted) {
          delete persisted.apiKeys
          delete persisted.providerBaseUrls
          delete persisted.providerModelNames
        }
        return persisted
      },
      partialize: (state) => ({
        brushConfig: state.brushConfig,
        defaultFont: state.defaultFont,
        customSystemPrompt: state.customSystemPrompt,
        glossary: state.glossary,
      }),
    },
  ),
)
