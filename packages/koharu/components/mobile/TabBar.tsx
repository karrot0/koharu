'use client'

import { FolderOpen, PenTool, Settings } from 'lucide-react'
import { useTranslation } from 'react-i18next'

export type MobileTab = 'projects' | 'editor' | 'settings'

const tabs = [
  ['projects', FolderOpen, 'start.title'],
  ['editor', PenTool, 'mobile.editor'],
  ['settings', Settings, 'menu.settings'],
] as const

export function TabBar({ tab, onSelect }: { tab: MobileTab; onSelect: (tab: MobileTab) => void }) {
  const { t } = useTranslation()

  return (
    <nav className='shrink-0 border-t border-border/80 bg-[var(--surface-titlebar)] pb-[env(safe-area-inset-bottom)]'>
      <div className='grid h-[50px] grid-cols-3' role='tablist'>
        {tabs.map(([id, Icon, label]) => (
          <button
            key={id}
            type='button'
            role='tab'
            aria-selected={tab === id}
            className='flex flex-col items-center justify-center gap-0.5 text-[10px] font-medium text-muted-foreground outline-none focus-visible:text-foreground aria-selected:text-primary'
            onClick={() => onSelect(id)}
          >
            <Icon className='size-6' strokeWidth={1.75} />
            {t(label)}
          </button>
        ))}
      </div>
    </nav>
  )
}
