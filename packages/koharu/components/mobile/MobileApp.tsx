'use client'

import { useEffect, useState } from 'react'
import { useTranslation } from 'react-i18next'

import { EditorTab } from '@/components/mobile/EditorTab'
import { ProjectsTab } from '@/components/mobile/ProjectsTab'
import { SettingsTab } from '@/components/mobile/SettingsTab'
import { TabBar, type MobileTab } from '@/components/mobile/TabBar'
import { useProject } from '@/lib/queries'
import { useKoharuStore } from '@/lib/store'

/**
 * Koharu on iPhone and iPad: Projects, Editor, and Settings tabs instead of the desktop window's
 * menu bar. The Settings tab is the store's `settingsOpen`, so anything that opens settings, such
 * as the run control's settings link, switches to it.
 */
export function MobileApp() {
  const { t } = useTranslation()
  const project = useProject().data
  const settingsOpen = useKoharuStore((state) => state.settingsOpen)
  const setSettingsOpen = useKoharuStore((state) => state.setSettingsOpen)
  const [tab, setTab] = useState<Exclude<MobileTab, 'settings'>>('projects')
  const projectName = project?.name

  // A project that opens, including one restored at launch, is opened to be edited.
  useEffect(() => {
    if (projectName) setTab('editor')
  }, [projectName])

  const select = (next: MobileTab) => {
    setSettingsOpen(next === 'settings')
    if (next !== 'settings') setTab(next)
  }

  return (
    <div className='flex h-dvh w-screen flex-col overflow-hidden bg-[var(--surface-canvas)] pr-[env(safe-area-inset-right)] pl-[env(safe-area-inset-left)] text-foreground'>
      {project === undefined ? (
        <main className='grid flex-1 place-items-center'>
          <div className='flex items-center gap-3 text-[15px] text-muted-foreground'>
            <span className='size-2 rounded-full bg-primary' />
            {t('startup.opening')}
          </div>
        </main>
      ) : settingsOpen ? (
        <SettingsTab />
      ) : tab === 'projects' ? (
        <ProjectsTab onOpened={() => select('editor')} />
      ) : (
        <EditorTab onShowProjects={() => select('projects')} />
      )}
      <TabBar tab={settingsOpen ? 'settings' : tab} onSelect={select} />
    </div>
  )
}
