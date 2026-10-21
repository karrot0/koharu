'use client'

import { openUrl } from '@tauri-apps/plugin-opener'
import { ChevronLeft, ChevronRight, Code, Info, MessageCircle } from 'lucide-react'
import { useState, type ComponentType, type ReactNode } from 'react'
import { useTranslation } from 'react-i18next'

import { AboutDialog } from '@/components/app/AboutDialog'
import { NavigationBar } from '@/components/mobile/NavigationBar'
import {
  SettingsContent,
  settingsTabs,
  useSettingsDraft,
  type SettingsTab as Section,
} from '@/components/preferences/SettingsPage'
import { Button } from '@koharu/ui/components/button'

// Keyboard shortcuts have nothing to bind to on a touch screen.
const sections = settingsTabs.filter(([id]) => id !== 'shortcuts')

/** The Settings tab: an inset list of sections, each pushed as its own screen. */
export function SettingsTab() {
  const { t } = useTranslation()
  const draft = useSettingsDraft()
  const [section, setSection] = useState<Section | null>(null)
  const [aboutOpen, setAboutOpen] = useState(false)

  if (section) {
    return (
      <section className='settings-page flex min-h-0 flex-1 flex-col'>
        <NavigationBar
          leading={
            <Button
              type='button'
              variant='ghost'
              className='h-9 gap-0.5 px-1 text-[16px] text-primary'
              onClick={() => setSection(null)}
            >
              <ChevronLeft className='size-6' /> {t('settings.title')}
            </Button>
          }
        />
        <div className='min-h-0 flex-1 overflow-y-auto px-4 pt-2 pb-8'>
          <SettingsContent tab={section} draft={draft} />
        </div>
      </section>
    )
  }

  const link = (url: string) => void openUrl(url).catch(() => undefined)

  return (
    <section className='flex min-h-0 flex-1 flex-col'>
      <NavigationBar largeTitle={t('settings.title')} />
      <div className='min-h-0 flex-1 overflow-y-auto px-4 pb-8'>
        <InsetList>
          {sections.map(([id, Icon]) => (
            <Row key={id} icon={Icon} onClick={() => setSection(id)}>
              {t(`settings.tabs.${id}`)}
            </Row>
          ))}
        </InsetList>
        <InsetList>
          <Row icon={MessageCircle} onClick={() => link('https://discord.gg/mHvHkxGnUY')}>
            {t('menu.discord')}
          </Row>
          <Row icon={Code} onClick={() => link('https://github.com/koharu-rs/koharu')}>
            {t('menu.github')}
          </Row>
          <Row icon={Info} onClick={() => setAboutOpen(true)}>
            {t('menu.about')}
          </Row>
        </InsetList>
      </div>
      <AboutDialog open={aboutOpen} onOpenChange={setAboutOpen} />
    </section>
  )
}

function InsetList({ children }: { children: ReactNode }) {
  return (
    <ul className='mt-5 divide-y divide-border/70 overflow-hidden rounded-xl bg-[var(--surface-panel)]'>
      {children}
    </ul>
  )
}

function Row({
  icon: Icon,
  onClick,
  children,
}: {
  icon: ComponentType<{ className?: string }>
  onClick: () => void
  children: ReactNode
}) {
  return (
    <li>
      <button
        type='button'
        className='flex min-h-12 w-full items-center gap-3 px-4 text-left text-[16px] outline-none active:bg-foreground/[0.06]'
        onClick={onClick}
      >
        <span className='grid size-7 shrink-0 place-items-center rounded-md bg-primary text-primary-foreground'>
          <Icon className='size-4' />
        </span>
        <span className='min-w-0 flex-1 truncate'>{children}</span>
        <ChevronRight className='size-5 shrink-0 text-muted-foreground/70' />
      </button>
    </li>
  )
}
