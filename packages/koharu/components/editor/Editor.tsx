'use client'

import { Layers, SlidersHorizontal } from 'lucide-react'
import { useState } from 'react'
import { useTranslation } from 'react-i18next'

import { ColorSamplingProvider } from '@/components/controls/ColorSampling'
import { ActivityCenter } from '@/components/editor/ActivityCenter'
import { CanvasWorkspace } from '@/components/editor/CanvasWorkspace'
import { PageRail } from '@/components/editor/PageRail'
import { RightSidebar } from '@/components/editor/RightSidebar'
import { useCompactLayout } from '@/lib/layout'
import { Button } from '@koharu/ui/components/button'
import {
  ResizableHandle,
  ResizablePanel,
  ResizablePanelGroup,
} from '@koharu/ui/components/resizable'
import { Sheet, SheetContent, SheetTitle } from '@koharu/ui/components/sheet'

export function Editor() {
  const compact = useCompactLayout()
  return (
    <ColorSamplingProvider>
      <div className='relative min-h-0 flex-1 bg-transparent'>
        <ActivityCenter />
        {compact ? <CompactEditor /> : <PanelEditor />}
      </div>
    </ColorSamplingProvider>
  )
}

function PanelEditor() {
  return (
    <ResizablePanelGroup id='editor' orientation='horizontal' className='h-full min-h-0'>
      <ResizablePanel
        id='pages'
        defaultSize='18%'
        minSize='14%'
        maxSize='22%'
        collapsible
        collapsedSize='4%'
        className='min-h-0 overflow-hidden'
      >
        <PageRail />
      </ResizablePanel>
      <ResizableHandle className='w-0 bg-transparent' />
      <ResizablePanel
        id='canvas'
        defaultSize='58%'
        minSize='50%'
        className='workspace-corner-mask relative z-10 min-h-0 rounded-tl-2xl bg-transparent shadow-[var(--shadow-content)]'
      >
        <CanvasWorkspace />
      </ResizablePanel>
      <ResizableHandle className='w-0 bg-transparent' />
      <ResizablePanel
        id='inspector'
        defaultSize='24%'
        minSize='20%'
        maxSize='27%'
        collapsible
        collapsedSize='4%'
        className='min-h-0 overflow-hidden border-l border-border/40 bg-[var(--surface-panel)]'
      >
        <RightSidebar />
      </ResizablePanel>
    </ResizablePanelGroup>
  )
}

/** Phones and portrait tablets: the canvas fills the screen and the panels slide in. */
function CompactEditor() {
  const { t } = useTranslation()
  const [sheet, setSheet] = useState<'pages' | 'inspector' | null>(null)
  const close = (open: boolean) => {
    if (!open) setSheet(null)
  }

  return (
    <div className='flex h-full min-h-0 flex-col'>
      <div className='relative min-h-0 flex-1'>
        <CanvasWorkspace />
      </div>
      <nav className='flex h-12 shrink-0 items-center gap-2 border-t border-border/80 bg-[var(--surface-toolbar)] px-3'>
        <Button
          variant='ghost'
          className='h-10 flex-1 gap-2 text-[12px]'
          aria-pressed={sheet === 'pages'}
          onClick={() => setSheet('pages')}
        >
          <Layers className='size-4' /> {t('navigator.pages')}
        </Button>
        <Button
          variant='ghost'
          className='h-10 flex-1 gap-2 text-[12px]'
          aria-pressed={sheet === 'inspector'}
          onClick={() => setSheet('inspector')}
        >
          <SlidersHorizontal className='size-4' /> {t('agent.properties')}
        </Button>
      </nav>
      <Sheet open={sheet === 'pages'} onOpenChange={close}>
        <SheetContent
          side='left'
          showCloseButton={false}
          className='w-[85vw] max-w-sm gap-0 p-0 pt-[env(safe-area-inset-top)] pb-[env(safe-area-inset-bottom)] pl-[env(safe-area-inset-left)]'
        >
          <SheetTitle className='sr-only'>{t('navigator.pages')}</SheetTitle>
          <PageRail />
        </SheetContent>
      </Sheet>
      <Sheet open={sheet === 'inspector'} onOpenChange={close}>
        <SheetContent
          side='right'
          showCloseButton={false}
          className='w-[90vw] max-w-md gap-0 bg-[var(--surface-panel)] p-0 pt-[env(safe-area-inset-top)] pr-[env(safe-area-inset-right)] pb-[env(safe-area-inset-bottom)]'
        >
          <SheetTitle className='sr-only'>{t('agent.properties')}</SheetTitle>
          <RightSidebar />
        </SheetContent>
      </Sheet>
    </div>
  )
}
