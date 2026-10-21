'use client'

import {
  ChevronDown,
  Ellipsis,
  FilePlus2,
  Maximize,
  PenTool,
  Redo2,
  Share,
  SlidersHorizontal,
  Undo2,
  X,
} from 'lucide-react'
import { useEffect, useState } from 'react'
import { useTranslation } from 'react-i18next'

import { ColorSamplingProvider } from '@/components/controls/ColorSampling'
import { ActivityCenter } from '@/components/editor/ActivityCenter'
import { CanvasCommandBar } from '@/components/editor/CanvasCommandBar'
import { CanvasWorkspace } from '@/components/editor/CanvasWorkspace'
import { Editor } from '@/components/editor/Editor'
import { PageRail } from '@/components/editor/PageRail'
import { RightSidebar } from '@/components/editor/RightSidebar'
import { ToolBar } from '@/components/editor/ToolBar'
import { NavigationBar } from '@/components/mobile/NavigationBar'
import { call } from '@/lib/backend'
import { useCompactLayout } from '@/lib/layout'
import {
  pageKey,
  pagesKey,
  projectKey,
  refresh,
  useCommand,
  useImportPages,
  usePage,
  usePages,
  useProject,
} from '@/lib/queries'
import { useKoharuStore } from '@/lib/store'
import { commands } from '@koharu/bridge/protocol'
import { Button } from '@koharu/ui/components/button'
import { Drawer, DrawerContent, DrawerTitle } from '@koharu/ui/components/drawer'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuSub,
  DropdownMenuSubContent,
  DropdownMenuSubTrigger,
  DropdownMenuTrigger,
} from '@koharu/ui/components/dropdown-menu'

/**
 * The Editor tab. Phones and iPads in portrait get the canvas full width with pages and
 * properties in sheets; an iPad in landscape keeps the three-panel split view.
 */
export function EditorTab({ onShowProjects }: { onShowProjects: () => void }) {
  const { t } = useTranslation()
  const project = useProject().data
  const compact = useCompactLayout()
  const [pagesOpen, setPagesOpen] = useState(false)

  if (!project) {
    return (
      <section className='flex min-h-0 flex-1 flex-col'>
        <NavigationBar largeTitle={t('mobile.editor')} />
        <div className='grid flex-1 place-items-center px-8 text-center' role='status'>
          <div>
            <span className='mx-auto grid size-14 place-items-center rounded-2xl bg-muted text-muted-foreground'>
              <PenTool className='size-7' />
            </span>
            <p className='mt-4 text-[17px] font-semibold'>{t('mobile.noProject')}</p>
            <p className='mt-1 text-[14px] text-muted-foreground'>
              {t('mobile.noProjectDescription')}
            </p>
            <Button type='button' className='mt-5 h-11 px-5 text-[15px]' onClick={onShowProjects}>
              {t('start.title')}
            </Button>
          </div>
        </div>
      </section>
    )
  }

  return (
    <section className='flex min-h-0 flex-1 flex-col'>
      <EditorNavigation compact={compact} onShowPages={() => setPagesOpen(true)} />
      {compact ? (
        <PhoneEditor pagesOpen={pagesOpen} onPagesOpenChange={setPagesOpen} />
      ) : (
        <Editor />
      )}
    </section>
  )
}

function EditorNavigation({ compact, onShowPages }: { compact: boolean; onShowPages: () => void }) {
  const { t } = useTranslation()
  const project = useProject().data
  const page = usePage().data
  const pages = usePages().data ?? []
  const requestCanvasFit = useKoharuStore((state) => state.requestCanvasFit)
  const { importPages, importing } = useImportPages()
  const { run: exportProject, busy: exporting } = useCommand(
    ['export-project'],
    commands.export,
    'menu.export',
  )

  const history = (command: typeof commands.undo) =>
    void call(command)
      .then(() => refresh(projectKey, pagesKey, pageKey))
      .catch(() => undefined)

  return (
    <NavigationBar
      title={compact ? undefined : project?.name}
      leading={
        compact && (
          <Button
            type='button'
            variant='ghost'
            className='h-9 max-w-[45vw] gap-1 px-2 text-[15px] font-semibold'
            aria-label={t('navigator.pages')}
            onClick={onShowPages}
          >
            <span className='truncate'>{page?.label ?? t('navigator.pages')}</span>
            <ChevronDown className='size-4 shrink-0 text-muted-foreground' />
          </Button>
        )
      }
      trailing={
        <>
          <Button
            type='button'
            variant='ghost'
            size='icon'
            className='size-9 text-primary'
            aria-label={t('menu.undo')}
            disabled={!project?.can_undo}
            onClick={() => history(commands.undo)}
          >
            <Undo2 className='size-5' />
          </Button>
          <Button
            type='button'
            variant='ghost'
            size='icon'
            className='size-9 text-primary'
            aria-label={t('menu.redo')}
            disabled={!project?.can_redo}
            onClick={() => history(commands.redo)}
          >
            <Redo2 className='size-5' />
          </Button>
          <DropdownMenu>
            <DropdownMenuTrigger
              render={
                <Button
                  type='button'
                  variant='ghost'
                  size='icon'
                  className='size-9 text-primary'
                  aria-label={t('mobile.more')}
                />
              }
            >
              <Ellipsis className='size-5' />
            </DropdownMenuTrigger>
            <DropdownMenuContent align='end' className='min-w-52'>
              <DropdownMenuItem disabled={importing} onClick={() => importPages('files')}>
                <FilePlus2 /> {t('menu.import')}
              </DropdownMenuItem>
              <DropdownMenuSub>
                <DropdownMenuSubTrigger disabled={pages.length === 0 || exporting}>
                  <Share /> {t('menu.export')}
                </DropdownMenuSubTrigger>
                <DropdownMenuSubContent>
                  {(['png', 'psd', 'cbz'] as const).map((format) => (
                    <DropdownMenuItem
                      key={format}
                      disabled={exporting}
                      onClick={() => exportProject(format)}
                    >
                      {format.toUpperCase()}…
                    </DropdownMenuItem>
                  ))}
                </DropdownMenuSubContent>
              </DropdownMenuSub>
              <DropdownMenuItem disabled={!page} onClick={requestCanvasFit}>
                <Maximize /> {t('canvas.fit')}
              </DropdownMenuItem>
              <DropdownMenuSeparator />
              <DropdownMenuItem
                onClick={() => void call(commands.closeProject).catch(() => undefined)}
              >
                <X /> {t('menu.closeProject')}
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </>
      }
    />
  )
}

/** The canvas full width, its tools in a strip below, and pages and properties in sheets. */
function PhoneEditor({
  pagesOpen,
  onPagesOpenChange,
}: {
  pagesOpen: boolean
  onPagesOpenChange: (open: boolean) => void
}) {
  const { t } = useTranslation()
  const activePage = useProject().data?.active_page
  const [inspectorOpen, setInspectorOpen] = useState(false)

  // Choosing a page in the sheet is the reason it was open.
  useEffect(() => {
    onPagesOpenChange(false)
  }, [activePage, onPagesOpenChange])

  return (
    <ColorSamplingProvider>
      <div className='relative flex min-h-0 flex-1 flex-col'>
        <ActivityCenter />
        <CanvasCommandBar />
        <div className='relative min-h-0 flex-1'>
          <CanvasWorkspace chrome='touch' />
        </div>
        <div className='flex shrink-0 items-stretch bg-[var(--surface-toolbar)]'>
          <div className='min-w-0 flex-1'>
            <ToolBar orientation='horizontal' />
          </div>
          <Button
            type='button'
            variant='ghost'
            size='icon'
            className='h-12 w-12 shrink-0 rounded-none border-t border-l border-border/80 text-primary'
            aria-label={t('agent.properties')}
            aria-pressed={inspectorOpen}
            onClick={() => setInspectorOpen(!inspectorOpen)}
          >
            <SlidersHorizontal className='size-5' />
          </Button>
        </div>
      </div>

      <Drawer open={pagesOpen} onOpenChange={onPagesOpenChange} showSwipeHandle>
        <DrawerContent className='h-[75dvh] bg-[var(--surface-sidebar)] pb-[env(safe-area-inset-bottom)]'>
          <DrawerTitle className='sr-only'>{t('navigator.pages')}</DrawerTitle>
          <PageRail />
        </DrawerContent>
      </Drawer>

      {/* Like Maps: the canvas stays usable under the sheet, which rests at half or full height. */}
      <Drawer
        open={inspectorOpen}
        onOpenChange={setInspectorOpen}
        modal={false}
        disablePointerDismissal
        snapPoints={[0.45, 0.92]}
        showSwipeHandle
      >
        <DrawerContent className='bg-[var(--surface-panel)] pb-[env(safe-area-inset-bottom)]'>
          <DrawerTitle className='sr-only'>{t('agent.properties')}</DrawerTitle>
          <RightSidebar />
        </DrawerContent>
      </Drawer>
    </ColorSamplingProvider>
  )
}
