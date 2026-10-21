'use client'

import { Check, ChevronRight, Folder, MinusCircle, Plus, Trash2 } from 'lucide-react'
import { useState, type FormEvent } from 'react'
import { useTranslation } from 'react-i18next'

import { NavigationBar } from '@/components/mobile/NavigationBar'
import { useProjectLibrary } from '@/lib/projects'
import { useProject } from '@/lib/queries'
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogMedia,
  AlertDialogTitle,
} from '@koharu/ui/components/alert-dialog'
import { Button } from '@koharu/ui/components/button'
import { Drawer, DrawerContent, DrawerTitle } from '@koharu/ui/components/drawer'
import { Input } from '@koharu/ui/components/input'

/** The Projects tab: the project library as an inset list, with creation in a sheet. */
export function ProjectsTab({ onOpened }: { onOpened: () => void }) {
  const { t } = useTranslation()
  const current = useProject().data?.name
  const { projects, busy, createProject, openProject, deleteProject } = useProjectLibrary()
  const [editing, setEditing] = useState(false)
  const [creating, setCreating] = useState(false)
  const [name, setName] = useState('')
  const [projectToDelete, setProjectToDelete] = useState<string | null>(null)

  const open = async (projectName: string) => {
    if (projectName === current || (await openProject(projectName))) onOpened()
  }

  const create = async (event: FormEvent) => {
    event.preventDefault()
    const projectName = name.trim()
    if (!projectName || !(await createProject(projectName))) return
    setName('')
    setCreating(false)
    onOpened()
  }

  const confirmDelete = async () => {
    if (projectToDelete && (await deleteProject(projectToDelete))) setProjectToDelete(null)
  }

  return (
    <section className='flex min-h-0 flex-1 flex-col'>
      <NavigationBar
        largeTitle={t('start.title')}
        leading={
          projects.length > 0 && (
            <Button
              type='button'
              variant='ghost'
              className='h-9 px-2 text-[15px] text-primary'
              onClick={() => setEditing(!editing)}
            >
              {editing ? t('mobile.done') : t('mobile.edit')}
            </Button>
          )
        }
        trailing={
          <Button
            type='button'
            variant='ghost'
            size='icon'
            className='size-9 text-primary'
            aria-label={t('start.newProject')}
            onClick={() => setCreating(true)}
          >
            <Plus className='size-6' />
          </Button>
        }
      />
      <div className='min-h-0 flex-1 overflow-y-auto px-4 pb-6'>
        {busy === 'list' && projects.length === 0 ? (
          <p className='py-16 text-center text-[15px] text-muted-foreground'>
            {t('start.loadingProjects')}
          </p>
        ) : projects.length === 0 ? (
          <div className='grid place-items-center py-16 text-center' role='status'>
            <span className='grid size-14 place-items-center rounded-2xl bg-muted text-muted-foreground'>
              <Folder className='size-7' />
            </span>
            <p className='mt-4 text-[17px] font-semibold'>{t('start.emptyTitle')}</p>
            <p className='mt-1 max-w-[30ch] text-[14px] text-muted-foreground'>
              {t('start.emptyDescription')}
            </p>
            <Button
              type='button'
              className='mt-5 h-11 px-5 text-[15px]'
              onClick={() => setCreating(true)}
            >
              <Plus className='size-4' /> {t('start.newProject')}
            </Button>
          </div>
        ) : (
          <ul
            className='mt-2 divide-y divide-border/70 overflow-hidden rounded-xl bg-[var(--surface-panel)]'
            aria-label={t('start.projectList')}
          >
            {projects.map((project) => (
              <li key={project.name} className='flex items-center'>
                {editing && (
                  <Button
                    type='button'
                    variant='ghost'
                    size='icon'
                    className='ml-2 size-9 text-destructive'
                    aria-label={t('start.deleteLabel', { name: project.name })}
                    disabled={busy !== null}
                    onClick={() => setProjectToDelete(project.name)}
                  >
                    <MinusCircle className='size-5 fill-destructive text-background' />
                  </Button>
                )}
                <button
                  type='button'
                  className='flex min-h-14 min-w-0 flex-1 items-center gap-3 px-4 py-2 text-left outline-none active:bg-foreground/[0.06] disabled:opacity-60'
                  disabled={busy !== null || editing}
                  onClick={() => void open(project.name).catch(() => undefined)}
                >
                  <span className='grid size-9 shrink-0 place-items-center rounded-lg bg-accent text-accent-foreground'>
                    <Folder className='size-[18px]' />
                  </span>
                  <span className='min-w-0 flex-1'>
                    <span className='block truncate text-[16px] font-medium'>{project.name}</span>
                    <span className='block text-[13px] text-muted-foreground'>
                      {t('start.projectKind')}
                    </span>
                  </span>
                  {project.name === current && (
                    <Check className='size-5 shrink-0 text-primary' aria-hidden='true' />
                  )}
                  {!editing && (
                    <ChevronRight className='size-5 shrink-0 text-muted-foreground/70' />
                  )}
                </button>
              </li>
            ))}
          </ul>
        )}
        <p className='mt-3 px-4 text-[12px] text-muted-foreground'>{t('start.storageHint')}</p>
      </div>

      <Drawer open={creating} onOpenChange={setCreating} showSwipeHandle>
        <DrawerContent className='pb-[env(safe-area-inset-bottom)]'>
          <form
            className='grid gap-4 p-5'
            onSubmit={(event) => void create(event).catch(() => undefined)}
          >
            <DrawerTitle className='text-[17px] font-semibold'>{t('start.newProject')}</DrawerTitle>
            <Input
              value={name}
              onChange={(event) => setName(event.target.value)}
              placeholder={t('start.projectNamePlaceholder')}
              aria-label={t('start.projectName')}
              autoComplete='off'
              disabled={busy !== null}
              className='h-11 text-[16px]'
            />
            <Button
              type='submit'
              className='h-11 text-[15px]'
              disabled={!name.trim() || busy !== null}
            >
              {t('start.create')}
            </Button>
          </form>
        </DrawerContent>
      </Drawer>

      <AlertDialog
        open={projectToDelete !== null}
        onOpenChange={(open) => {
          if (!open && busy === null) setProjectToDelete(null)
        }}
      >
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogMedia className='bg-destructive/10 text-destructive'>
              <Trash2 className='size-5' />
            </AlertDialogMedia>
            <AlertDialogTitle>{t('start.deleteTitle')}</AlertDialogTitle>
            <AlertDialogDescription>
              {t('start.deleteDescription', { name: projectToDelete })}
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel disabled={busy !== null}>{t('common.cancel')}</AlertDialogCancel>
            <AlertDialogAction
              variant='destructive'
              disabled={busy !== null}
              aria-busy={busy !== null}
              onClick={() => void confirmDelete().catch(() => undefined)}
            >
              {busy !== null ? t('start.deleting') : t('start.deleteAction')}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </section>
  )
}
