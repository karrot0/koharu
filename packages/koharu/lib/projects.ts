'use client'

import { useCallback, useEffect, useState } from 'react'

import { call } from '@/lib/backend'
import { pageKey, pagesKey, projectKey, refresh } from '@/lib/queries'
import { commands, type ProjectSummary } from '@koharu/bridge/protocol'

/**
 * The project library behind the desktop start view and the iOS Projects tab: listing, creating,
 * opening, and deleting projects. `busy` names the operation in flight (`'list'`, `'create'`, or a
 * project name) so only one runs at a time.
 */
export function useProjectLibrary() {
  const [projects, setProjects] = useState<ProjectSummary[]>([])
  const [busy, setBusy] = useState<string | null>('list')

  const reload = useCallback(async () => {
    setBusy('list')
    try {
      setProjects(await call(commands.listProjects))
    } finally {
      setBusy(null)
    }
  }, [])

  useEffect(() => {
    void reload().catch(() => undefined)
  }, [reload])

  const createProject = async (name: string) => {
    if (busy) return false
    setBusy('create')
    try {
      await call(commands.createProject, name)
      await refresh(projectKey, pagesKey, pageKey)
      return true
    } finally {
      setBusy(null)
    }
  }

  const openProject = async (name: string) => {
    if (busy) return false
    setBusy(name)
    try {
      await call(commands.openProject, name)
      await refresh(projectKey, pagesKey, pageKey)
      return true
    } finally {
      setBusy(null)
    }
  }

  const deleteProject = async (name: string) => {
    if (busy) return false
    setBusy(name)
    try {
      await call(commands.deleteProject, name)
      setProjects(await call(commands.listProjects))
      return true
    } finally {
      setBusy(null)
    }
  }

  return { projects, busy, createProject, openProject, deleteProject }
}
