'use client'

import type { ReactNode } from 'react'

/**
 * An iOS navigation bar below the status bar: leading and trailing controls around an inline
 * title, with an optional large title underneath for a tab's root screen.
 */
export function NavigationBar({
  title,
  largeTitle,
  leading,
  trailing,
}: {
  title?: ReactNode
  largeTitle?: string
  leading?: ReactNode
  trailing?: ReactNode
}) {
  return (
    <header className='shrink-0 bg-[var(--surface-titlebar)] pt-[env(safe-area-inset-top)]'>
      <div className='relative flex h-11 items-center gap-1 px-2'>
        <div className='relative z-10 flex min-w-0 items-center gap-1'>{leading}</div>
        {title && (
          <div className='pointer-events-none absolute inset-x-28 truncate text-center text-[15px] font-semibold'>
            {title}
          </div>
        )}
        <div className='relative z-10 ml-auto flex shrink-0 items-center gap-1'>{trailing}</div>
      </div>
      {largeTitle && (
        <h1 className='px-4 pb-2 text-[30px] leading-tight font-bold tracking-[-0.02em]'>
          {largeTitle}
        </h1>
      )}
    </header>
  )
}
