import { useEffect, useState } from 'react'

export type Platform = 'macos' | 'ios' | 'other'

function detectPlatform(): Platform {
  const { userAgent, platform, maxTouchPoints } = navigator
  // iPadOS WebKit reports a Mac user agent; only its touch screen tells the two apart.
  if (/iPhone|iPad/.test(userAgent) || (userAgent.includes('Macintosh') && maxTouchPoints > 1)) {
    return 'ios'
  }
  if (userAgent.includes('Macintosh') || platform.toLowerCase().startsWith('mac')) return 'macos'
  return 'other'
}

/** Returns the host platform once mounted, or `null` during the static prerender. */
export function usePlatform() {
  const [platform, setPlatform] = useState<Platform | null>(null)

  useEffect(() => {
    setPlatform(detectPlatform())
  }, [])

  return platform
}
