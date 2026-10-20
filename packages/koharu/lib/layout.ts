import { useSyncExternalStore } from 'react'

// Below this width the three-panel editor leaves the canvas too little room: phones, and
// tablets in portrait.
const compactQuery = '(max-width: 1023px)'

function subscribe(onChange: () => void) {
  const query = window.matchMedia(compactQuery)
  query.addEventListener('change', onChange)
  return () => query.removeEventListener('change', onChange)
}

/** Whether the editor shows only the canvas, with its side panels in sheets. */
export function useCompactLayout() {
  return useSyncExternalStore(
    subscribe,
    () => window.matchMedia(compactQuery).matches,
    () => false,
  )
}
