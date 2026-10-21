import { useSyncExternalStore } from 'react'

// Below this width the three-panel editor leaves the canvas too little room: iPhones, and iPads
// in portrait.
const compactQuery = '(max-width: 1023px)'

function subscribe(onChange: () => void) {
  const query = window.matchMedia(compactQuery)
  query.addEventListener('change', onChange)
  return () => query.removeEventListener('change', onChange)
}

/** Whether the iOS editor shows only the canvas, with pages and properties in sheets. */
export function useCompactLayout() {
  return useSyncExternalStore(
    subscribe,
    () => window.matchMedia(compactQuery).matches,
    () => false,
  )
}
