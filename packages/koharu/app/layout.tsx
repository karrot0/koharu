import type { Metadata, Viewport } from 'next'
import { ThemeProvider } from 'next-themes'
import localFont from 'next/font/local'

import './globals.css'
import Providers from '@/app/providers'

const notoSansCJK = localFont({
  src: './fonts/NotoSansCJK-VF.otf',
  weight: '100 900',
  variable: '--font-noto-sans-cjk',
})

export const metadata: Metadata = {
  title: 'Koharu',
}

// Fill the whole screen on iOS (content keeps clear of the notch and home indicator through
// safe-area insets) and leave pinch zoom to the canvas instead of the page.
export const viewport: Viewport = {
  width: 'device-width',
  initialScale: 1,
  maximumScale: 1,
  userScalable: false,
  viewportFit: 'cover',
}

function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang='en-US' suppressHydrationWarning style={{ backgroundColor: 'transparent' }}>
      <body
        className={`${notoSansCJK.variable} antialiased`}
        style={{ backgroundColor: 'transparent' }}
      >
        <ThemeProvider attribute='class' defaultTheme='system' enableSystem>
          <Providers>{children}</Providers>
        </ThemeProvider>
      </body>
    </html>
  )
}

export default RootLayout
