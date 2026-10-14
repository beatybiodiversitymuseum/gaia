import type { Metadata } from 'next';
import './globals.css';

const site_url = 'https://beaty-biodiversity-query.noslopplsthx.chatgpt.site';
const site_title = 'Ask Gaia | Beaty Biodiversity Museum Collections';
const site_description =
  'Query the Beaty Biodiversity Museum collections with Gaia.';

export const metadata: Metadata = {
  metadataBase: new URL(site_url),
  title: site_title,
  description: site_description,
  icons: {
    icon: '/assets/images.png',
    apple: '/assets/images.png',
  },
  alternates: {
    canonical: '/',
  },
  openGraph: {
    type: 'website',
    url: site_url,
    siteName: 'Ask Gaia',
    title: site_title,
    description: site_description,
    images: [
      {
        url: `${site_url}/assets/og.png`,
        width: 1672,
        height: 941,
        alt: 'Ask Gaia — Explore the biodiversity collections.',
      },
    ],
  },
  twitter: {
    card: 'summary_large_image',
    title: site_title,
    description: site_description,
    images: [`${site_url}/assets/og.png`],
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
