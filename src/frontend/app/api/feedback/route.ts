import { proxy_gaia_post } from '../_proxy';

export async function POST(request: Request) {
  return proxy_gaia_post('/api/feedback', request, 10_000);
}
