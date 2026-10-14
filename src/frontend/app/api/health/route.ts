import { proxy_gaia_request } from '../_proxy';
import { read_gaia_session, unauthorized_response } from '../_auth';

export async function GET(request: Request) {
  if (!(await read_gaia_session(request))) {
    return unauthorized_response();
  }
  return proxy_gaia_request(
    '/api/health',
    { method: 'GET', signal: request.signal },
    4_000,
  );
}
