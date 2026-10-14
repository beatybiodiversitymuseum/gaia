import { read_gaia_session, unauthorized_response } from './_auth';

const gaia_local_api_url = (
  process.env.GAIA_LOCAL_API_URL || 'http://127.0.0.1:8765'
).replace(/\/+$/, '');

export async function proxy_gaia_post(
  path: string,
  request: Request,
  timeout_ms: number,
  accept = 'application/json',
) {
  const username = await read_gaia_session(request);
  if (!username) {
    return unauthorized_response();
  }

  const body: unknown = await request.json().catch(() => null);
  if (typeof body !== 'object' || body === null || Array.isArray(body)) {
    return Response.json(
      {
        error: {
          code: 'invalid_request',
          message: 'The request body must be a JSON object.',
        },
      },
      { status: 400, headers: { 'Cache-Control': 'no-store' } },
    );
  }

  return proxy_gaia_request(
    path,
    {
      method: 'POST',
      headers: { Accept: accept, 'Content-Type': 'application/json' },
      body: JSON.stringify({ ...body, username }),
      signal: request.signal,
    },
    timeout_ms,
  );
}

export async function proxy_gaia_request(
  path: string,
  init: RequestInit,
  timeout_ms: number,
) {
  try {
    const headers = new Headers(init.headers);
    if (!headers.has('Accept')) {
      headers.set('Accept', 'application/json');
    }
    const timeout_signal = AbortSignal.timeout(timeout_ms);
    const signal = init.signal
      ? AbortSignal.any([init.signal, timeout_signal])
      : timeout_signal;
    const response = await fetch(`${gaia_local_api_url}${path}`, {
      ...init,
      headers,
      signal,
    });

    const response_headers = new Headers({ 'Cache-Control': 'no-store' });
    for (const name of [
      'Content-Type',
      'Content-Disposition',
      'Content-Length',
      'X-Content-Type-Options',
    ]) {
      const value = response.headers.get(name);
      if (value) {
        response_headers.set(name, value);
      }
    }

    return new Response(response.body, {
      status: response.status,
      headers: response_headers,
    });
  } catch {
    return Response.json(
      {
        error: {
          code: 'service_unavailable',
          message: 'Gaia is not available on this computer right now.',
        },
      },
      {
        status: 503,
        headers: { 'Cache-Control': 'no-store' },
      },
    );
  }
}
