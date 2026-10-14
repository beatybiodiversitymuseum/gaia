import { proxy_gaia_post, proxy_gaia_request } from '../_proxy';
import { read_gaia_session, unauthorized_response } from '../_auth';

export async function POST(request: Request) {
  return proxy_gaia_post(
    '/api/chat',
    request,
    10 * 60 * 1_000,
    'application/x-ndjson',
  );
}

export async function DELETE(request: Request) {
  const username = await read_gaia_session(request);
  if (!username) {
    return unauthorized_response();
  }

  const request_body: unknown = await request.json().catch(() => null);
  if (
    typeof request_body !== 'object' ||
    request_body === null ||
    Array.isArray(request_body) ||
    !('conversation_id' in request_body) ||
    typeof request_body.conversation_id !== 'string' ||
    !request_body.conversation_id
  ) {
    return Response.json(
      {
        error: {
          code: 'invalid_request',
          message: 'A conversation ID is required to stop a request.',
        },
      },
      { status: 400, headers: { 'Cache-Control': 'no-store' } },
    );
  }

  return proxy_gaia_request(
    '/api/chat',
    {
      method: 'DELETE',
      headers: {
        Accept: 'application/json',
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        username,
        conversation_id: request_body.conversation_id,
      }),
    },
    10 * 60 * 1_000,
  );
}
