import { read_gaia_session, unauthorized_response } from '../_auth';
import { proxy_gaia_request } from '../_proxy';

const identifier_pattern = /^[A-Za-z0-9_-]{1,128}$/;

export async function GET(request: Request) {
  const username = await read_gaia_session(request);
  if (!username) {
    return unauthorized_response();
  }

  const parameters = new URL(request.url).searchParams;
  const conversation_id = parameters.get('conversation_id');
  const file_id = parameters.get('file_id');
  if (
    [...parameters.keys()].some(
      (key) => key !== 'conversation_id' && key !== 'file_id',
    ) ||
    parameters.getAll('conversation_id').length !== 1 ||
    parameters.getAll('file_id').length !== 1 ||
    !conversation_id ||
    !identifier_pattern.test(conversation_id) ||
    !file_id ||
    !identifier_pattern.test(file_id)
  ) {
    return Response.json(
      {
        error: {
          code: 'invalid_request',
          message: 'A valid conversation ID and file ID are required.',
        },
      },
      { status: 400, headers: { 'Cache-Control': 'no-store' } },
    );
  }

  const backend_parameters = new URLSearchParams({
    conversation_id,
    file_id,
    username,
  });
  return proxy_gaia_request(
    `/api/files?${backend_parameters}`,
    {
      method: 'GET',
      headers: {
        Accept:
          'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet, application/json',
      },
      signal: request.signal,
    },
    2 * 60 * 1_000,
  );
}
