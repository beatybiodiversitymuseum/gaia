import {
  authentication_is_configured,
  create_gaia_session,
  password_is_valid,
  read_gaia_session,
} from '../_auth';

export const dynamic = 'force-dynamic';

export async function GET(request: Request) {
  const username = await read_gaia_session(request);
  return Response.json(
    {
      authenticated: Boolean(username),
      ...(username ? { username } : {}),
    },
    { headers: { 'Cache-Control': 'no-store' } },
  );
}

export async function POST(request: Request) {
  if (!authentication_is_configured()) {
    return Response.json(
      { error: { message: 'Gaia authentication is not configured.' } },
      { status: 503, headers: { 'Cache-Control': 'no-store' } },
    );
  }
  const credentials: unknown = await request.json().catch(() => null);

  if (
    typeof credentials !== 'object' ||
    credentials === null ||
    !('username' in credentials) ||
    !('password' in credentials) ||
    typeof credentials.username !== 'string' ||
    typeof credentials.password !== 'string'
  ) {
    return Response.json(
      { error: { message: 'Enter a username and password.' } },
      { status: 400, headers: { 'Cache-Control': 'no-store' } },
    );
  }

  const username = credentials.username.trim();
  if (!username || username.length > 64 || !/[A-Za-z0-9]/.test(username)) {
    return Response.json(
      {
        error: {
          message: 'Username must contain at least one ASCII letter or number.',
        },
      },
      { status: 400, headers: { 'Cache-Control': 'no-store' } },
    );
  }
  if (!password_is_valid(credentials.password)) {
    return Response.json(
      { error: { message: 'Incorrect password.' } },
      { status: 401, headers: { 'Cache-Control': 'no-store' } },
    );
  }

  const token = await create_gaia_session(username);
  return Response.json(
    { authenticated: true, username, token },
    {
      headers: { 'Cache-Control': 'no-store' },
    },
  );
}
