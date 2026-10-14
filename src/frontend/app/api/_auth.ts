const session_lifetime_seconds = 12 * 60 * 60;
const minimum_session_secret_length = 32;
const gaia_ui_password = process.env.GAIA_UI_PASSWORD;
const gaia_session_secret = process.env.GAIA_SESSION_SECRET?.trim();
const authorization_scheme = 'Bearer ';
const text_encoder = new TextEncoder();
const text_decoder = new TextDecoder();

function encode_base64url(bytes: Uint8Array) {
  let binary = '';
  for (const byte of bytes) {
    binary += String.fromCodePoint(byte);
  }
  return btoa(binary)
    .replaceAll('+', '-')
    .replaceAll('/', '_')
    .replace(/=+$/, '');
}

function decode_base64url(value: string) {
  const base64 = value.replaceAll('-', '+').replaceAll('_', '/');
  const padded = base64.padEnd(Math.ceil(base64.length / 4) * 4, '=');
  const binary = atob(padded);
  return Uint8Array.from(binary, (character) => character.codePointAt(0) ?? 0);
}

async function sign_payload(payload: string) {
  if (!authentication_is_configured()) {
    throw new Error('Gaia authentication is not configured.');
  }
  const key = await crypto.subtle.importKey(
    'raw',
    text_encoder.encode(`gaia-session-v1:${gaia_session_secret}`),
    { name: 'HMAC', hash: 'SHA-256' },
    false,
    ['sign'],
  );
  const signature = await crypto.subtle.sign(
    'HMAC',
    key,
    text_encoder.encode(payload),
  );
  return encode_base64url(new Uint8Array(signature));
}

function signatures_match(left: string, right: string) {
  if (left.length !== right.length) {
    return false;
  }
  let difference = 0;
  for (let index = 0; index < left.length; index += 1) {
    difference |= left.codePointAt(index)! ^ right.codePointAt(index)!;
  }
  return difference === 0;
}

function request_session_token(request: Request) {
  const authorization = request.headers.get('Authorization');
  if (!authorization?.startsWith(authorization_scheme)) {
    return null;
  }
  const token = authorization.slice(authorization_scheme.length).trim();
  return token || null;
}

export function password_is_valid(password: string) {
  return authentication_is_configured() && password === gaia_ui_password;
}

export function authentication_is_configured() {
  return Boolean(
    gaia_ui_password &&
      gaia_session_secret &&
      gaia_session_secret.length >= minimum_session_secret_length &&
      gaia_session_secret !== gaia_ui_password,
  );
}

export async function create_gaia_session(username: string) {
  const payload = encode_base64url(
    text_encoder.encode(
      JSON.stringify({
        username,
        expires_at: Math.floor(Date.now() / 1000) + session_lifetime_seconds,
      }),
    ),
  );
  return `${payload}.${await sign_payload(payload)}`;
}

export async function read_gaia_session(request: Request) {
  if (!authentication_is_configured()) {
    return null;
  }
  const session = request_session_token(request);
  if (!session) {
    return null;
  }

  const [payload, signature, ...remainder] = session.split('.');
  if (!payload || !signature || remainder.length) {
    return null;
  }
  const expected_signature = await sign_payload(payload);
  if (!signatures_match(signature, expected_signature)) {
    return null;
  }

  try {
    const value: unknown = JSON.parse(
      text_decoder.decode(decode_base64url(payload)),
    );
    if (
      typeof value !== 'object' ||
      value === null ||
      !('username' in value) ||
      !('expires_at' in value) ||
      typeof value.username !== 'string' ||
      !value.username ||
      typeof value.expires_at !== 'number' ||
      !Number.isSafeInteger(value.expires_at) ||
      value.expires_at <= Math.floor(Date.now() / 1000)
    ) {
      return null;
    }
    return value.username;
  } catch {
    return null;
  }
}

export function unauthorized_response() {
  return Response.json(
    {
      error: {
        code: 'authentication_required',
        message: 'Sign in to use Gaia.',
      },
    },
    {
      status: 401,
      headers: { 'Cache-Control': 'no-store' },
    },
  );
}
