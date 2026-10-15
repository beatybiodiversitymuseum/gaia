const chat_stream_media_type = 'application/x-ndjson';
const workbook_media_type =
  'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet';
const session_storage_key = 'gaia_session';
const identifier_pattern = /^[A-Za-z0-9_-]{1,128}$/;
export const context_compacted_notice =
  'Earlier conversation has been compacted. Continuing with its summary.';

export interface GaiaAttachment {
  file_id: string;
  filename: string;
  media_type: string;
  size_bytes: number;
}

export interface GaiaProgressActivity {
  id: string;
  kind: 'query' | 'catalogue' | 'export';
  status: 'completed' | 'failed';
  label: string;
  detail: string;
  interpretation?: string;
}

interface GaiaWorkingEvent {
  phase:
    | 'thinking'
    | 'searching'
    | 'reviewing'
    | 'preparing_file'
    | 'preparing_answer'
    | 'context_compacting'
    | 'context_compacted';
}

interface GaiaActivityEvent {
  phase: 'activity';
  activities: GaiaProgressActivity[];
}

export type GaiaProgressEvent = GaiaWorkingEvent | GaiaActivityEvent;

export interface GaiaResponseMetrics {
  elapsed_seconds: number;
}

export type GaiaJsonValue =
  | null
  | boolean
  | number
  | string
  | GaiaJsonValue[]
  | { [key: string]: GaiaJsonValue };

export interface GaiaVerificationStep {
  sequence: number;
  capability: string;
  purpose: string;
  interpretation?: string;
  arguments: Record<string, GaiaJsonValue>;
  status: 'completed' | 'failed';
  evidence: Record<string, GaiaJsonValue>;
}

export type GaiaResponseRating = 'good' | 'bad';

export interface GaiaResponseFeedback {
  rating: GaiaResponseRating;
  reasons: string[];
  details: string;
}

export interface GaiaChatResult {
  conversation_id: string;
  interaction: number;
  content: string;
  attachments: GaiaAttachment[];
  notices: string[];
  metrics: GaiaResponseMetrics;
  verification: GaiaVerificationStep[];
}

export interface GaiaSession {
  token: string;
  username: string;
}

export class GaiaRequestError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
  }
}

export class GaiaRequestCancelled extends Error {
  constructor(readonly conversation_id: string) {
    super('Stopped thinking');
  }
}

function is_record(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function response_error_message(value: unknown) {
  if (!is_record(value) || !is_record(value.error)) {
    return 'Gaia could not complete this request.';
  }
  return typeof value.error.message === 'string' && value.error.message.trim()
    ? value.error.message
    : 'Gaia could not complete this request.';
}

function parse_nonnegative_integer(value: unknown, label: string) {
  if (typeof value !== 'number' || !Number.isSafeInteger(value) || value < 0) {
    throw new Error(`Gaia returned invalid ${label}.`);
  }
  return value;
}

function parse_attachments(value: unknown): GaiaAttachment[] {
  if (!Array.isArray(value)) {
    throw new Error('Gaia returned invalid file attachments.');
  }
  const file_ids = new Set<string>();
  return value.map((attachment: unknown) => {
    if (
      !is_record(attachment) ||
      typeof attachment.file_id !== 'string' ||
      !identifier_pattern.test(attachment.file_id) ||
      file_ids.has(attachment.file_id) ||
      typeof attachment.filename !== 'string' ||
      !attachment.filename.trim() ||
      attachment.filename.length > 255 ||
      /[/\\]/u.test(attachment.filename) ||
      Array.from(attachment.filename).some(
        (character) =>
          character.charCodeAt(0) < 32 || character.charCodeAt(0) === 127,
      ) ||
      !attachment.filename.toLowerCase().endsWith('.xlsx') ||
      attachment.media_type !== workbook_media_type
    ) {
      throw new Error('Gaia returned an invalid file attachment.');
    }
    file_ids.add(attachment.file_id);
    return {
      file_id: attachment.file_id,
      filename: attachment.filename,
      media_type: attachment.media_type,
      size_bytes: parse_nonnegative_integer(attachment.size_bytes, 'file size'),
    };
  });
}

function parse_json_object(
  value: unknown,
  label: string,
  maximum_serialized_length: number,
): Record<string, GaiaJsonValue> {
  let node_count = 0;

  function validate_json(candidate: unknown, depth: number): GaiaJsonValue {
    node_count += 1;
    if (node_count > 10_000 || depth > 8) {
      throw new Error(`Gaia returned invalid ${label}.`);
    }
    if (
      candidate === null ||
      typeof candidate === 'boolean' ||
      typeof candidate === 'string'
    ) {
      return candidate;
    }
    if (typeof candidate === 'number' && Number.isFinite(candidate)) {
      return candidate;
    }
    if (Array.isArray(candidate)) {
      return candidate.map((item) => validate_json(item, depth + 1));
    }
    if (is_record(candidate)) {
      const result: Record<string, GaiaJsonValue> = {};
      for (const [key, item] of Object.entries(candidate)) {
        if (!key || key.length > 256) {
          throw new Error(`Gaia returned invalid ${label}.`);
        }
        result[key] = validate_json(item, depth + 1);
      }
      return result;
    }
    throw new Error(`Gaia returned invalid ${label}.`);
  }

  if (!is_record(value)) {
    throw new Error(`Gaia returned invalid ${label}.`);
  }
  const parsed = validate_json(value, 0);
  if (
    Array.isArray(parsed) ||
    parsed === null ||
    typeof parsed !== 'object' ||
    JSON.stringify(parsed).length > maximum_serialized_length
  ) {
    throw new Error(`Gaia returned invalid ${label}.`);
  }
  return parsed;
}

function parse_verification(value: unknown): GaiaVerificationStep[] {
  if (!Array.isArray(value) || value.length > 15) {
    throw new Error('Gaia returned an invalid verification trace.');
  }
  return value.map((step, index) => {
    if (
      !is_record(step) ||
      step.sequence !== index + 1 ||
      typeof step.capability !== 'string' ||
      !/^[a-z][a-z0-9_]{0,127}$/u.test(step.capability) ||
      typeof step.purpose !== 'string' ||
      !step.purpose.trim() ||
      step.purpose.length > 500 ||
      (step.interpretation !== undefined &&
        (typeof step.interpretation !== 'string' ||
          !step.interpretation.trim() ||
          step.interpretation.length > 300)) ||
      !['completed', 'failed'].includes(String(step.status))
    ) {
      throw new Error('Gaia returned an invalid verification trace.');
    }
    return {
      sequence: step.sequence,
      capability: step.capability,
      purpose: step.purpose,
      interpretation: step.interpretation as string | undefined,
      arguments: parse_json_object(
        step.arguments,
        'verification arguments',
        50_000,
      ),
      status: step.status as GaiaVerificationStep['status'],
      evidence: parse_json_object(
        step.evidence,
        'verification evidence',
        250_000,
      ),
    };
  });
}

function parse_chat_result(value: unknown): GaiaChatResult {
  if (!is_record(value) || !is_record(value.message)) {
    throw new Error('Gaia returned an invalid response.');
  }
  const { message, metrics } = value;
  if (
    typeof value.conversation_id !== 'string' ||
    !identifier_pattern.test(value.conversation_id) ||
    typeof value.interaction !== 'number' ||
    !Number.isSafeInteger(value.interaction) ||
    value.interaction < 1 ||
    message.role !== 'assistant' ||
    typeof message.content !== 'string' ||
    !message.content.trim() ||
    !is_record(metrics)
  ) {
    throw new Error('Gaia returned an invalid response.');
  }
  if (
    typeof metrics.elapsed_seconds !== 'number' ||
    !Number.isFinite(metrics.elapsed_seconds) ||
    metrics.elapsed_seconds < 0
  ) {
    throw new Error('Gaia returned invalid elapsed time.');
  }
  const notices = message.notices === undefined ? [] : message.notices;
  if (
    !Array.isArray(notices) ||
    notices.length > 1 ||
    notices.some((notice) => notice !== context_compacted_notice)
  ) {
    throw new Error('Gaia returned an invalid conversation notice.');
  }
  return {
    conversation_id: value.conversation_id,
    interaction: value.interaction,
    content: message.content,
    attachments: parse_attachments(message.attachments),
    verification: parse_verification(value.verification),
    notices,
    metrics: {
      elapsed_seconds: metrics.elapsed_seconds,
    },
  };
}

function parse_progress_activity(value: unknown): GaiaProgressActivity {
  if (
    !is_record(value) ||
    typeof value.id !== 'string' ||
    !value.id ||
    !['query', 'catalogue', 'export'].includes(String(value.kind)) ||
    !['completed', 'failed'].includes(String(value.status)) ||
    typeof value.label !== 'string' ||
    !value.label.trim() ||
    typeof value.detail !== 'string' ||
    !value.detail.trim() ||
    (value.interpretation !== undefined &&
      (typeof value.interpretation !== 'string' ||
        !value.interpretation.trim() ||
        value.interpretation.length > 300))
  ) {
    throw new Error('Gaia returned an invalid activity trace.');
  }
  return {
    id: value.id,
    kind: value.kind as GaiaProgressActivity['kind'],
    status: value.status as GaiaProgressActivity['status'],
    label: value.label,
    detail: value.detail,
    interpretation: value.interpretation as string | undefined,
  };
}

function parse_progress_event(
  value: Record<string, unknown>,
): GaiaProgressEvent {
  const phase = String(value.phase);
  if (
    ![
      'thinking',
      'searching',
      'reviewing',
      'preparing_file',
      'preparing_answer',
      'context_compacting',
      'context_compacted',
      'activity',
    ].includes(phase)
  ) {
    throw new Error('Gaia returned an invalid progress update.');
  }

  if (phase === 'activity') {
    if (!Array.isArray(value.activities)) {
      throw new Error('Gaia returned an invalid completed activity update.');
    }
    return {
      phase,
      activities: value.activities.map(parse_progress_activity),
    };
  }

  return { phase: phase as GaiaWorkingEvent['phase'] };
}

interface StreamState {
  conversation_id: string | null;
}

function parse_stream_line(
  line: string,
  state: StreamState,
  on_progress: (progress: GaiaProgressEvent) => void,
  on_conversation_started: (conversation_id: string) => void,
) {
  let event: unknown;
  try {
    event = JSON.parse(line);
  } catch {
    throw new Error('Gaia returned an invalid response stream.');
  }
  if (!is_record(event) || typeof event.type !== 'string') {
    throw new Error('Gaia returned an invalid response stream.');
  }

  if (event.type === 'conversation') {
    if (
      state.conversation_id !== null ||
      typeof event.conversation_id !== 'string' ||
      !event.conversation_id
    ) {
      throw new Error('Gaia returned an invalid conversation update.');
    }
    state.conversation_id = event.conversation_id;
    on_conversation_started(event.conversation_id);
    return undefined;
  }

  if (event.type === 'progress') {
    if (state.conversation_id === null) {
      throw new Error('Gaia returned progress before starting a conversation.');
    }
    on_progress(parse_progress_event(event));
    return undefined;
  }

  if (event.type === 'response') {
    if (state.conversation_id === null) {
      throw new Error('Gaia returned an answer without a conversation.');
    }
    const result = parse_chat_result(event.data);
    if (result.conversation_id !== state.conversation_id) {
      throw new Error('Gaia returned conflicting conversation identifiers.');
    }
    return result;
  }

  if (event.type === 'cancelled') {
    if (
      state.conversation_id === null ||
      event.conversation_id !== state.conversation_id
    ) {
      throw new Error('Gaia returned an invalid cancellation update.');
    }
    throw new GaiaRequestCancelled(state.conversation_id);
  }

  if (event.type === 'error') {
    const status =
      typeof event.status === 'number' &&
      Number.isSafeInteger(event.status) &&
      event.status >= 400 &&
      event.status <= 599
        ? event.status
        : 500;
    throw new GaiaRequestError(response_error_message(event), status);
  }

  throw new Error('Gaia returned an unknown response-stream event.');
}

async function read_chat_stream(
  response: Response,
  on_progress: (progress: GaiaProgressEvent) => void,
  on_conversation_started: (conversation_id: string) => void,
) {
  if (!response.body) {
    throw new Error('Gaia returned an empty response stream.');
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  const state: StreamState = { conversation_id: null };
  let pending_text = '';

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) {
        pending_text += decoder.decode();
        break;
      }
      pending_text += decoder.decode(value, { stream: true });

      let newline_index = pending_text.indexOf('\n');
      while (newline_index >= 0) {
        const line = pending_text.slice(0, newline_index).trim();
        pending_text = pending_text.slice(newline_index + 1);
        if (line) {
          const result = parse_stream_line(
            line,
            state,
            on_progress,
            on_conversation_started,
          );
          if (result) {
            await reader.cancel().catch(() => undefined);
            return result;
          }
        }
        newline_index = pending_text.indexOf('\n');
      }
    }

    const final_line = pending_text.trim();
    if (final_line) {
      const result = parse_stream_line(
        final_line,
        state,
        on_progress,
        on_conversation_started,
      );
      if (result) {
        return result;
      }
    }
    throw new Error('Gaia ended the response stream before sending an answer.');
  } catch (error) {
    await reader.cancel().catch(() => undefined);
    throw error;
  } finally {
    reader.releaseLock();
  }
}

function authorization_headers(token: string) {
  return { Authorization: `Bearer ${token}` };
}

async function error_from_response(response: Response) {
  const response_body: unknown = await response.json().catch(() => null);
  return new GaiaRequestError(
    response_error_message(response_body),
    response.status,
  );
}

async function read_json_response(response: Response): Promise<unknown> {
  if (!response.ok) {
    throw await error_from_response(response);
  }
  return response.json().catch(() => null);
}

export function read_gaia_session_token() {
  const token = globalThis.sessionStorage.getItem(session_storage_key)?.trim();
  return token || null;
}

export function store_gaia_session_token(token: string) {
  globalThis.sessionStorage.setItem(session_storage_key, token);
}

export function clear_gaia_session_token() {
  globalThis.sessionStorage.removeItem(session_storage_key);
}

export async function create_gaia_session(username: string, password: string) {
  const response = await fetch('/api/session', {
    method: 'POST',
    headers: {
      Accept: 'application/json',
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ username, password }),
  });
  const response_body = await read_json_response(response);
  if (
    !is_record(response_body) ||
    response_body.authenticated !== true ||
    typeof response_body.token !== 'string' ||
    !response_body.token ||
    typeof response_body.username !== 'string' ||
    !response_body.username
  ) {
    throw new Error('Gaia returned an invalid sign-in response.');
  }
  return {
    token: response_body.token,
    username: response_body.username,
  } satisfies GaiaSession;
}

export async function restore_gaia_session(token: string, signal: AbortSignal) {
  const response = await fetch('/api/session', {
    cache: 'no-store',
    headers: {
      Accept: 'application/json',
      ...authorization_headers(token),
    },
    signal,
  });
  const response_body: unknown = await response.json().catch(() => null);
  if (
    !response.ok ||
    !is_record(response_body) ||
    response_body.authenticated !== true ||
    typeof response_body.username !== 'string' ||
    !response_body.username
  ) {
    return null;
  }
  return response_body.username;
}

export async function check_gaia_health(token: string, signal: AbortSignal) {
  const response = await fetch('/api/health', {
    cache: 'no-store',
    headers: {
      Accept: 'application/json',
      ...authorization_headers(token),
    },
    signal,
  });
  if (response.status === 401) {
    throw new GaiaRequestError('Your session ended. Sign in again.', 401);
  }
  return response.ok;
}

export async function send_gaia_message({
  token,
  message,
  conversation_id,
  signal,
  on_conversation_started,
  on_progress,
}: {
  token: string;
  message: string;
  conversation_id: string | null;
  signal: AbortSignal;
  on_conversation_started: (conversation_id: string) => void;
  on_progress: (progress: GaiaProgressEvent) => void;
}) {
  const response = await fetch('/api/chat', {
    method: 'POST',
    headers: {
      Accept: chat_stream_media_type,
      'Content-Type': 'application/json',
      ...authorization_headers(token),
    },
    body: JSON.stringify({
      message,
      ...(conversation_id ? { conversation_id } : {}),
    }),
    signal,
  });
  if (!response.ok) {
    throw await error_from_response(response);
  }

  const response_media_type = response.headers
    .get('Content-Type')
    ?.split(';', 1)[0]
    .trim()
    .toLowerCase();
  if (response_media_type !== chat_stream_media_type) {
    throw new Error('Gaia returned an unsupported response format.');
  }
  return read_chat_stream(response, on_progress, on_conversation_started);
}

export async function cancel_gaia_message(
  token: string,
  conversation_id: string,
) {
  const response = await fetch('/api/chat', {
    method: 'DELETE',
    headers: {
      Accept: 'application/json',
      'Content-Type': 'application/json',
      ...authorization_headers(token),
    },
    body: JSON.stringify({ conversation_id }),
  });
  const response_body = await read_json_response(response);
  if (
    !is_record(response_body) ||
    response_body.status !== 'cancelled' ||
    response_body.conversation_id !== conversation_id
  ) {
    throw new Error('Gaia returned an invalid cancellation response.');
  }
}

export async function submit_gaia_feedback(
  token: string,
  conversation_id: string,
  interaction: number,
  feedback: GaiaResponseFeedback,
) {
  const response = await fetch('/api/feedback', {
    method: 'POST',
    headers: {
      Accept: 'application/json',
      'Content-Type': 'application/json',
      ...authorization_headers(token),
    },
    body: JSON.stringify({ conversation_id, interaction, ...feedback }),
  });
  const response_body = await read_json_response(response);
  if (
    !is_record(response_body) ||
    response_body.status !== 'recorded' ||
    response_body.conversation_id !== conversation_id ||
    response_body.interaction !== interaction ||
    response_body.rating !== feedback.rating
  ) {
    throw new Error('Gaia returned an invalid feedback response.');
  }
}

export async function download_gaia_attachment(
  token: string,
  conversation_id: string,
  attachment: GaiaAttachment,
) {
  const parameters = new URLSearchParams({
    conversation_id,
    file_id: attachment.file_id,
  });
  const response = await fetch(`/api/files?${parameters}`, {
    cache: 'no-store',
    headers: {
      Accept: attachment.media_type,
      ...authorization_headers(token),
    },
    signal: AbortSignal.timeout(2 * 60 * 1_000),
  });
  if (!response.ok) {
    throw await error_from_response(response);
  }
  const media_type = response.headers
    .get('Content-Type')
    ?.split(';', 1)[0]
    .trim()
    .toLowerCase();
  if (media_type !== attachment.media_type) {
    throw new Error('Gaia returned an unsupported file format.');
  }
  const file = await response.blob();
  if (file.size !== attachment.size_bytes) {
    throw new Error('The file download was incomplete. Try downloading again.');
  }

  const object_url = URL.createObjectURL(file);
  const link = document.createElement('a');
  link.href = object_url;
  link.download = attachment.filename;
  link.hidden = true;
  document.body.appendChild(link);
  try {
    link.click();
  } finally {
    link.remove();
    // Browsers may start consuming the object URL after the click returns.
    globalThis.setTimeout(() => URL.revokeObjectURL(object_url), 30_000);
  }
}
