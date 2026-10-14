'use client';

import {
  ArrowUp,
  LockKeyhole,
  LoaderCircle,
  MessageSquarePlus,
  PanelLeft,
  Search,
  Square,
  SquarePen,
} from 'lucide-react';
import Image from 'next/image';
import {
  type KeyboardEvent,
  type SubmitEvent,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from 'react';

import { GaiaLoadingMessage, GaiaMessage } from '@/components/gaia-response';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { useGaiaSession } from '@/hooks/use-gaia-session';
import {
  cancel_gaia_message,
  download_gaia_attachment,
  GaiaRequestCancelled,
  GaiaRequestError,
  type GaiaAttachment,
  type GaiaProgressEvent,
  type GaiaResponseFeedback,
  type GaiaResponseMetrics,
  type GaiaVerificationStep,
  send_gaia_message,
  submit_gaia_feedback,
} from '@/lib/gaia-api';

const chat_timeout_ms = 10 * 60 * 1000;

interface UserMessage {
  id: string;
  role: 'user';
  content: string;
}

interface AssistantMessage {
  id: string;
  role: 'assistant';
  conversation_id: string;
  interaction: number;
  content: string;
  activity: GaiaProgressEvent[];
  attachments: GaiaAttachment[];
  metrics: GaiaResponseMetrics;
  verification: GaiaVerificationStep[];
  returned_at: string;
  feedback?: GaiaResponseFeedback;
}

interface ErrorMessage {
  id: string;
  role: 'error';
  content: string;
}

interface StoppedMessage {
  id: string;
  role: 'stopped';
}

type ChatMessage =
  | UserMessage
  | AssistantMessage
  | ErrorMessage
  | StoppedMessage;

type RequestStatus = 'idle' | 'running' | 'stopping';
type StopReason = 'user' | 'timeout';

interface PendingRequestError {
  message: ErrorMessage;
  expires_conversation: boolean;
}

interface ActiveRequest {
  controller: AbortController;
  conversation_local_id: string;
  server_id: string | null;
  stop_reason: StopReason | null;
  cancellation_sent: boolean;
  pending_error: PendingRequestError | null;
  timeout_id: ReturnType<typeof setTimeout> | null;
}

interface Conversation {
  local_id: string;
  server_id: string | null;
  title: string;
  messages: ChatMessage[];
  is_expired: boolean;
}

function local_id() {
  return globalThis.crypto.randomUUID();
}

function conversation_title(message: string) {
  return message.length <= 54 ? message : `${message.slice(0, 51)}…`;
}

function MuseumWordmark({
  className,
  decorative = false,
}: {
  className?: string;
  decorative?: boolean;
}) {
  return (
    <Image
      className={className}
      src="/assets/beaty_background_transparent.png"
      alt={decorative ? '' : 'Beaty Biodiversity Museum'}
      width={1296}
      height={207}
      priority
    />
  );
}

export default function Home() {
  const [login_username, set_login_username] = useState('');
  const [login_password, set_login_password] = useState('');
  const [is_sidebar_collapsed, set_is_sidebar_collapsed] = useState(false);
  const [conversations, set_conversations] = useState<Conversation[]>([]);
  const [active_conversation_local_id, set_active_conversation_local_id] =
    useState<string | null>(null);
  const [draft, set_draft] = useState('');
  const [request_status, set_request_status] = useState<RequestStatus>('idle');
  const [request_activity, set_request_activity] = useState<
    GaiaProgressEvent[]
  >([]);
  const [revealing_message_id, set_revealing_message_id] = useState<
    string | null
  >(null);
  const active_request = useRef<ActiveRequest | null>(null);
  const thread_ref = useRef<HTMLDivElement | null>(null);
  const textarea_ref = useRef<HTMLTextAreaElement | null>(null);

  const active_conversation = useMemo(
    () =>
      conversations.find(
        (conversation) =>
          conversation.local_id === active_conversation_local_id,
      ) ?? null,
    [active_conversation_local_id, conversations],
  );
  const messages = active_conversation?.messages ?? [];
  const active_conversation_is_expired =
    active_conversation?.is_expired ?? false;
  const is_loading = request_status !== 'idle';
  const is_stopping = request_status === 'stopping';

  const clear_chat = useCallback(() => {
    const request = active_request.current;
    request?.controller.abort();
    if (request && request.timeout_id !== null) {
      globalThis.clearTimeout(request.timeout_id);
    }
    active_request.current = null;
    set_conversations([]);
    set_active_conversation_local_id(null);
    set_draft('');
    set_request_status('idle');
    set_request_activity([]);
    set_revealing_message_id(null);
    set_login_password('');
  }, []);

  const { authentication, login_error, is_signing_in, sign_in, end_session } =
    useGaiaSession(clear_chat);
  const session_token = authentication.session?.token ?? null;

  useEffect(() => {
    const thread = thread_ref.current;
    if (thread) {
      thread.scrollTo({ top: thread.scrollHeight, behavior: 'smooth' });
    }
  }, [active_conversation_local_id, is_loading, messages.length]);

  function update_conversation(
    conversation_local_id: string,
    update: (conversation: Conversation) => Conversation,
  ) {
    set_conversations((current) =>
      current.map((conversation) =>
        conversation.local_id === conversation_local_id
          ? update(conversation)
          : conversation,
      ),
    );
  }

  function append_conversation_error(
    conversation_local_id: string,
    message: ErrorMessage,
    expires_conversation: boolean,
  ) {
    update_conversation(conversation_local_id, (conversation) => ({
      ...conversation,
      server_id: expires_conversation ? null : conversation.server_id,
      messages: [...conversation.messages, message],
      is_expired: expires_conversation || conversation.is_expired,
    }));
  }

  function finish_active_request(request: ActiveRequest) {
    if (active_request.current !== request) {
      return;
    }
    if (request.timeout_id !== null) {
      globalThis.clearTimeout(request.timeout_id);
    }
    request.controller.abort();
    active_request.current = null;
    set_request_status('idle');
    set_request_activity([]);
    globalThis.setTimeout(() => textarea_ref.current?.focus(), 0);
  }

  function finish_cancelled_request(
    request: ActiveRequest,
    conversation_id: string,
  ) {
    if (active_request.current !== request) {
      return;
    }

    const terminal_message: ErrorMessage | StoppedMessage =
      request.stop_reason === 'timeout'
        ? {
            id: local_id(),
            role: 'error',
            content:
              'Gaia did not finish within ten minutes. Start a new chat and try again.',
          }
        : { id: local_id(), role: 'stopped' };
    update_conversation(request.conversation_local_id, (conversation) => ({
      ...conversation,
      server_id: conversation_id,
      messages: [...conversation.messages, terminal_message],
    }));
    finish_active_request(request);
  }

  async function send_cancellation(request: ActiveRequest, token: string) {
    if (
      active_request.current !== request ||
      request.server_id === null ||
      request.cancellation_sent
    ) {
      return;
    }

    request.cancellation_sent = true;
    const conversation_id = request.server_id;
    try {
      await cancel_gaia_message(token, conversation_id);
      finish_cancelled_request(request, conversation_id);
    } catch {
      if (active_request.current === request) {
        request.cancellation_sent = false;
        request.stop_reason = null;
        if (request.pending_error) {
          append_conversation_error(
            request.conversation_local_id,
            request.pending_error.message,
            request.pending_error.expires_conversation,
          );
          finish_active_request(request);
          return;
        }
        set_request_status('running');
      }
    }
  }

  function cancel_active_request() {
    const request = active_request.current;
    if (!request || !session_token || request_status !== 'running') {
      return;
    }
    request.stop_reason = 'user';
    set_request_status('stopping');
    void send_cancellation(request, session_token);
  }

  function start_new_chat() {
    if (active_request.current) {
      return;
    }
    set_active_conversation_local_id(null);
    set_revealing_message_id(null);
    set_draft('');
  }

  function keep_latest_response_visible() {
    const thread = thread_ref.current;
    if (!thread) {
      return;
    }
    const distance_from_bottom =
      thread.scrollHeight - thread.scrollTop - thread.clientHeight;
    if (distance_from_bottom < 160) {
      thread.scrollTop = thread.scrollHeight;
    }
  }

  function finish_response_reveal(message_id: string) {
    set_revealing_message_id((current) =>
      current === message_id ? null : current,
    );
  }

  function select_conversation(conversation_local_id: string) {
    if (is_loading) {
      return;
    }
    set_revealing_message_id(null);
    set_active_conversation_local_id(conversation_local_id);
  }

  async function save_response_feedback(
    conversation_local_id: string,
    message_id: string,
    conversation_id: string,
    interaction: number,
    feedback: GaiaResponseFeedback,
  ) {
    if (!session_token) {
      throw new Error('Your session ended. Sign in again.');
    }
    try {
      await submit_gaia_feedback(
        session_token,
        conversation_id,
        interaction,
        feedback,
      );
    } catch (error) {
      if (error instanceof GaiaRequestError && error.status === 401) {
        end_session('Your session ended. Sign in again.');
      }
      throw error;
    }
    update_conversation(conversation_local_id, (conversation) => ({
      ...conversation,
      messages: conversation.messages.map((message) =>
        message.id === message_id && message.role === 'assistant'
          ? {
              ...message,
              feedback: { ...feedback, reasons: [...feedback.reasons] },
            }
          : message,
      ),
    }));
  }

  async function download_attachment(
    conversation_id: string,
    attachment: GaiaAttachment,
  ) {
    if (!session_token) {
      throw new Error('Your session ended. Sign in again.');
    }
    try {
      await download_gaia_attachment(
        session_token,
        conversation_id,
        attachment,
      );
    } catch (error) {
      if (error instanceof GaiaRequestError && error.status === 401) {
        end_session('Your session ended. Sign in again.');
      }
      throw error;
    }
  }

  async function submit_login(event: SubmitEvent<HTMLFormElement>) {
    event.preventDefault();
    if (await sign_in(login_username, login_password)) {
      set_login_password('');
    }
  }

  async function submit_query(event: SubmitEvent<HTMLFormElement>) {
    event.preventDefault();
    const query = draft.trim();
    if (
      !query ||
      !session_token ||
      is_loading ||
      active_request.current !== null ||
      active_conversation_is_expired
    ) {
      return;
    }

    const existing_conversation = active_conversation;
    const conversation_local_id = existing_conversation?.local_id ?? local_id();
    const user_message: UserMessage = {
      id: local_id(),
      role: 'user',
      content: query,
    };

    if (existing_conversation) {
      update_conversation(conversation_local_id, (conversation) => ({
        ...conversation,
        messages: [...conversation.messages, user_message],
      }));
    } else {
      set_conversations((current) => [
        {
          local_id: conversation_local_id,
          server_id: null,
          title: conversation_title(query),
          messages: [user_message],
          is_expired: false,
        },
        ...current,
      ]);
    }
    set_active_conversation_local_id(conversation_local_id);
    set_revealing_message_id(null);
    set_draft('');
    set_request_status('running');
    set_request_activity([]);

    const controller = new AbortController();
    const request: ActiveRequest = {
      controller,
      conversation_local_id,
      server_id: existing_conversation?.server_id ?? null,
      stop_reason: null,
      cancellation_sent: false,
      pending_error: null,
      timeout_id: null,
    };
    active_request.current = request;
    const activity: GaiaProgressEvent[] = [];
    request.timeout_id = globalThis.setTimeout(() => {
      if (active_request.current !== request || request.stop_reason) {
        return;
      }
      request.stop_reason = 'timeout';
      set_request_status('stopping');
      void send_cancellation(request, session_token);
    }, chat_timeout_ms);

    try {
      const result = await send_gaia_message({
        token: session_token,
        message: query,
        conversation_id: existing_conversation?.server_id ?? null,
        signal: controller.signal,
        on_conversation_started(conversation_id) {
          if (active_request.current !== request) {
            return;
          }
          if (
            existing_conversation?.server_id &&
            conversation_id !== existing_conversation.server_id
          ) {
            throw new Error('Gaia returned a different conversation.');
          }
          request.server_id = conversation_id;
          update_conversation(conversation_local_id, (conversation) => ({
            ...conversation,
            server_id: conversation_id,
          }));
          if (request.stop_reason) {
            void send_cancellation(request, session_token);
          }
        },
        on_progress(progress) {
          if (active_request.current === request && !request.stop_reason) {
            activity.push(progress);
            set_request_activity([...activity]);
          }
        },
      });
      if (active_request.current !== request) {
        return;
      }
      request.stop_reason = null;

      const assistant_message: AssistantMessage = {
        id: local_id(),
        role: 'assistant',
        conversation_id: result.conversation_id,
        interaction: result.interaction,
        content: result.content,
        activity: [...activity],
        attachments: result.attachments,
        metrics: result.metrics,
        verification: result.verification,
        returned_at: new Date().toISOString(),
      };
      update_conversation(conversation_local_id, (conversation) => ({
        ...conversation,
        server_id: result.conversation_id,
        messages: [...conversation.messages, assistant_message],
        is_expired: false,
      }));
      set_revealing_message_id(assistant_message.id);
    } catch (error) {
      if (active_request.current !== request) {
        return;
      }

      if (error instanceof GaiaRequestCancelled) {
        finish_cancelled_request(request, error.conversation_id);
        return;
      }

      if (controller.signal.aborted) {
        return;
      }

      const is_session_ended =
        error instanceof GaiaRequestError && error.status === 401;
      if (is_session_ended) {
        end_session('Your session ended. Sign in again.');
        return;
      }

      const is_network_error = error instanceof TypeError;
      const is_expired_conversation =
        error instanceof GaiaRequestError && error.status === 404;
      const error_message: ErrorMessage = {
        id: local_id(),
        role: 'error',
        content: is_network_error
          ? 'Gaia’s local query service is unavailable. Start the Gaia web service and try again.'
          : error instanceof Error
            ? error.message
            : 'Gaia could not complete this request.',
      };
      if (request.stop_reason && request.cancellation_sent) {
        request.pending_error = {
          message: error_message,
          expires_conversation: is_expired_conversation,
        };
        return;
      }
      request.stop_reason = null;
      append_conversation_error(
        conversation_local_id,
        error_message,
        is_expired_conversation,
      );
    } finally {
      if (active_request.current === request && !request.stop_reason) {
        finish_active_request(request);
      }
    }
  }

  function handle_composer_key_down(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (
      event.key === 'Enter' &&
      !event.shiftKey &&
      !event.nativeEvent.isComposing
    ) {
      event.preventDefault();
      event.currentTarget.form?.requestSubmit();
    }
  }

  if (authentication.status !== 'signed_in') {
    return (
      <main className="signin-shell">
        <section className="signin-card" aria-labelledby="signin-title">
          <MuseumWordmark className="signin-brand" />
          {authentication.status === 'checking' ? (
            <output className="signin-checking">
              <LoaderCircle aria-hidden="true" />
              <span>Checking access…</span>
            </output>
          ) : (
            <>
              <div className="signin-heading">
                <p>Collections research</p>
                <h1 id="signin-title">Sign in to Gaia</h1>
                <span>Enter your username and the collection password.</span>
              </div>
              <form className="signin-form" onSubmit={submit_login}>
                <div className="signin-field">
                  <label htmlFor="gaia-username">Username</label>
                  <Input
                    autoComplete="username"
                    id="gaia-username"
                    maxLength={64}
                    onChange={(event) => set_login_username(event.target.value)}
                    placeholder="Username"
                    required
                    value={login_username}
                  />
                </div>
                <div className="signin-field">
                  <label htmlFor="gaia-password">Password</label>
                  <Input
                    aria-invalid={Boolean(login_error)}
                    autoComplete="current-password"
                    id="gaia-password"
                    onChange={(event) => set_login_password(event.target.value)}
                    placeholder="Password"
                    required
                    type="password"
                    value={login_password}
                  />
                </div>
                {login_error ? (
                  <p className="signin-error" role="alert">
                    {login_error}
                  </p>
                ) : null}
                <Button
                  className="signin-button"
                  disabled={
                    !login_username.trim() || !login_password || is_signing_in
                  }
                  type="submit"
                >
                  {is_signing_in ? (
                    <LoaderCircle aria-hidden="true" />
                  ) : (
                    <LockKeyhole aria-hidden="true" />
                  )}
                  {is_signing_in ? 'Signing in…' : 'Access Gaia'}
                </Button>
              </form>
            </>
          )}
        </section>
      </main>
    );
  }

  return (
    <main className="site-shell">
      <section
        className={
          is_sidebar_collapsed
            ? 'app-window sidebar-is-collapsed'
            : 'app-window'
        }
        aria-label="Beaty museum query interface"
      >
        <aside
          className={
            is_sidebar_collapsed
              ? 'query-sidebar is-collapsed'
              : 'query-sidebar'
          }
          id="query-sidebar"
        >
          <div className="sidebar-brand">
            {is_sidebar_collapsed ? (
              <button
                aria-label="Expand chat menu"
                className="sidebar-rail-logo"
                onClick={() => set_is_sidebar_collapsed(false)}
                type="button"
              >
                <Image
                  src="/assets/images.png"
                  alt=""
                  width={512}
                  height={512}
                  priority
                />
              </button>
            ) : (
              <>
                <MuseumWordmark className="sidebar-brand-logo" />
                <button
                  aria-controls="query-sidebar"
                  aria-expanded="true"
                  aria-label="Collapse chat menu"
                  className="sidebar-menu-toggle"
                  onClick={() => set_is_sidebar_collapsed(true)}
                  type="button"
                >
                  <PanelLeft aria-hidden="true" />
                </button>
              </>
            )}
          </div>

          {is_sidebar_collapsed ? null : (
            <>
              <button
                className="new-query-button"
                disabled={is_loading}
                onClick={start_new_chat}
                type="button"
              >
                <SquarePen aria-hidden="true" />
                <span>New Chat</span>
              </button>

              <nav className="query-navigation" aria-label="Chats">
                <div className="nav-label">Chats</div>
                {conversations.length ? (
                  <ul>
                    {conversations.map((conversation) => (
                      <li key={conversation.local_id}>
                        <button
                          aria-current={
                            conversation.local_id ===
                            active_conversation_local_id
                              ? 'page'
                              : undefined
                          }
                          className={
                            conversation.local_id ===
                            active_conversation_local_id
                              ? 'active'
                              : undefined
                          }
                          disabled={is_loading}
                          onClick={() =>
                            select_conversation(conversation.local_id)
                          }
                          type="button"
                        >
                          {conversation.title}
                        </button>
                      </li>
                    ))}
                  </ul>
                ) : (
                  <p className="chats-empty">Your chats will appear here.</p>
                )}
              </nav>
            </>
          )}
        </aside>

        <div className="query-workspace">
          <header className="workspace-header" aria-label="Gaia controls">
            {is_sidebar_collapsed && messages.length === 0 ? (
              <MuseumWordmark className="collapsed-workspace-brand" />
            ) : null}
          </header>

          <section
            className={
              messages.length === 0 ? 'conversation is-empty' : 'conversation'
            }
            aria-label="Gaia conversation"
          >
            {messages.length ? (
              <div className="museum-watermark" aria-hidden="true">
                <MuseumWordmark decorative />
              </div>
            ) : null}

            <div
              aria-live="polite"
              aria-relevant="additions text"
              className="thread"
              ref={thread_ref}
              role="log"
            >
              {messages.length === 0 ? (
                <section className="empty-state">
                  <h2>
                    Good to see you,{' '}
                    <span>{authentication.session.username}</span>
                  </h2>
                </section>
              ) : (
                messages.map((message) => {
                  if (message.role === 'user') {
                    return (
                      <article className="user-message" key={message.id}>
                        <p>{message.content}</p>
                      </article>
                    );
                  }
                  if (message.role === 'stopped') {
                    return (
                      <p className="stopped-message" key={message.id}>
                        Stopped thinking
                      </p>
                    );
                  }
                  return (
                    <GaiaMessage
                      attachments={
                        message.role === 'assistant'
                          ? message.attachments
                          : undefined
                      }
                      activity={
                        message.role === 'assistant'
                          ? message.activity
                          : undefined
                      }
                      content={message.content}
                      feedback={
                        message.role === 'assistant'
                          ? message.feedback
                          : undefined
                      }
                      key={message.id}
                      metrics={
                        message.role === 'assistant'
                          ? message.metrics
                          : undefined
                      }
                      verification={
                        message.role === 'assistant'
                          ? message.verification
                          : undefined
                      }
                      on_download_attachment={
                        message.role === 'assistant'
                          ? (attachment) =>
                              download_attachment(
                                message.conversation_id,
                                attachment,
                              )
                          : undefined
                      }
                      on_reveal_complete={() =>
                        finish_response_reveal(message.id)
                      }
                      on_reveal_progress={keep_latest_response_visible}
                      on_submit_feedback={
                        message.role === 'assistant' && active_conversation
                          ? (feedback) =>
                              save_response_feedback(
                                active_conversation.local_id,
                                message.id,
                                message.conversation_id,
                                message.interaction,
                                feedback,
                              )
                          : undefined
                      }
                      returned_at={
                        message.role === 'assistant'
                          ? message.returned_at
                          : undefined
                      }
                      reveal={revealing_message_id === message.id}
                      role={message.role}
                    />
                  );
                })
              )}

              {is_loading ? (
                <GaiaLoadingMessage
                  activity={request_activity}
                  is_stopping={is_stopping}
                />
              ) : null}
            </div>

            <div className="composer-wrap">
              <form className="composer" onSubmit={submit_query}>
                <Search className="composer-search" aria-hidden="true" />
                <Textarea
                  aria-label="Ask a question about the collections"
                  disabled={is_loading || active_conversation_is_expired}
                  maxLength={4000}
                  onChange={(event) => set_draft(event.target.value)}
                  onKeyDown={handle_composer_key_down}
                  placeholder={
                    active_conversation_is_expired
                      ? 'This conversation has ended. Start a new chat.'
                      : 'Ask about a species, place, collector, or date…'
                  }
                  ref={textarea_ref}
                  rows={1}
                  value={draft}
                />
                <div className="composer-actions">
                  <Button
                    aria-label={
                      active_conversation_is_expired
                        ? 'Start a new chat'
                        : is_stopping
                          ? 'Stopping query'
                          : is_loading
                            ? 'Stop query'
                            : 'Send query'
                    }
                    className="send-button"
                    data-loading={is_loading}
                    disabled={
                      is_stopping ||
                      (!active_conversation_is_expired &&
                        !is_loading &&
                        !draft.trim())
                    }
                    onClick={
                      active_conversation_is_expired
                        ? start_new_chat
                        : is_loading
                          ? cancel_active_request
                          : undefined
                    }
                    type={
                      active_conversation_is_expired || is_loading
                        ? 'button'
                        : 'submit'
                    }
                  >
                    {active_conversation_is_expired ? (
                      <MessageSquarePlus />
                    ) : is_loading ? (
                      <Square />
                    ) : (
                      <ArrowUp />
                    )}
                  </Button>
                </div>
              </form>
            </div>
          </section>
        </div>
      </section>
    </main>
  );
}
