'use client';

import {
  AlertCircle,
  Check,
  ChevronDown,
  Copy,
  Database,
  Plus,
  Search,
  ThumbsDown,
  ThumbsUp,
  X,
} from 'lucide-react';
import Image from 'next/image';
import ReactMarkdown, { type Components } from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { type SubmitEvent, useEffect, useId, useRef, useState } from 'react';

import { GaiaAttachments } from '@/components/gaia-attachments';
import { useResponseReveal } from '@/hooks/use-response-reveal';
import { context_compacted_notice } from '@/lib/gaia-api';

import type {
  GaiaAttachment,
  GaiaProgressActivity,
  GaiaProgressEvent,
  GaiaResponseFeedback,
  GaiaResponseMetrics,
  GaiaResponseRating,
  GaiaJsonValue,
  GaiaVerificationStep,
} from '@/lib/gaia-api';

const maximum_activity_steps = 5;

const feedback_options: Record<GaiaResponseRating, readonly string[]> = {
  good: [
    'Accurate and complete',
    'Followed my instructions',
    'Clear and useful',
    'Relevant',
    'Fast and smooth',
    'Other',
  ],
  bad: [
    'Incorrect or incomplete',
    "Didn't follow my instructions",
    'Off track / wrong scope',
    'Lost context',
    'Slow or buggy',
    'Safety or legal concern',
    'Other',
  ],
};

const markdown_components: Components = {
  img({ alt }) {
    // Generated content must never cause an automatic network request.
    return alt ? <span>{alt}</span> : null;
  },
  a({ children, href }) {
    if (!href) {
      return <span>{children}</span>;
    }
    return (
      <a
        href={href}
        rel="noopener noreferrer"
        target="_blank"
        referrerPolicy="no-referrer"
      >
        {children}
      </a>
    );
  },
  table({ node: _node, ...table_properties }) {
    return (
      <section
        aria-label="Scrollable collection records"
        className="answer-table-scroll"
      >
        <table {...table_properties} />
      </section>
    );
  },
};

function markdown_url(url: string, key: string) {
  if (key !== 'href' || !/^https?:\/\//iu.test(url)) {
    return undefined;
  }
  try {
    const target = new URL(url);
    const pathname = decodeURIComponent(target.pathname);
    // Workbook downloads come from validated attachments, never generated links.
    if (
      target.username ||
      target.password ||
      /(?:^|\/)api\/files(?:\/|$)/iu.test(pathname) ||
      /\.(?:xlsx?|csv|tsv)(?:$|\/)/iu.test(pathname)
    ) {
      return undefined;
    }
    return target.href;
  } catch {
    return undefined;
  }
}

function AssistantMark() {
  return (
    <div className="assistant-mark" aria-hidden="true">
      <Image src="/assets/images.png" alt="" width={512} height={512} />
    </div>
  );
}

function progress_label(progress: GaiaProgressEvent | null) {
  if (!progress) {
    return 'Thinking';
  }
  if (progress.phase === 'preparing_answer') {
    return 'Preparing the answer';
  }
  if (progress.phase === 'context_compacting') {
    return 'Compacting earlier conversation';
  }
  if (progress.phase === 'context_compacted') {
    return 'Continuing with the conversation summary';
  }
  if (progress.phase === 'activity') {
    return progress.activities.at(-1)?.label ?? 'Reviewing catalogue results';
  }

  if (progress.phase === 'searching') {
    return 'Searching collection records';
  }
  if (progress.phase === 'reviewing') {
    return 'Reviewing catalogue fields';
  }
  if (progress.phase === 'preparing_file') {
    return 'Preparing the workbook';
  }
  return 'Thinking';
}

function activity_steps(activity: readonly GaiaProgressEvent[]) {
  const steps_by_id = new Map<string, GaiaProgressActivity>();
  for (const progress of activity) {
    if (progress.phase !== 'activity') {
      continue;
    }
    for (const trace of progress.activities) {
      steps_by_id.set(trace.id, trace);
    }
  }
  return Array.from(steps_by_id.values()).slice(0, maximum_activity_steps);
}

function ActivityIcon({ activity }: { activity: GaiaProgressActivity }) {
  if (activity.status === 'failed') {
    return <AlertCircle aria-hidden="true" />;
  }
  if (activity.kind === 'catalogue') {
    return <Database aria-hidden="true" />;
  }
  if (activity.kind === 'query') {
    return <Search aria-hidden="true" />;
  }
  return <Check aria-hidden="true" />;
}

function ActivityDisclosure({
  activity,
  is_live = false,
  label,
}: {
  activity: readonly GaiaProgressEvent[];
  is_live?: boolean;
  label: string;
}) {
  const [is_open, set_is_open] = useState(false);
  const summary_id = useId();
  const steps = activity_steps(activity);

  return (
    <div className="activity-disclosure">
      <button
        aria-controls={summary_id}
        aria-expanded={is_open}
        className="activity-toggle"
        onClick={() => set_is_open((current) => !current)}
        type="button"
      >
        {is_live ? (
          <span className="request-progress-dots" aria-hidden="true">
            <span />
            <span />
            <span />
          </span>
        ) : null}
        <span
          aria-atomic={is_live ? 'true' : undefined}
          aria-live={is_live ? 'polite' : undefined}
          className="request-progress-phase"
          key={label}
        >
          {label}
        </span>
        <ChevronDown aria-hidden="true" data-open={is_open} />
      </button>
      {activity.some((event) => event.phase === 'context_compacted') ? (
        <output className="context-compaction-notice">
          {context_compacted_notice}
        </output>
      ) : null}
      {is_open ? (
        <section
          aria-label="Gaia activity summary"
          className="activity-summary"
          id={summary_id}
        >
          {steps.length ? (
            <ol>
              {steps.map((step) => (
                <li key={step.id}>
                  <span
                    className="activity-step-icon"
                    data-status={step.status}
                  >
                    <ActivityIcon activity={step} />
                  </span>
                  <div>
                    <strong>{step.label}</strong>
                    <p>{step.detail}</p>
                    {step.interpretation ? (
                      <p className="activity-interpretation">
                        <strong>Interpretation:</strong> {step.interpretation}
                      </p>
                    ) : null}
                  </div>
                </li>
              ))}
            </ol>
          ) : (
            <p className="activity-empty">
              {is_live
                ? 'Waiting for the first catalogue result…'
                : 'No catalogue lookup was needed for this answer.'}
            </p>
          )}
        </section>
      ) : null}
    </div>
  );
}

function RollingMarkdown({
  animate,
  content,
  on_complete,
  on_progress,
}: {
  animate: boolean;
  content: string;
  on_complete: () => void;
  on_progress: () => void;
}) {
  const { visible_content, is_revealing } = useResponseReveal({
    animate,
    content,
    on_complete,
    on_progress,
  });

  return (
    <div aria-busy={is_revealing} className="answer-body">
      <ReactMarkdown
        components={markdown_components}
        remarkPlugins={[remarkGfm]}
        skipHtml
        urlTransform={markdown_url}
      >
        {visible_content}
      </ReactMarkdown>
    </div>
  );
}

function VerificationDisclosure({
  verification,
}: {
  verification: readonly GaiaVerificationStep[];
}) {
  if (!verification.length) {
    return null;
  }

  return (
    <details className="verification-disclosure">
      <summary>Verify this answer</summary>
      <p className="verification-replay-help">
        Copy a call and run it in a terminal from your Gaia repository, using the
        same database configuration.
      </p>
      <ol>
        {verification.map((step) => (
          <li key={step.sequence}>
            <h4>Tool call {step.sequence}</h4>
            {step.status === 'completed' && step.capability !== 'generate_excel' ? (
              <ReplayCommand step={step} />
            ) : null}
            <section className="verification-tool">
              <strong>Tool</strong>
              <div>
                <code>{step.capability}</code>
                <span data-status={step.status}>{step.status}</span>
              </div>
            </section>
            {step.interpretation ? (
              <section className="verification-purpose">
                <strong>Interpretation</strong>
                <p>{step.interpretation}</p>
              </section>
            ) : null}
            <section className="verification-purpose">
              <strong>Purpose</strong>
              <p>{step.purpose}</p>
            </section>
            <section>
              <VerificationData title="Arguments" value={step.arguments} />
            </section>
            <section>
              <VerificationData title="Evidence" value={step.evidence} />
            </section>
          </li>
        ))}
      </ol>
    </details>
  );
}

function replay_command(step: GaiaVerificationStep): string {
  const payload = JSON.stringify(
    { capability: step.capability, arguments: step.arguments },
    null,
    2,
  );
  let delimiter = 'GAIA_REPLAY_JSON';
  while (payload.split('\n').includes(delimiter)) {
    delimiter += '_END';
  }
  return `.venv/bin/python -m src.gaia_agent.replay <<'${delimiter}'\n${payload}\n${delimiter}`;
}

function ReplayCommand({ step }: { step: GaiaVerificationStep }) {
  const [copy_status, set_copy_status] = useState<'idle' | 'copied' | 'failed'>(
    'idle',
  );

  async function copy_command() {
    try {
      await globalThis.navigator.clipboard.writeText(replay_command(step));
      set_copy_status('copied');
    } catch {
      set_copy_status('failed');
    }
  }

  return (
    <button
      className="verification-replay"
      type="button"
      onClick={() => void copy_command()}
      title="Copy a command to rerun this call from the Gaia repository"
    >
      {copy_status === 'copied' ? <Check aria-hidden="true" /> : <Copy aria-hidden="true" />}
      {copy_status === 'copied'
        ? 'Copied terminal command'
        : copy_status === 'failed'
          ? 'Copy failed'
          : 'Copy terminal command'}
    </button>
  );
}

function verification_label(key: string) {
  const spaced = key
    .replace(/([a-z0-9])([A-Z])/gu, '$1 $2')
    .replace(/[_-]+/gu, ' ')
    .trim();
  return spaced
    ? `${spaced.charAt(0).toLocaleUpperCase()}${spaced.slice(1)}`
    : key;
}

function verification_value_kind(key: string) {
  if (/(?:^|_)(?:count|total|returned|shown|omitted)$/iu.test(key)) {
    return key.toLocaleLowerCase().endsWith('omitted') ? 'omitted' : 'count';
  }
  return undefined;
}

function VerificationScalar({
  field_name = '',
  value,
}: {
  field_name?: string;
  value: null | boolean | number | string;
}) {
  if (value === null) {
    return <span className="verification-scalar is-empty">Not provided</span>;
  }
  if (typeof value === 'boolean') {
    return (
      <span className="verification-scalar" data-value-type="boolean">
        {value ? 'Yes' : 'No'}
      </span>
    );
  }
  return (
    <span
      className="verification-scalar"
      data-value-kind={verification_value_kind(field_name)}
    >
      {String(value)}
    </span>
  );
}

function VerificationValue({
  field_name,
  value,
}: {
  field_name?: string;
  value: GaiaJsonValue;
}) {
  if (Array.isArray(value)) {
    if (!value.length) {
      return <span className="verification-empty">No items</span>;
    }
    return (
      <ol className="verification-array">
        {value.map((item, index) => (
          <li key={index}>
            <VerificationValue value={item} />
          </li>
        ))}
      </ol>
    );
  }
  if (typeof value === 'object' && value !== null) {
    const fields = Object.entries(value);
    if (!fields.length) {
      return <span className="verification-empty">No fields</span>;
    }
    return (
      <dl className="verification-object">
        {fields.map(([key, item]) => (
          <div key={key}>
            <dt>{verification_label(key)}</dt>
            <dd>
              <VerificationValue field_name={key} value={item} />
            </dd>
          </div>
        ))}
      </dl>
    );
  }
  return <VerificationScalar field_name={field_name} value={value} />;
}

function VerificationData({
  title,
  value,
}: {
  title: string;
  value: Record<string, GaiaJsonValue>;
}) {
  return (
    <div className="verification-data">
      <strong>{title}</strong>
      <div className="verification-structured">
        <VerificationValue value={value} />
      </div>
      <details className="verification-json">
        <summary>View {title.toLocaleLowerCase()} JSON</summary>
        <pre>{JSON.stringify(value, null, 2)}</pre>
      </details>
    </div>
  );
}

function ResponseActions({
  content,
  feedback,
  returned_at,
  on_submit_feedback,
}: {
  content: string;
  feedback?: GaiaResponseFeedback;
  returned_at: string;
  on_submit_feedback?: (feedback: GaiaResponseFeedback) => Promise<void>;
}) {
  const [copy_status, set_copy_status] = useState<'idle' | 'copied' | 'failed'>(
    'idle',
  );
  const [rating_menu_is_open, set_rating_menu_is_open] = useState(false);
  const [draft_rating, set_draft_rating] = useState<GaiaResponseRating | null>(
    null,
  );
  const [feedback_dialog_is_open, set_feedback_dialog_is_open] =
    useState(false);
  const [selected_feedback_options, set_selected_feedback_options] = useState<
    string[]
  >([]);
  const [feedback_details, set_feedback_details] = useState('');
  const [is_submitting_feedback, set_is_submitting_feedback] = useState(false);
  const [feedback_error, set_feedback_error] = useState<string | null>(null);
  const copy_reset_timeout = useRef<ReturnType<typeof setTimeout> | null>(null);
  const rating_menu_ref = useRef<HTMLDivElement | null>(null);
  const feedback_dialog_ref = useRef<HTMLDialogElement | null>(null);
  const feedback_title_id = useId();

  useEffect(
    () => () => {
      if (copy_reset_timeout.current) {
        globalThis.clearTimeout(copy_reset_timeout.current);
      }
    },
    [],
  );

  useEffect(() => {
    if (!rating_menu_is_open) {
      return;
    }

    function close_rating_menu_on_pointer_down(event: PointerEvent) {
      if (!rating_menu_ref.current?.contains(event.target as Node)) {
        set_rating_menu_is_open(false);
      }
    }

    function close_rating_menu_on_escape(event: KeyboardEvent) {
      if (event.key === 'Escape') {
        set_rating_menu_is_open(false);
      }
    }

    globalThis.document.addEventListener(
      'pointerdown',
      close_rating_menu_on_pointer_down,
    );
    globalThis.document.addEventListener(
      'keydown',
      close_rating_menu_on_escape,
    );
    return () => {
      globalThis.document.removeEventListener(
        'pointerdown',
        close_rating_menu_on_pointer_down,
      );
      globalThis.document.removeEventListener(
        'keydown',
        close_rating_menu_on_escape,
      );
    };
  }, [rating_menu_is_open]);

  useEffect(() => {
    const dialog = feedback_dialog_ref.current;
    if (!dialog) {
      return;
    }
    if (feedback_dialog_is_open && !dialog.open) {
      dialog.showModal();
    } else if (!feedback_dialog_is_open && dialog.open) {
      dialog.close();
    }
  }, [feedback_dialog_is_open]);

  async function copy_response() {
    if (copy_reset_timeout.current) {
      globalThis.clearTimeout(copy_reset_timeout.current);
    }
    try {
      await globalThis.navigator.clipboard.writeText(content);
      set_copy_status('copied');
    } catch {
      set_copy_status('failed');
    }
    copy_reset_timeout.current = globalThis.setTimeout(
      () => set_copy_status('idle'),
      1_800,
    );
  }

  function open_feedback_dialog(next_rating: GaiaResponseRating) {
    const existing_feedback =
      feedback?.rating === next_rating ? feedback : undefined;
    set_draft_rating(next_rating);
    set_selected_feedback_options(existing_feedback?.reasons ?? []);
    set_feedback_details(existing_feedback?.details ?? '');
    set_feedback_error(null);
    set_rating_menu_is_open(false);
    set_feedback_dialog_is_open(true);
  }

  function close_feedback_dialog() {
    if (!is_submitting_feedback) {
      set_feedback_dialog_is_open(false);
    }
  }

  function toggle_feedback_option(option: string) {
    set_selected_feedback_options((current) =>
      current.includes(option)
        ? current.filter((current_option) => current_option !== option)
        : [...current, option],
    );
  }

  async function submit_feedback(event: SubmitEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!draft_rating || !on_submit_feedback || is_submitting_feedback) {
      return;
    }
    set_is_submitting_feedback(true);
    set_feedback_error(null);
    try {
      await on_submit_feedback({
        rating: draft_rating,
        reasons: selected_feedback_options,
        details: feedback_details.trim(),
      });
      set_feedback_dialog_is_open(false);
    } catch (error) {
      set_feedback_error(
        error instanceof Error
          ? error.message
          : 'Gaia could not save this feedback.',
      );
    } finally {
      set_is_submitting_feedback(false);
    }
  }

  const returned_time = new Intl.DateTimeFormat(undefined, {
    hour: 'numeric',
    minute: '2-digit',
  }).format(new Date(returned_at));
  const can_submit_feedback = draft_rating !== null && !is_submitting_feedback;
  const submitted_rating = feedback?.rating;

  return (
    <>
      <footer className="answer-meta">
        <div className="answer-actions">
          <button
            className="answer-action"
            data-state={copy_status}
            onClick={() => void copy_response()}
            title={
              copy_status === 'failed'
                ? 'Could not copy response'
                : copy_status === 'copied'
                  ? 'Response copied'
                  : 'Copy response'
            }
            type="button"
          >
            {copy_status === 'copied' ? (
              <Check aria-hidden="true" />
            ) : (
              <Copy aria-hidden="true" />
            )}
            <span>
              {copy_status === 'copied'
                ? 'Copied'
                : copy_status === 'failed'
                  ? 'Copy failed'
                  : 'Copy'}
            </span>
          </button>
          <div className="rating-control" ref={rating_menu_ref}>
            <button
              aria-expanded={rating_menu_is_open}
              aria-haspopup="menu"
              className="answer-action"
              data-rated={submitted_rating ? 'true' : 'false'}
              onClick={() => set_rating_menu_is_open((current) => !current)}
              title={
                submitted_rating
                  ? `${submitted_rating === 'good' ? 'Good' : 'Bad'} response feedback saved`
                  : 'Rate response'
              }
              type="button"
            >
              {submitted_rating === 'bad' ? (
                <ThumbsDown aria-hidden="true" />
              ) : (
                <ThumbsUp aria-hidden="true" />
              )}
              <span>
                {submitted_rating
                  ? `${submitted_rating === 'good' ? 'Good' : 'Bad'} response`
                  : 'Rate response'}
              </span>
            </button>
            {rating_menu_is_open ? (
              <div
                aria-label="Rate response"
                className="rating-menu"
                role="menu"
              >
                <button
                  onClick={() => open_feedback_dialog('good')}
                  role="menuitem"
                  type="button"
                >
                  <ThumbsUp aria-hidden="true" />
                  <span>Good response</span>
                </button>
                <button
                  onClick={() => open_feedback_dialog('bad')}
                  role="menuitem"
                  type="button"
                >
                  <ThumbsDown aria-hidden="true" />
                  <span>Bad response</span>
                </button>
              </div>
            ) : null}
          </div>
          <time
            className="answer-returned-at"
            dateTime={returned_at}
            title={`Answer returned at ${returned_time}`}
          >
            {returned_time}
          </time>
        </div>
      </footer>

      <dialog
        aria-labelledby={feedback_title_id}
        className="feedback-dialog"
        onCancel={(event) => {
          event.preventDefault();
          close_feedback_dialog();
        }}
        onClose={close_feedback_dialog}
        ref={feedback_dialog_ref}
      >
        <form className="feedback-form" onSubmit={submit_feedback}>
          <header>
            <h2 id={feedback_title_id}>Share feedback</h2>
            <button
              aria-label="Close feedback"
              className="feedback-close"
              disabled={is_submitting_feedback}
              onClick={close_feedback_dialog}
              type="button"
            >
              <X aria-hidden="true" />
            </button>
          </header>

          <fieldset
            aria-label={`${draft_rating === 'good' ? 'Good' : 'Bad'} response feedback reasons`}
            className="feedback-options"
            disabled={is_submitting_feedback}
          >
            {(draft_rating ? feedback_options[draft_rating] : []).map(
              (option) => (
                <button
                  aria-pressed={selected_feedback_options.includes(option)}
                  key={option}
                  onClick={() => toggle_feedback_option(option)}
                  type="button"
                >
                  <Plus aria-hidden="true" />
                  <span>{option}</span>
                </button>
              ),
            )}
          </fieldset>

          <textarea
            aria-label="Feedback details"
            disabled={is_submitting_feedback}
            maxLength={2000}
            onChange={(event) => set_feedback_details(event.target.value)}
            placeholder="Share details (optional)"
            rows={6}
            value={feedback_details}
          />

          <p
            data-error={feedback_error !== null}
            role={feedback_error !== null ? 'alert' : undefined}
          >
            {feedback_error || 'Your feedback helps improve Gaia.'}
          </p>

          <button
            className="feedback-submit"
            disabled={!can_submit_feedback}
            type="submit"
          >
            {is_submitting_feedback ? 'Saving…' : 'Submit'}
          </button>
        </form>
      </dialog>
    </>
  );
}

export function GaiaMessage({
  activity = [],
  attachments = [],
  role,
  content,
  metrics,
  feedback,
  returned_at,
  reveal,
  on_download_attachment,
  on_submit_feedback,
  on_reveal_complete,
  on_reveal_progress,
  verification = [],
}: {
  activity?: readonly GaiaProgressEvent[];
  attachments?: readonly GaiaAttachment[];
  role: 'assistant' | 'error';
  content: string;
  metrics?: GaiaResponseMetrics;
  feedback?: GaiaResponseFeedback;
  returned_at?: string;
  reveal: boolean;
  on_download_attachment?: (attachment: GaiaAttachment) => Promise<void>;
  on_submit_feedback?: (feedback: GaiaResponseFeedback) => Promise<void>;
  on_reveal_complete: () => void;
  on_reveal_progress: () => void;
  verification?: readonly GaiaVerificationStep[];
}) {
  const is_error = role === 'error';

  return (
    <article className={`assistant-message ${is_error ? 'has-error' : ''}`}>
      <AssistantMark />
      <div className="assistant-content">
        <header>
          <div>
            <strong>Gaia</strong>
            {is_error ? (
              <span>Request issue</span>
            ) : (
              <ActivityDisclosure activity={activity} label="Answer complete" />
            )}
          </div>
        </header>
        <div className="assistant-answer">
          {is_error ? (
            <AlertCircle className="answer-alert" aria-hidden="true" />
          ) : null}
          <RollingMarkdown
            animate={reveal}
            content={content}
            on_complete={on_reveal_complete}
            on_progress={on_reveal_progress}
          />
          {attachments.length && on_download_attachment ? (
            <GaiaAttachments
              attachments={attachments}
              on_download={on_download_attachment}
            />
          ) : null}
          <VerificationDisclosure verification={verification} />
          {metrics && returned_at ? (
            <ResponseActions
              content={content}
              feedback={feedback}
              returned_at={returned_at}
              on_submit_feedback={on_submit_feedback}
            />
          ) : null}
        </div>
      </div>
    </article>
  );
}

export function GaiaLoadingMessage({
  activity,
  is_stopping,
}: {
  activity: readonly GaiaProgressEvent[];
  is_stopping: boolean;
}) {
  return (
    <article
      aria-label={
        is_stopping ? 'Gaia is stopping' : 'Gaia is preparing a response'
      }
      className="assistant-message loading-message"
    >
      <AssistantMark />
      <div className="assistant-content">
        <header>
          <div>
            <strong>Gaia</strong>
            <ActivityDisclosure
              activity={activity}
              is_live
              label={
                is_stopping
                  ? 'Stopping'
                  : progress_label(activity.at(-1) ?? null)
              }
            />
          </div>
        </header>
      </div>
    </article>
  );
}
