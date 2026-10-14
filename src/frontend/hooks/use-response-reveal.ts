'use client';

import { useEffect, useEffectEvent, useRef, useState } from 'react';

const response_reveal_frame_ms = 32;

function next_reveal_boundary(content: string, approximate_end: number) {
  if (approximate_end >= content.length) {
    return content.length;
  }
  const boundary_limit = Math.min(content.length, approximate_end + 24);
  for (let index = approximate_end; index < boundary_limit; index += 1) {
    if (/\s/.test(content[index])) {
      return index + 1;
    }
  }
  return approximate_end;
}

export function useResponseReveal({
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
  const initial_length = animate ? 0 : content.length;
  const [visible_length, set_visible_length] = useState(initial_length);
  const [reduce_motion, set_reduce_motion] = useState(
    () =>
      globalThis.matchMedia?.('(prefers-reduced-motion: reduce)').matches ??
      false,
  );
  const visible_length_ref = useRef(initial_length);
  const completed_content_ref = useRef('');
  const complete = useEffectEvent(on_complete);
  const progress = useEffectEvent(on_progress);

  useEffect(() => {
    const media_query = globalThis.matchMedia?.(
      '(prefers-reduced-motion: reduce)',
    );
    if (!media_query) {
      return;
    }
    const handle_change = (event: MediaQueryListEvent) =>
      set_reduce_motion(event.matches);
    media_query.addEventListener('change', handle_change);
    return () => media_query.removeEventListener('change', handle_change);
  }, []);

  useEffect(() => {
    if (!animate || reduce_motion) {
      visible_length_ref.current = content.length;
      if (animate && completed_content_ref.current !== content) {
        completed_content_ref.current = content;
        globalThis.queueMicrotask(() => complete());
      }
      return;
    }

    if (visible_length_ref.current > content.length) {
      visible_length_ref.current = content.length;
    }
    if (visible_length_ref.current === content.length) {
      if (completed_content_ref.current !== content) {
        completed_content_ref.current = content;
        globalThis.queueMicrotask(() => complete());
      }
      return;
    }

    const remaining_characters = content.length - visible_length_ref.current;
    const reveal_duration_ms = Math.min(
      7_000,
      Math.max(700, remaining_characters * 7),
    );
    const reveal_steps = Math.max(
      1,
      Math.round(reveal_duration_ms / response_reveal_frame_ms),
    );
    const characters_per_step = Math.max(
      1,
      Math.ceil(remaining_characters / reveal_steps),
    );

    const interval = globalThis.setInterval(() => {
      const approximate_end = Math.min(
        content.length,
        visible_length_ref.current + characters_per_step,
      );
      const next_length = next_reveal_boundary(content, approximate_end);
      visible_length_ref.current = next_length;
      set_visible_length(next_length);
      progress();
      if (next_length === content.length) {
        globalThis.clearInterval(interval);
        if (completed_content_ref.current !== content) {
          completed_content_ref.current = content;
          complete();
        }
      }
    }, response_reveal_frame_ms);

    return () => globalThis.clearInterval(interval);
  }, [animate, content, reduce_motion]);

  const visible_content =
    animate && !reduce_motion
      ? content.slice(0, Math.min(visible_length, content.length))
      : content;
  const is_revealing =
    animate && !reduce_motion && visible_content.length < content.length;

  return { visible_content, is_revealing };
}
