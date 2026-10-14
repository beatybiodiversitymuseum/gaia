'use client';

import { useCallback, useEffect, useState } from 'react';

import {
  check_gaia_health,
  clear_gaia_session_token,
  create_gaia_session,
  GaiaRequestError,
  type GaiaSession,
  read_gaia_session_token,
  restore_gaia_session,
  store_gaia_session_token,
} from '@/lib/gaia-api';

const health_check_interval_ms = 20 * 1000;

type AuthenticationState =
  | { status: 'checking' | 'signed_out'; session: null }
  | { status: 'signed_in'; session: GaiaSession };

export function useGaiaSession(on_session_ended: () => void) {
  const [authentication, set_authentication] = useState<AuthenticationState>({
    status: 'checking',
    session: null,
  });
  const [login_error, set_login_error] = useState('');
  const [is_signing_in, set_is_signing_in] = useState(false);
  const token = authentication.session?.token;

  const end_session = useCallback(
    (message = '') => {
      on_session_ended();
      try {
        clear_gaia_session_token();
      } catch {
        // Local state still ends the session when browser storage is unavailable.
      }
      set_authentication({ status: 'signed_out', session: null });
      set_login_error(message);
    },
    [on_session_ended],
  );

  useEffect(() => {
    const controller = new AbortController();

    async function restore_session() {
      let stored_token: string | null;
      try {
        stored_token = read_gaia_session_token();
      } catch {
        set_authentication({ status: 'signed_out', session: null });
        return;
      }
      if (!stored_token) {
        set_authentication({ status: 'signed_out', session: null });
        return;
      }

      try {
        const username = await restore_gaia_session(
          stored_token,
          controller.signal,
        );
        if (controller.signal.aborted) {
          return;
        }
        if (!username) {
          end_session();
          return;
        }
        set_authentication({
          status: 'signed_in',
          session: { token: stored_token, username },
        });
      } catch {
        if (!controller.signal.aborted) {
          end_session();
        }
      }
    }

    void restore_session();
    return () => controller.abort();
  }, [end_session]);

  useEffect(() => {
    if (!token) {
      return;
    }

    const session_token = token;
    let controller: AbortController | null = null;

    async function check_connection() {
      controller?.abort();
      const request_controller = new AbortController();
      controller = request_controller;
      try {
        await check_gaia_health(
          session_token,
          AbortSignal.any([
            request_controller.signal,
            AbortSignal.timeout(3_500),
          ]),
        );
      } catch (error) {
        if (
          !request_controller.signal.aborted &&
          error instanceof GaiaRequestError &&
          error.status === 401
        ) {
          end_session(error.message);
        }
      }
    }

    void check_connection();
    const interval = globalThis.setInterval(
      check_connection,
      health_check_interval_ms,
    );
    return () => {
      controller?.abort();
      globalThis.clearInterval(interval);
    };
  }, [end_session, token]);

  async function sign_in(username: string, password: string): Promise<boolean> {
    const trimmed_username = username.trim();
    if (!trimmed_username || !password || is_signing_in) {
      return false;
    }

    set_is_signing_in(true);
    set_login_error('');
    try {
      const session = await create_gaia_session(trimmed_username, password);
      store_gaia_session_token(session.token);
      set_authentication({ status: 'signed_in', session });
      return true;
    } catch (error) {
      set_login_error(
        error instanceof Error
          ? error.message
          : 'Gaia could not sign you in. Try again.',
      );
      return false;
    } finally {
      set_is_signing_in(false);
    }
  }

  return { authentication, login_error, is_signing_in, sign_in, end_session };
}
