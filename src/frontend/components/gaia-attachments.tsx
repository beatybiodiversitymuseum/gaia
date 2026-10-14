'use client';

import { Check, Download, FileSpreadsheet, LoaderCircle } from 'lucide-react';
import { useId, useState } from 'react';

import type { GaiaAttachment } from '@/lib/gaia-api';

function format_file_size(size_bytes: number) {
  if (size_bytes < 1_000) {
    return `${size_bytes.toLocaleString()} B`;
  }
  if (size_bytes < 1_000_000) {
    return `${(size_bytes / 1_000).toLocaleString(undefined, { maximumFractionDigits: 1 })} kB`;
  }
  return `${(size_bytes / 1_000_000).toLocaleString(undefined, { maximumFractionDigits: 1 })} MB`;
}

function FileAttachment({
  attachment,
  on_download,
}: {
  attachment: GaiaAttachment;
  on_download: (attachment: GaiaAttachment) => Promise<void>;
}) {
  const [status, set_status] = useState<'idle' | 'downloading' | 'started'>(
    'idle',
  );
  const [error, set_error] = useState<string | null>(null);
  const status_id = useId();
  const is_downloading = status === 'downloading';

  async function download_file() {
    if (is_downloading) {
      return;
    }
    set_status('downloading');
    set_error(null);
    try {
      await on_download(attachment);
      set_status('started');
    } catch (download_error) {
      set_status('idle');
      set_error(
        download_error instanceof TypeError
          ? 'The download could not connect. Try again.'
          : download_error instanceof Error
            ? download_error.message
            : 'The file could not be downloaded. Try again.',
      );
    }
  }

  return (
    <li className="file-attachment">
      <div className="file-attachment-row">
        <span className="file-attachment-icon">
          <FileSpreadsheet aria-hidden="true" />
        </span>
        <div className="file-attachment-details">
          <strong>{attachment.filename}</strong>
          <span>
            Excel workbook · {format_file_size(attachment.size_bytes)}
          </span>
        </div>
        <button
          aria-busy={is_downloading}
          aria-describedby={status_id}
          aria-label={`Download ${attachment.filename}`}
          className="file-download-button"
          disabled={is_downloading}
          onClick={() => void download_file()}
          type="button"
        >
          {is_downloading ? (
            <LoaderCircle aria-hidden="true" />
          ) : (
            <Download aria-hidden="true" />
          )}
          <span>{is_downloading ? 'Downloading…' : 'Download'}</span>
        </button>
      </div>
      <div aria-live="polite" id={status_id}>
        {error ? (
          <p className="file-download-error" role="alert">
            {error}
          </p>
        ) : status === 'started' ? (
          <p className="file-download-status">
            <Check aria-hidden="true" /> Download started
          </p>
        ) : null}
      </div>
    </li>
  );
}

export function GaiaAttachments({
  attachments,
  on_download,
}: {
  attachments: readonly GaiaAttachment[];
  on_download: (attachment: GaiaAttachment) => Promise<void>;
}) {
  return (
    <ul aria-label="Files from Gaia" className="file-attachments">
      {attachments.map((attachment) => (
        <FileAttachment
          attachment={attachment}
          key={attachment.file_id}
          on_download={on_download}
        />
      ))}
    </ul>
  );
}
