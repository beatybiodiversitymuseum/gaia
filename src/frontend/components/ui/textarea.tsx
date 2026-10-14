import type { ComponentProps } from 'react';

const base_class =
  'flex field-sizing-content w-full rounded-lg text-sm transition-colors outline-none disabled:cursor-not-allowed disabled:opacity-50';

function Textarea({
  className = '',
  ...properties
}: ComponentProps<'textarea'>) {
  return (
    <textarea className={`${base_class} ${className}`.trim()} {...properties} />
  );
}

export { Textarea };
