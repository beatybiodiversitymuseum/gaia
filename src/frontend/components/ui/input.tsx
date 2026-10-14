import type { ComponentProps } from 'react';

const base_class =
  'w-full min-w-0 border text-sm transition-colors outline-none placeholder:text-muted-foreground aria-invalid:ring-3 aria-invalid:ring-destructive/20';

function Input({ className = '', ...properties }: ComponentProps<'input'>) {
  return (
    <input className={`${base_class} ${className}`.trim()} {...properties} />
  );
}

export { Input };
