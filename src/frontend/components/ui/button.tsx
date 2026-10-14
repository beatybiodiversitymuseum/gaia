import type { ComponentProps } from 'react';

const base_class =
  'inline-flex shrink-0 items-center justify-center border border-transparent bg-clip-padding text-sm font-medium whitespace-nowrap transition-all outline-none select-none focus-visible:border-ring focus-visible:ring-3 focus-visible:ring-ring/50 active:translate-y-px disabled:pointer-events-none disabled:opacity-50 [&_svg]:pointer-events-none [&_svg]:shrink-0 [&_svg]:size-4';

function Button({ className = '', ...properties }: ComponentProps<'button'>) {
  return (
    <button className={`${base_class} ${className}`.trim()} {...properties} />
  );
}

export { Button };
