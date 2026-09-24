import { useEffect, useRef, useState, type KeyboardEvent } from 'react';
import * as tokens from '@cloudscape-design/design-tokens';

export type Entry = {
  id: number;
  line: string | null; // null: a banner, not a command
  stdout: string;
  stderr: string;
  code: number;
  ms: number;
};

type Props = {
  entries: Entry[];
  busy: boolean;
  onRun: (line: string) => void;
  onClear: () => void;
  history: string[];
  complete: (line: string) => string;
};

export default function Terminal({ entries, busy, onRun, onClear, history, complete }: Props) {
  const [line, setLine] = useState('');
  const [cursor, setCursor] = useState<number | null>(null); // position in history while browsing
  const [copied, setCopied] = useState(false);
  const screen = useRef<HTMLDivElement>(null);
  const input = useRef<HTMLInputElement>(null);

  useEffect(() => {
    const el = screen.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [entries]);

  const submit = () => {
    if (busy) return;
    onRun(line);
    setLine('');
    setCursor(null);
  };

  const onKey = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      submit();
    } else if (e.key === 'ArrowUp' && history.length) {
      e.preventDefault();
      const at = cursor === null ? history.length - 1 : Math.max(0, cursor - 1);
      setCursor(at);
      setLine(history[at]);
    } else if (e.key === 'ArrowDown' && cursor !== null) {
      e.preventDefault();
      const at = cursor + 1;
      if (at >= history.length) {
        setCursor(null);
        setLine('');
      } else {
        setCursor(at);
        setLine(history[at]);
      }
    } else if (e.key === 'Tab') {
      e.preventDefault();
      setLine(complete(line));
    } else if (e.key === 'l' && e.ctrlKey) {
      e.preventDefault();
      onClear();
    }
  };

  const copy = async () => {
    const text = entries
      .map((x) => (x.line === null ? x.stdout : `$ ${x.line}\n${x.stdout}${x.stderr}`))
      .join('');
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      /* clipboard unavailable (insecure context or denied): nothing to do */
    }
  };

  return (
    <section className="card pg-terminal" aria-label="Terminal">
      <div className="flex-between pg-bar">
        <span className="pg-bar-title">cleanroom-transformer · memcore.wasm</span>
        <span className="pg-bar-actions">
          <button type="button" className="btn btn-outline" onClick={copy}>
            {copied ? 'Copied' : 'Copy'}
          </button>
          <button type="button" className="btn btn-outline" onClick={onClear}>
            Clear
          </button>
        </span>
      </div>
      <div className="pg-screen" ref={screen} role="log" aria-live="polite" onClick={() => input.current?.focus()}>
        {entries.map((x) => (
          <div className="pg-entry" key={x.id} data-testid={x.line === null ? 'banner' : 'entry'}>
            {x.line === null ? (
              <pre className="pg-banner">{x.stdout}</pre>
            ) : (
              <>
                <div className="pg-cmd">
                  <span className="pg-ps1">$ </span>
                  {x.line}
                  <span className="pg-meta">
                    {x.code !== 0 && (
                      <span
                        className="pg-exit"
                        style={{ color: tokens.colorTextStatusError }}
                        title="exit status"
                        data-testid="exit"
                      >
                        exit {x.code}
                      </span>
                    )}
                    <span title="time in the core">{x.ms < 1 ? '<1' : Math.round(x.ms)} ms</span>
                  </span>
                </div>
                {x.stdout && <pre data-testid="stdout">{x.stdout}</pre>}
                {x.stderr && (
                  <pre data-testid="stderr" style={{ color: tokens.colorTextStatusError }}>
                    {x.stderr}
                  </pre>
                )}
              </>
            )}
          </div>
        ))}
      </div>
      <form
        className="pg-input-row"
        onSubmit={(e) => {
          e.preventDefault();
          submit();
        }}
      >
        <span className="pg-ps1" aria-hidden="true">
          $
        </span>
        <input
          ref={input}
          className="input"
          value={line}
          onChange={(e) => setLine(e.target.value)}
          onKeyDown={onKey}
          placeholder="cleanroom-transformer memory-root --memory memory.jsonl"
          aria-label="Command line"
          autoCapitalize="off"
          autoCorrect="off"
          autoComplete="off"
          spellCheck={false}
          disabled={busy}
          data-testid="command-input"
        />
        <span className="pg-hint">Tab completes · ↑ history</span>
      </form>
    </section>
  );
}
