import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { applyMode, Mode } from '@cloudscape-design/global-styles';
import Alert from '@cloudscape-design/components/alert';
import AppLayout from '@cloudscape-design/components/app-layout';
import Box from '@cloudscape-design/components/box';
import Container from '@cloudscape-design/components/container';
import ContentLayout from '@cloudscape-design/components/content-layout';
import Grid from '@cloudscape-design/components/grid';
import Header from '@cloudscape-design/components/header';
import KeyValuePairs from '@cloudscape-design/components/key-value-pairs';
import SideNavigation from '@cloudscape-design/components/side-navigation';
import SpaceBetween from '@cloudscape-design/components/space-between';
import Spinner from '@cloudscape-design/components/spinner';
import TopNavigation from '@cloudscape-design/components/top-navigation';
import wasmUrl from '../core/memcore.wasm?url';
import { MemCore, CoreError } from './core/host.js';
import { buildMemory, sampleEntries } from './core/fixtures.js';
import { PROGRAMS, decode, runLine, tokenize } from './shell.js';
import Terminal, { type Entry } from './components/Terminal';
import FilesPanel, { type FileInfo } from './components/FilesPanel';
import CommandBuilder, { COMMANDS, type CommandId } from './components/CommandBuilder';

const STORE = 'memcore-playground-files-v1';
const THEME = 'memcore-playground-theme';
const REPO = 'https://github.com/SNAPKITTYAGENT9NOVA/SemIf-OpenJev';
const SUBCOMMANDS = ['memory-root', 'memory-recall', 'memory-prove', 'memory-verify', 'memory-similar', 'memory-keygen', 'memory-sign'];
const FLAGS = ['--memory', '--id', '--last', '--from', '--to', '--proof', '--root'];
const BUILTINS = ['help', 'ls', 'cat', 'rm', 'echo', 'sample', 'shake256', 'history', 'clear'];

const TEMPLATES: Record<CommandId, string> = {
  'memory-root': 'cleanroom-transformer memory-root --memory memory.jsonl',
  'memory-recall': 'cleanroom-transformer memory-recall --memory memory.jsonl --last 3',
  'memory-prove-id': 'cleanroom-transformer memory-prove --memory memory.jsonl --id route-1 > proof.json',
  'memory-prove-window': 'cleanroom-transformer memory-prove --memory memory.jsonl --from 1700000000000100 --to 1700000000000200 > gap.json',
  'memory-verify': 'cleanroom-transformer memory-verify --proof proof.json',
  'memory-similar': 'cleanroom-transformer memory-similar --memory memory.jsonl --id route-1 --last 3',
};

const BANNER = `cleanroom-transformer playground: the memory-* commands, on a freestanding WebAssembly core.
Output matches the native binary byte for byte. Type help, or pick a command on the left. Try:

  cleanroom-transformer memory-root --memory memory.jsonl
  cleanroom-transformer memory-prove --memory memory.jsonl --id route-1 > proof.json
  cleanroom-transformer memory-verify --proof proof.json
`;

// ---- per-viewer persistence of the virtual files (best effort: storage can be missing or full)
function loadFiles(): Map<string, Uint8Array> {
  try {
    const raw = localStorage.getItem(STORE);
    if (!raw) return new Map();
    const obj = JSON.parse(raw) as Record<string, string>;
    return new Map(Object.entries(obj).map(([k, v]) => [k, Uint8Array.from(atob(v), (c) => c.charCodeAt(0))]));
  } catch {
    return new Map();
  }
}
function saveFiles(fs: Map<string, Uint8Array>) {
  try {
    const obj: Record<string, string> = {};
    let total = 0;
    for (const [k, v] of fs) {
      total += v.length;
      if (total > 3 << 20) return; // keep localStorage small; larger sets live only in this tab
      let s = '';
      for (let i = 0; i < v.length; i += 0x8000) s += String.fromCharCode(...v.subarray(i, i + 0x8000));
      obj[k] = btoa(s);
    }
    localStorage.setItem(STORE, JSON.stringify(obj));
  } catch {
    /* storage unavailable: files stay in memory for this tab */
  }
}

function describe(core: MemCore, name: string, data: Uint8Array): FileInfo {
  const text = decode(data.subarray(0, 4096));
  const base = { name, size: data.length };
  if (/^\{"seq": 0,/.test(text) || (data.length === 0 && name.endsWith('.jsonl'))) {
    try {
      const r = core.roots(data);
      return { ...base, kind: `memory · ${r.count} entries`, status: 'success', detail: `root ${r.root}` };
    } catch (e) {
      const why = e instanceof CoreError ? e.message : String(e);
      return { ...base, kind: 'memory · rejected', status: 'error', detail: why };
    }
  }
  const type = /^\{"type": "([a-z-]+)"/.exec(text)?.[1];
  if (type) return { ...base, kind: `proof · ${type}`, status: 'info', detail: 'check it with memory-verify' };
  return { ...base, kind: 'file', status: 'info', detail: '' };
}

export default function App() {
  const [core, setCore] = useState<MemCore | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const fs = useRef<Map<string, Uint8Array>>(loadFiles());
  const [fsVersion, setFsVersion] = useState(0);
  const [entries, setEntries] = useState<Entry[]>([{ id: 0, line: null, stdout: BANNER, stderr: '', code: 0, ms: 0 }]);
  const [history, setHistory] = useState<string[]>([]);
  const [busy, setBusy] = useState(false);
  const [command, setCommand] = useState<CommandId>('memory-root');
  const [navOpen, setNavOpen] = useState(true);
  const [dark, setDark] = useState(() => {
    try {
      const saved = localStorage.getItem(THEME);
      if (saved) return saved === 'dark';
    } catch {
      /* no storage */
    }
    return window.matchMedia?.('(prefers-color-scheme: dark)').matches ?? false;
  });
  const [stats, setStats] = useState<ReturnType<MemCore['stats']> | null>(null);
  const nextId = useRef(1);

  useEffect(() => {
    applyMode(dark ? Mode.Dark : Mode.Light);
    document.documentElement.dataset.theme = dark ? 'dark' : 'light';
    try {
      localStorage.setItem(THEME, dark ? 'dark' : 'light');
    } catch {
      /* no storage */
    }
  }, [dark]);

  useEffect(() => {
    let live = true;
    MemCore.load(fetch(wasmUrl))
      .then((c) => {
        if (!live) return;
        if (!fs.current.size) fs.current.set('memory.jsonl', buildMemory(c, sampleEntries()).bytes);
        setCore(c);
        setStats(c.stats());
        setFsVersion((v) => v + 1);
      })
      .catch((e) => live && setLoadError(String(e?.message ?? e)));
    return () => {
      live = false;
    };
  }, []);

  const run = useCallback(
    (line: string) => {
      if (!core) return;
      const trimmed = line.trim();
      if (trimmed) setHistory((h) => (h[h.length - 1] === trimmed ? h : [...h, trimmed]));
      setBusy(true);
      const t0 = performance.now();
      let r;
      try {
        r = runLine(line, { core, fs: fs.current, history: trimmed ? [...history, trimmed] : history });
      } catch (e) {
        r = { code: 70, stdout: new Uint8Array(0), stderr: new TextEncoder().encode(`core: ${String((e as Error).message ?? e)}\n`) };
      }
      const ms = performance.now() - t0;
      setBusy(false);
      if ('clear' in r && r.clear) {
        setEntries([]);
        return;
      }
      if (!trimmed) return;
      const entry = { id: nextId.current++, line, stdout: decode(r.stdout), stderr: decode(r.stderr), code: r.code, ms };
      setEntries((xs) => [...xs.slice(-199), entry]);
      setStats(core.stats());
      saveFiles(fs.current);
      setFsVersion((v) => v + 1);
    },
    [core, history],
  );

  const files = useMemo(
    () => (core ? [...fs.current].sort(([a], [b]) => a.localeCompare(b)).map(([n, d]) => describe(core, n, d)) : []),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [core, fsVersion],
  );

  const complete = useCallback(
    (line: string) => {
      let words: string[];
      try {
        words = tokenize(line).words;
      } catch {
        return line;
      }
      const partial = /\s$/.test(line) || !line ? '' : words.pop() ?? '';
      const pool =
        words.length === 0
          ? [PROGRAMS[0], ...BUILTINS]
          : words.length === 1 && PROGRAMS.includes(words[0])
            ? SUBCOMMANDS
            : [...FLAGS, ...fs.current.keys()];
      const hits = pool.filter((p) => p.startsWith(partial));
      if (!hits.length) return line;
      let lcp = hits[0];
      for (const h of hits) while (!h.startsWith(lcp)) lcp = lcp.slice(0, -1);
      const head = line.slice(0, line.length - partial.length);
      return head + lcp + (hits.length === 1 ? ' ' : '');
    },
    [],
  );

  const upload = useCallback(async (list: File[]) => {
    for (const f of list) fs.current.set(f.name, new Uint8Array(await f.arrayBuffer()));
    saveFiles(fs.current);
    setFsVersion((v) => v + 1);
  }, []);

  const pick = (id: CommandId) => {
    setCommand(id);
    run(TEMPLATES[id]);
  };

  const pageContent = loadError ? (
    <Alert type="error" header="The WebAssembly core did not load">
      {loadError}. The playground needs a browser with WebAssembly and bulk-memory support.
    </Alert>
  ) : !core ? (
    <Box textAlign="center" padding="xxl">
      <Spinner size="large" /> <Box variant="p">Loading memcore.wasm…</Box>
    </Box>
  ) : (
    <SpaceBetween size="l">
      <Terminal
        entries={entries}
        busy={busy}
        onRun={run}
        onClear={() => setEntries([])}
        history={history}
        complete={complete}
      />
      <Grid gridDefinition={[{ colspan: { default: 12, m: 8 } }, { colspan: { default: 12, m: 4 } }]}>
          <FilesPanel
            files={files}
            read={(n) => fs.current.get(n)}
            onUpload={upload}
            onSample={() => run('sample memory.jsonl')}
            onDelete={(n) => run(`rm '${n.replace(/'/g, `'\\''`)}'`)}
          />
          <Container header={<Header variant="h2">Core</Header>}>
            {stats && (
              <KeyValuePairs
                columns={2}
                items={[
                  { label: 'ABI version', value: String(stats.abi) },
                  { label: 'Linear memory', value: `${stats.memoryBytes / 65536} pages · ${(stats.memoryBytes / (1 << 20)).toFixed(0)} MiB` },
                  { label: 'Stack', value: `${stats.stackSize / 1024} KiB at 0x0` },
                  { label: 'Heap start', value: `0x${stats.heapLo.toString(16)}` },
                  // the stats call's own 72-byte output block is the only one live while it runs
                  { label: 'Heap between commands', value: `${stats.heapHi - stats.heapLo - 72} B (freed and zeroed)` },
                  { label: 'Faults', value: stats.fault ? `bad free (${stats.fault})` : 'none' },
                ]}
              />
            )}
          </Container>
      </Grid>
      <CommandBuilder command={command} onCommand={setCommand} fileNames={files.map((f) => f.name)} onRun={run} />
    </SpaceBetween>
  );

  return (
    <>
      <div id="top-nav">
        <TopNavigation
          identity={{ href: '#', title: 'cleanroom-transformer playground' }}
          utilities={[
            { type: 'button', text: dark ? 'Light mode' : 'Dark mode', onClick: () => setDark((d) => !d) },
            { type: 'button', text: 'Source', href: `${REPO}/tree/master/playground`, external: true },
          ]}
        />
      </div>
      <AppLayout
        headerSelector="#top-nav"
        toolsHide
        navigationOpen={navOpen}
        onNavigationChange={({ detail }) => setNavOpen(detail.open)}
        navigation={
          <SideNavigation
            header={{ text: 'Commands', href: '#' }}
            activeHref={`#${command}`}
            onFollow={(e) => {
              if (e.detail.external) return; // let the browser open external links
              e.preventDefault();
              const id = e.detail.href.slice(1);
              if (id in TEMPLATES) pick(id as CommandId);
              else if (id === 'help') run('help');
              else if (id === 'ls') run('ls -l');
            }}
            items={[
              ...COMMANDS.map((c) => ({ type: 'link' as const, text: 'label' in c ? c.label : c.value, href: `#${c.value}` })),
              { type: 'divider' },
              { type: 'link', text: 'help', href: '#help' },
              { type: 'link', text: 'ls -l', href: '#ls' },
              { type: 'divider' },
              { type: 'link', text: 'SPEC.md §6 (decision memory)', href: `${REPO}/blob/master/cleanroom-transformer/SPEC.md#6-decision-memory`, external: true },
            ]}
          />
        }
        content={
          <ContentLayout
            header={
              <Header
                variant="h1"
                description="Commit, prove and verify decision memories in your browser. The commands and their output are those of the native engine; nothing leaves this tab."
              >
                Decision memory playground
              </Header>
            }
          >
            {pageContent}
          </ContentLayout>
        }
      />
    </>
  );
}
