import { useMemo, useState } from 'react';
import Box from '@cloudscape-design/components/box';
import Button from '@cloudscape-design/components/button';
import ColumnLayout from '@cloudscape-design/components/column-layout';
import Container from '@cloudscape-design/components/container';
import FormField from '@cloudscape-design/components/form-field';
import Header from '@cloudscape-design/components/header';
import Input from '@cloudscape-design/components/input';
import Select, { type SelectProps } from '@cloudscape-design/components/select';
import SpaceBetween from '@cloudscape-design/components/space-between';
import { quote } from '../shell.js';

export const COMMANDS = [
  { value: 'memory-root', description: 'Commitment roots of a memory file' },
  { value: 'memory-recall', description: 'Stored decisions, oldest first' },
  { value: 'memory-prove-id', label: 'memory-prove (id)', description: 'Proof that an id has a latest entry, or none' },
  { value: 'memory-prove-window', label: 'memory-prove (time window)', description: 'Proof that nothing was recorded in a window' },
  { value: 'memory-verify', description: 'Check a proof file on its own' },
  { value: 'memory-similar', description: 'Entries whose hidden states are closest' },
] as const;
export type CommandId = (typeof COMMANDS)[number]['value'];

type Props = {
  command: CommandId;
  onCommand: (c: CommandId) => void;
  fileNames: string[];
  onRun: (line: string) => void;
};

const opt = (v: string): SelectProps.Option => ({ value: v, label: v });

export default function CommandBuilder({ command, onCommand, fileNames, onRun }: Props) {
  const [memory, setMemory] = useState('memory.jsonl');
  const [proof, setProof] = useState('proof.json');
  const [id, setId] = useState('route-1');
  const [last, setLast] = useState('');
  const [from, setFrom] = useState('1700000000000100');
  const [to, setTo] = useState('1700000000000200');
  const [root, setRoot] = useState('');
  const [out, setOut] = useState('');

  const line = useMemo(() => {
    const w: string[] = ['cleanroom-transformer'];
    const sub = command.startsWith('memory-prove') ? 'memory-prove' : command;
    w.push(sub);
    if (command === 'memory-verify') {
      w.push('--proof', proof);
      if (root.trim()) w.push('--root', root.trim());
    } else {
      w.push('--memory', memory);
      if (command === 'memory-recall' && id.trim()) w.push('--id', id.trim());
      if (command === 'memory-prove-id' || command === 'memory-similar') w.push('--id', id);
      if (command === 'memory-prove-window') w.push('--from', from.trim(), '--to', to.trim());
      if ((command === 'memory-recall' || command === 'memory-similar') && last.trim()) w.push('--last', last.trim());
    }
    const cmd = w.map(quote).join(' ');
    return out.trim() ? `${cmd} > ${quote(out.trim())}` : cmd;
  }, [command, memory, proof, id, last, from, to, root, out]);

  const fileSelect = (value: string, set: (v: string) => void, label: string) => (
    <FormField label={label}>
      <Select
        selectedOption={opt(value)}
        onChange={({ detail }) => set(detail.selectedOption.value ?? '')}
        options={(fileNames.includes(value) ? fileNames : [value, ...fileNames]).map(opt)}
        filteringType="auto"
      />
    </FormField>
  );

  const current = COMMANDS.find((c) => c.value === command)!;
  return (
    <Container
      header={
        <Header
          variant="h2"
          description="Fill in the flags; the command runs in the terminal exactly as typed."
          actions={
            <Button variant="primary" onClick={() => onRun(line)} data-testid="builder-run">
              Run
            </Button>
          }
        >
          Command builder
        </Header>
      }
    >
      <SpaceBetween size="m">
        <ColumnLayout columns={2}>
          <FormField label="Command" description={current.description}>
            <Select
              selectedOption={{ value: command, label: 'label' in current ? current.label : current.value }}
              onChange={({ detail }) => onCommand(detail.selectedOption.value as CommandId)}
              options={COMMANDS.map((c) => ({ value: c.value, label: 'label' in c ? c.label : c.value, description: c.description }))}
            />
          </FormField>
          {command === 'memory-verify' ? fileSelect(proof, setProof, '--proof') : fileSelect(memory, setMemory, '--memory')}
          {(command === 'memory-recall' || command === 'memory-prove-id' || command === 'memory-similar') && (
            <FormField label="--id" description={command === 'memory-recall' ? 'Optional: one id only' : undefined}>
              <Input value={id} onChange={({ detail }) => setId(detail.value)} />
            </FormField>
          )}
          {command === 'memory-prove-window' && (
            <>
              <FormField label="--from (µs)">
                <Input value={from} onChange={({ detail }) => setFrom(detail.value)} inputMode="numeric" />
              </FormField>
              <FormField label="--to (µs)">
                <Input value={to} onChange={({ detail }) => setTo(detail.value)} inputMode="numeric" />
              </FormField>
            </>
          )}
          {(command === 'memory-recall' || command === 'memory-similar') && (
            <FormField label="--last" description={command === 'memory-similar' ? 'Default 10' : 'Default: all'}>
              <Input value={last} onChange={({ detail }) => setLast(detail.value)} inputMode="numeric" />
            </FormField>
          )}
          {command === 'memory-verify' && (
            <FormField label="--root" description="Optional: the published root to check against">
              <Input value={root} onChange={({ detail }) => setRoot(detail.value)} placeholder="128 hex digits" />
            </FormField>
          )}
          <FormField label="Save stdout to" description="Optional: > FILE">
            <Input value={out} onChange={({ detail }) => setOut(detail.value)} placeholder="proof.json" />
          </FormField>
        </ColumnLayout>
        <Box variant="code">
          <span className="pg-preview" data-testid="builder-preview">
            {line}
          </span>
        </Box>
      </SpaceBetween>
    </Container>
  );
}
