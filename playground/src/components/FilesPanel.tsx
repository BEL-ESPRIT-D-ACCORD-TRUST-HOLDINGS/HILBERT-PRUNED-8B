import { useRef, useState, type DragEvent } from 'react';
import Box from '@cloudscape-design/components/box';
import Button from '@cloudscape-design/components/button';
import Header from '@cloudscape-design/components/header';
import Modal from '@cloudscape-design/components/modal';
import SpaceBetween from '@cloudscape-design/components/space-between';
import StatusIndicator from '@cloudscape-design/components/status-indicator';
import Table from '@cloudscape-design/components/table';

export type FileInfo = {
  name: string;
  size: number;
  kind: string;
  status: 'success' | 'error' | 'info';
  detail: string;
};

type Props = {
  files: FileInfo[];
  read: (name: string) => Uint8Array | undefined;
  onUpload: (files: File[]) => void;
  onSample: () => void;
  onDelete: (name: string) => void;
};

const fmtSize = (n: number) => (n < 1024 ? `${n} B` : n < 1 << 20 ? `${(n / 1024).toFixed(1)} KB` : `${(n / (1 << 20)).toFixed(1)} MB`);

export default function FilesPanel({ files, read, onUpload, onSample, onDelete }: Props) {
  const [selected, setSelected] = useState<FileInfo[]>([]);
  const [viewing, setViewing] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const picker = useRef<HTMLInputElement>(null);
  const current = selected.length ? files.find((f) => f.name === selected[0].name) : undefined;

  const download = (name: string) => {
    const data = read(name);
    if (!data) return;
    const url = URL.createObjectURL(new Blob([data as BlobPart], { type: 'application/octet-stream' }));
    const a = document.createElement('a');
    a.href = url;
    a.download = name.split('/').pop() || 'file';
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  };

  const onDrop = (e: DragEvent) => {
    e.preventDefault();
    setDragging(false);
    if (e.dataTransfer.files.length) onUpload([...e.dataTransfer.files]);
  };

  const viewed = viewing ? read(viewing) : undefined;

  return (
    <div
      className={dragging ? 'pg-drop' : undefined}
      onDragOver={(e) => {
        e.preventDefault();
        setDragging(true);
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={onDrop}
    >
      <input
        ref={picker}
        type="file"
        multiple
        hidden
        onChange={(e) => {
          if (e.target.files?.length) onUpload([...e.target.files]);
          e.target.value = '';
        }}
      />
      <Table
        variant="container"
        trackBy="name"
        selectionType="single"
        selectedItems={current ? [current] : []}
        onSelectionChange={({ detail }) => setSelected(detail.selectedItems)}
        items={files}
        header={
          <Header
            counter={`(${files.length})`}
            description="Files in this tab. Drop files here to add them."
            actions={
              <SpaceBetween direction="horizontal" size="xs">
                <Button onClick={() => picker.current?.click()} iconName="upload">
                  Upload
                </Button>
                <Button onClick={onSample}>Sample memory</Button>
              </SpaceBetween>
            }
          >
            Files
          </Header>
        }
        footer={
          current && (
            <SpaceBetween direction="horizontal" size="xs">
              <Button onClick={() => setViewing(current.name)} iconName="file-open">
                View
              </Button>
              <Button onClick={() => download(current.name)} iconName="download">
                Download
              </Button>
              <Button
                onClick={() => {
                  onDelete(current.name);
                  setSelected([]);
                }}
                iconName="remove"
              >
                Delete
              </Button>
            </SpaceBetween>
          )
        }
        columnDefinitions={[
          {
            id: 'name',
            header: 'Name',
            cell: (f) => (
              <Box variant="code" fontSize="body-s">
                {f.name}
              </Box>
            ),
            isRowHeader: true,
          },
          { id: 'size', header: 'Size', cell: (f) => fmtSize(f.size) },
          {
            id: 'kind',
            header: 'Contents',
            cell: (f) => (
              <span title={f.detail}>
                <StatusIndicator type={f.status}>{f.kind}</StatusIndicator>
              </span>
            ),
          },
        ]}
        wrapLines
        empty={
          <Box textAlign="center" color="inherit" padding="s">
            <SpaceBetween size="xs">
              <b>No files</b>
              <Button onClick={onSample}>Create the sample memory</Button>
            </SpaceBetween>
          </Box>
        }
      />
      <Modal
        visible={viewing !== null}
        onDismiss={() => setViewing(null)}
        header={viewing}
        size="max"
        footer={
          <Box float="right">
            <SpaceBetween direction="horizontal" size="xs">
              {viewing && <Button onClick={() => download(viewing)}>Download</Button>}
              <Button variant="primary" onClick={() => setViewing(null)}>
                Close
              </Button>
            </SpaceBetween>
          </Box>
        }
      >
        {viewed && <pre className="pg-file-view">{new TextDecoder().decode(viewed.subarray(0, 1 << 20))}</pre>}
        {viewed && viewed.length > 1 << 20 && <Box color="text-status-inactive">Showing the first 1 MB.</Box>}
      </Modal>
    </div>
  );
}
