'use client';
import { use, useState } from 'react';
import Workspace from '@/datasetStudio/Workspace';
import NativeMediaView from '@/datasetStudio/NativeMediaView';
export default function DatasetPage({ params }: { params: Promise<{ datasetName: string }> }) {
  const { datasetName } = use(params),
    [native, setNative] = useState(false),
    name = decodeURIComponent(datasetName);
  return native ? (
    <>
      <button
        className="fixed right-4 top-3 z-50 rounded bg-gray-800 border border-gray-500 px-3 py-2"
        onClick={() => setNative(false)}
      >
        Return to image Studio
      </button>
      <NativeMediaView datasetName={name} />
    </>
  ) : (
    <Workspace dataset={name} onNativeView={() => setNative(true)} />
  );
}
