'use client';

import { useMemo, useState, use } from 'react';
import { FaChevronLeft } from 'react-icons/fa';
import { MdDashboard, MdImage, MdShowChart, MdCode, MdExtension } from 'react-icons/md';
import { Button } from '@headlessui/react';
import { TopBar, MainContent } from '@/components/layout';
import useJob from '@/hooks/useJob';
import usePollLoop from '@/hooks/usePollLoop';
import SampleImages, { SampleImagesMenu } from '@/components/SampleImages';
import JobOverview from '@/components/JobOverview';
import { redirect, useRouter, useSearchParams } from 'next/navigation';
import JobActionBar from '@/components/JobActionBar';
import JobConfigViewer from '@/components/JobConfigViewer';
import JobMetricsGraph from '@/components/JobMetricsGraph';
import JobMetricsCompareGraph from '@/components/JobMetricsCompareGraph';
import DepthPreviews from '@/components/DepthPreviews';
import IdentityPreviews from '@/components/IdentityPreviews';
import useDepthPreviews from '@/hooks/useDepthPreviews';
import useIdentityPreviews from '@/hooks/useIdentityPreviews';
import JobLossGraph from '@/components/JobLossGraph';
import JobPlugin from '@/components/JobPlugin';
import { Job } from '@prisma/client';
import { apiClient } from '@/utils/api';

type PageKey =
  | 'overview'
  | 'samples'
  | 'depth_previews'
  | 'identity_previews'
  | 'config'
  | 'metrics'
  | 'metrics_compare'
  | 'loss_log'
  | 'plugin';
const PAGE_KEYS = new Set<PageKey>([
  'overview',
  'samples',
  'depth_previews',
  'identity_previews',
  'config',
  'metrics',
  'metrics_compare',
  'loss_log',
  'plugin',
]);

interface Page {
  name: string;
  value: PageKey;
  icon?: React.ComponentType<{ className?: string }>;
  component: React.ComponentType<{ job: Job }>;
  menuItem?: React.ComponentType<{ job?: Job | null }> | null;
  mainCss?: string;
  jobTypes?: string[]; // if specified, only show this page for these job types
}

const pages: Page[] = [
  {
    name: 'Overview',
    value: 'overview',
    icon: MdDashboard,
    component: JobOverview,
    mainCss: 'pt-24',
  },
  {
    name: 'Samples',
    value: 'samples',
    icon: MdImage,
    component: SampleImages,
    menuItem: SampleImagesMenu,
    mainCss: 'pt-24',
    jobTypes: ['train'],
  },
  {
    name: 'Depth Previews',
    value: 'depth_previews',
    component: DepthPreviews,
    mainCss: 'pt-24',
  },
  {
    name: 'Identity Previews',
    value: 'identity_previews',
    component: IdentityPreviews,
    mainCss: 'pt-24',
  },
  {
    name: 'Metrics',
    value: 'metrics',
    component: JobMetricsGraph,
    mainCss: 'pt-24',
  },
  {
    // Cross-job comparison: pick a metric, fan it across N selected jobs.
    // Anchored on the current job; additional jobs picked from the multi-
    // select. Same fetch pipeline as Metrics (new), N-way parallel.
    name: 'Compare Jobs',
    value: 'metrics_compare',
    component: JobMetricsCompareGraph,
    mainCss: 'pt-24',
  },
  {
    name: 'Loss Graph',
    value: 'loss_log',
    icon: MdShowChart,
    component: JobLossGraph,
    mainCss: 'pt-24 pb-4',
    jobTypes: ['train'],
  },
  {
    name: 'Config File',
    value: 'config',
    icon: MdCode,
    component: JobConfigViewer,
    mainCss: 'pt-[80px] px-0 pb-0',
  },
  {
    name: 'Plugin',
    value: 'plugin',
    icon: MdExtension,
    component: JobPlugin,
    mainCss: 'pt-[80px] px-0 pb-0',
  },
];

export default function JobPage({ params }: { params: { jobID: string } }) {
  const usableParams = use(params as any) as { jobID: string };
  const jobID = usableParams.jobID;
  const { job, status, refreshJob } = useJob(jobID, 5000);
  // Depth/Identity preview tabs are content-gated: poll the preview folders for
  // the viewed job and only surface a tab once its folder actually has files.
  // Lifting the fetch here (rather than inside the tab component) lets the tab
  // bar react to "first preview written" without the component mounting first,
  // and hands the already-loaded data down so the tab renders immediately
  // instead of flashing its own loading state.
  const depth = useDepthPreviews(jobID, 5000);
  const identity = useIdentityPreviews(jobID, 5000);
  const [hasPlugin, setHasPlugin] = useState(false);

  // poll for plugin.html in the job folder; show the Plugin tab if it exists
  usePollLoop(
    () =>
      apiClient
        .get(`/api/jobs/${jobID}/plugin?check=1`)
        .then(res => res.data)
        .then(data => setHasPlugin(!!data.exists))
        .catch(() => {}),
    5000,
    [jobID],
  );

  // Tab selection lives in the URL (`?tab=…`) so refresh + tab-switch-and-return
  // both preserve it, and the URL is shareable. Per-tab interior state (filters,
  // sort, etc.) is the tab component's own responsibility — see DepthPreviews
  // for the pattern.
  const router = useRouter();
  const searchParams = useSearchParams();
  const rawTab = searchParams.get('tab');
  const pageKey: PageKey = rawTab && PAGE_KEYS.has(rawTab as PageKey) ? (rawTab as PageKey) : 'overview';
  const setPageKey = (k: PageKey) => {
    const params = new URLSearchParams(searchParams.toString());
    params.set('tab', k);
    router.replace(`?${params.toString()}`, { scroll: false });
  };

  // Content-gated tabs: hide Depth/Identity Previews until their folder has at
  // least one file. Every other tab is always shown.
  const jobTypeForTabs = job?.job_type || 'unknown';
  const visiblePages = useMemo(
    () =>
      pages.filter(p => {
        if (p.jobTypes && !p.jobTypes.includes(jobTypeForTabs)) return false;
        if (p.value === 'depth_previews') return depth.previews.length > 0;
        if (p.value === 'identity_previews') return identity.previews.length > 0;
        if (p.value === 'plugin') return hasPlugin;
        return true;
      }),
    [depth.previews.length, identity.previews.length, hasPlugin, jobTypeForTabs],
  );
  // If the previously selected tab no longer applies (e.g. a preview tab whose
  // folder is still empty, or whose files were removed), bounce to the first
  // visible tab (overview).
  const page = visiblePages.find(p => p.value === pageKey) ?? visiblePages[0];
  const effectivePageKey = page?.value ?? 'overview';

  const jobType = job?.job_type || 'unknown';

  let title = `Job: ${job?.name || 'Loading...'}`;
  if (jobType === 'caption') {
    title = `Captioning: ${job?.job_ref || 'Loading...'}`;
  }

  return (
    <>
      {/* Fixed top bar */}
      <TopBar>
        <div className="flex-shrink-0">
          <Button className="text-gray-500 dark:text-gray-300 pl-0 pr-1 sm:px-3 mt-1" onClick={() => redirect('/jobs')}>
            <FaChevronLeft />
          </Button>
        </div>
        <div className="min-w-0 flex-shrink">
          <h1 className="text-base sm:text-lg truncate">{title}</h1>
        </div>
        <div className="flex-1"></div>
        {job && (
          <JobActionBar
            job={job}
            onRefresh={refreshJob}
            hideView
            afterDelete={() => {
              redirect('/jobs');
            }}
            autoStartQueue={true}
          />
        )}
      </TopBar>
      <MainContent className={page?.mainCss}>
        {status === 'loading' && job == null && <p>Loading...</p>}
        {status === 'error' && job == null && <p>Error fetching job</p>}
        {/* All tabs mount once and stay mounted; we hide inactive ones with
            display:none rather than unmounting so per-tab local state (zoom,
            selected series, scroll position, etc.) survives a tab switch.
            Tabs that need to persist across *refresh* still mirror to the
            URL on their own (see DepthPreviews for the pattern). */}
        {job && (
          <>
            {visiblePages.map(p => {
              const isActive = p.value === effectivePageKey;
              // Depth/Identity get the page-level fetched data injected; every
              // other tab just receives the job.
              let body: React.ReactNode;
              if (p.value === 'depth_previews') {
                body = <DepthPreviews job={job} previews={depth.previews} status={depth.status} />;
              } else if (p.value === 'identity_previews') {
                body = <IdentityPreviews job={job} previews={identity.previews} status={identity.status} />;
              } else {
                const Component = p.component;
                body = <Component job={job} />;
              }
              return (
                <div key={p.value} className={isActive ? 'contents' : 'hidden'} aria-hidden={!isActive}>
                  {body}
                </div>
              );
            })}
          </>
        )}
      </MainContent>
      <div className="bg-gray-800 absolute top-12 left-0 w-full h-8 flex items-center px-0 sm:px-2 text-sm sm:overflow-x-auto whitespace-nowrap">
        {visiblePages.map(p => (
          <Button
            key={p.value}
            onClick={() => setPageKey(p.value)}
            className={`flex-1 sm:flex-initial justify-center px-2 sm:px-4 py-1 h-8 flex items-center gap-1.5 sm:flex-shrink-0 ${p.value === effectivePageKey ? 'bg-gray-300 dark:bg-gray-700 text-white' : ''}`}
          >
            {p.icon && <p.icon className="text-sm" />}
            <span className={p.icon ? 'hidden sm:inline' : ''}>{p.name}</span>
          </Button>
        ))}
        {page?.menuItem && (
          <>
            <div className="hidden sm:block flex-grow"></div>
            <page.menuItem job={job} />
          </>
        )}
      </div>
    </>
  );
}
