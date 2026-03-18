<script lang="ts">
  import { onMount } from 'svelte';
  import { check, type Update } from '@tauri-apps/plugin-updater';
  import { relaunch } from '@tauri-apps/plugin-process';

  let update = $state<Update | null>(null);
  let checking = $state(false);
  let downloading = $state(false);
  let progress = $state(0);
  let totalBytes = $state(0);
  let downloadedBytes = $state(0);
  let error = $state<string | null>(null);
  let dismissed = $state(false);

  onMount(() => {
    // Check for updates 5 seconds after launch, then every 4 hours
    const timeout = setTimeout(checkForUpdate, 5000);
    const interval = setInterval(checkForUpdate, 4 * 60 * 60 * 1000);
    return () => {
      clearTimeout(timeout);
      clearInterval(interval);
    };
  });

  async function checkForUpdate() {
    if (checking || downloading) return;
    checking = true;
    error = null;
    try {
      const result = await check();
      if (result) {
        update = result;
        dismissed = false;
      }
    } catch (e) {
      // Silently fail on check errors — don't nag users
      console.warn('Update check failed:', e);
    } finally {
      checking = false;
    }
  }

  async function installUpdate() {
    if (!update) return;
    downloading = true;
    error = null;
    downloadedBytes = 0;
    totalBytes = 0;
    progress = 0;

    try {
      await update.downloadAndInstall((event) => {
        switch (event.event) {
          case 'Started':
            totalBytes = event.data.contentLength ?? 0;
            break;
          case 'Progress':
            downloadedBytes += event.data.chunkLength;
            if (totalBytes > 0) {
              progress = Math.round((downloadedBytes / totalBytes) * 100);
            }
            break;
          case 'Finished':
            progress = 100;
            break;
        }
      });
      // Relaunch after install
      await relaunch();
    } catch (e) {
      error = e instanceof Error ? e.message : String(e);
      downloading = false;
    }
  }

  function formatBytes(bytes: number): string {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
  }
</script>

{#if update && !dismissed}
  <div class="fixed bottom-4 right-4 z-50 w-96 bg-white dark:bg-surface-800 rounded-xl shadow-2xl border border-surface-200 dark:border-surface-700 overflow-hidden">
    <!-- Header -->
    <div class="flex items-center justify-between px-4 py-3 bg-primary-50 dark:bg-primary-900/30 border-b border-surface-200 dark:border-surface-700">
      <div class="flex items-center gap-2">
        <svg class="w-5 h-5 text-primary-600 dark:text-primary-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
        </svg>
        <span class="font-semibold text-sm text-surface-900 dark:text-surface-100">Update Available</span>
      </div>
      {#if !downloading}
        <button
          class="text-surface-400 hover:text-surface-600 dark:hover:text-surface-300 transition-colors"
          onclick={() => (dismissed = true)}
          aria-label="Dismiss update notification"
        >
          <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
      {/if}
    </div>

    <!-- Body -->
    <div class="px-4 py-3 space-y-3">
      <p class="text-sm text-surface-700 dark:text-surface-300">
        PMetal <span class="font-semibold">{update.version}</span> is available.
      </p>

      {#if update.body}
        <div class="text-xs text-surface-500 dark:text-surface-400 max-h-24 overflow-y-auto scrollbar-thin">
          {update.body}
        </div>
      {/if}

      {#if error}
        <p class="text-xs text-red-600 dark:text-red-400">{error}</p>
      {/if}

      {#if downloading}
        <!-- Progress bar -->
        <div class="space-y-1">
          <div class="w-full bg-surface-200 dark:bg-surface-700 rounded-full h-2">
            <div
              class="bg-primary-500 h-2 rounded-full transition-all duration-300"
              style="width: {progress}%"
            ></div>
          </div>
          <div class="flex justify-between text-xs text-surface-500">
            <span>{progress}%</span>
            {#if totalBytes > 0}
              <span>{formatBytes(downloadedBytes)} / {formatBytes(totalBytes)}</span>
            {/if}
          </div>
        </div>
      {:else}
        <div class="flex gap-2">
          <button class="btn-primary btn-sm flex-1" onclick={installUpdate}>
            Download & Install
          </button>
          <button class="btn-secondary btn-sm" onclick={() => (dismissed = true)}>
            Later
          </button>
        </div>
      {/if}
    </div>
  </div>
{/if}
