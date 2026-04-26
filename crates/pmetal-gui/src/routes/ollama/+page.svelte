<script lang="ts">
  import { startOllama, type OllamaAction } from '$lib/api';

  const actions: { value: OllamaAction; label: string; description: string; needsModel: boolean }[] = [
    { value: 'install', label: 'Install', description: 'Install the Ollama runtime (if not already present)', needsModel: false },
    { value: 'pull',    label: 'Pull',    description: 'Download a model from the Ollama library',           needsModel: true  },
    { value: 'run',     label: 'Run',     description: 'Start an interactive session with a model',          needsModel: true  },
    { value: 'list',    label: 'List',    description: 'List all locally available Ollama models',           needsModel: false },
  ];

  // Form state
  let action = $state<OllamaAction>('pull');
  let model = $state('llama3');

  // UI state
  let isRunning = $state(false);
  let runId = $state<string | null>(null);
  let status = $state<'idle' | 'running' | 'done' | 'failed'>('idle');
  let formError = $state<string | null>(null);
  let logs = $state<string[]>([]);

  let needsModel = $derived(actions.find(a => a.value === action)?.needsModel ?? true);

  function appendLog(line: string) {
    logs = [...logs.slice(-499), line];
  }

  async function handleSubmit(e: Event) {
    e.preventDefault();
    formError = null;
    logs = [];
    status = 'idle';

    if (needsModel && !model.trim()) {
      formError = 'Please enter a model name';
      return;
    }

    isRunning = true;
    status = 'running';

    try {
      runId = await startOllama(action, needsModel ? model.trim() : '', (e: Record<string, unknown>) => {
        const evt = e as { event?: string; line?: string };
        if (evt.event === 'log' && typeof evt.line === 'string') {
          appendLog(evt.line);
        } else if (evt.event === 'done') {
          status = 'done';
          isRunning = false;
        } else if (evt.event === 'failed') {
          status = 'failed';
          isRunning = false;
        }
      });
    } catch (err) {
      formError = err instanceof Error ? err.message : String(err);
      status = 'failed';
      isRunning = false;
    }
  }
</script>

<div class="space-y-6 max-w-2xl">
  <!-- Header -->
  <div>
    <h1 class="text-2xl font-bold text-surface-900 dark:text-surface-100">Ollama</h1>
    <p class="text-surface-500 dark:text-surface-400 mt-1">Manage and run Ollama models via pmetal</p>
  </div>

  <form onsubmit={handleSubmit} class="space-y-4">
    <!-- Action picker -->
    <div class="card">
      <div class="card-header">
        <h3 class="font-semibold text-surface-900 dark:text-surface-100">Action</h3>
      </div>
      <div class="card-body grid grid-cols-2 gap-2">
        {#each actions as a}
          <button
            type="button"
            class="p-3 rounded-lg border text-left transition-all {action === a.value
              ? 'border-primary-500 bg-primary-50 dark:bg-primary-900/30'
              : 'border-surface-200 dark:border-surface-700 hover:border-surface-300 dark:hover:border-surface-600'}"
            onclick={() => (action = a.value)}
          >
            <p class="text-sm font-semibold text-surface-900 dark:text-surface-100">{a.label}</p>
            <p class="text-xs text-surface-500 mt-0.5">{a.description}</p>
          </button>
        {/each}
      </div>
    </div>

    <!-- Model input -->
    {#if needsModel}
      <div class="card">
        <div class="card-header">
          <h3 class="font-semibold text-surface-900 dark:text-surface-100">Model</h3>
        </div>
        <div class="card-body">
          <label class="label" for="ollama-model">Model Name</label>
          <input
            id="ollama-model"
            type="text"
            class="input"
            placeholder="e.g. llama3, mistral, phi3:mini"
            bind:value={model}
          />
          <p class="text-xs text-surface-500 mt-1">Ollama model tag — see <a href="https://ollama.com/library" target="_blank" rel="noreferrer" class="underline">ollama.com/library</a> for the full list.</p>
        </div>
      </div>
    {/if}

    <!-- Status -->
    {#if status === 'running'}
      <div class="p-4 rounded-lg bg-primary-50 dark:bg-primary-900/20 border border-primary-200 dark:border-primary-800 text-primary-700 dark:text-primary-300 text-sm flex items-center gap-2" role="status">
        <div class="w-4 h-4 border-2 border-primary-500 border-t-transparent rounded-full animate-spin flex-shrink-0" aria-hidden="true"></div>
        Running… Run ID: {runId}
      </div>
    {/if}
    {#if status === 'done'}
      <div class="p-4 rounded-lg bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 text-green-700 dark:text-green-300 text-sm" role="status">
        Command completed successfully.
      </div>
    {/if}
    {#if status === 'failed'}
      <div class="p-4 rounded-lg bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 text-red-700 dark:text-red-300 text-sm" role="alert">
        Command failed. Check the output log below.
      </div>
    {/if}
    {#if formError}
      <div class="p-4 rounded-lg bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 text-red-700 dark:text-red-300 text-sm" role="alert">
        {formError}
      </div>
    {/if}

    <button type="submit" class="btn-primary w-full" disabled={isRunning || (needsModel && !model.trim())}>
      {#if isRunning}
        <div class="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" aria-hidden="true"></div>
        Running {action}...
      {:else}
        <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" />
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
        </svg>
        Run: ollama {action}{needsModel ? ` ${model || '<model>'}` : ''}
      {/if}
    </button>
  </form>

  <!-- Output log -->
  {#if logs.length > 0}
    <div class="card">
      <div class="card-header">
        <h3 class="font-semibold text-surface-900 dark:text-surface-100">Output</h3>
      </div>
      <div class="card-body">
        <pre class="text-xs font-mono text-surface-700 dark:text-surface-300 bg-surface-50 dark:bg-surface-900 rounded p-3 max-h-80 overflow-y-auto whitespace-pre-wrap">{logs.join('\n')}</pre>
      </div>
    </div>
  {/if}
</div>
