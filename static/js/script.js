document.addEventListener('DOMContentLoaded', function() {
    // DOM Elements
    const searchForm = document.getElementById('searchForm');
    const queryInput = document.getElementById('queryInput');
    const searchButton = document.getElementById('searchButton');
    const loadingIndicator = document.getElementById('loadingIndicator');
    const resultsDiv = document.getElementById('results');
    const englishResponse = document.getElementById('englishResponse');
    const arabicResponse = document.getElementById('arabicResponse');
    const errorDiv = document.getElementById('error');
    const sourcesList = document.querySelector('.sources-list');
    const historyList = document.querySelector('.history-list');
    const historySearch = document.querySelector('.history-search');
    const historySort = document.querySelector('.history-sort');

    // Operation status management
    let lastLogTimestamp = '';

    async function fetchAndDisplayLogs() {
        try {
            const response = await fetch('/logs');
            const data = await response.json();
            
            if (data.logs && data.logs.length > 0) {
                const statusElement = document.getElementById('operationStatus');
                const latestLog = data.logs[data.logs.length - 1];
                
                // Only update if it's a new message and different from last timestamp
                if (latestLog !== lastLogTimestamp) {
                    lastLogTimestamp = latestLog;
                    statusElement.textContent = latestLog;
                    console.log('Log updated:', latestLog); // Debug log
                }
            }
        } catch (error) {
            console.error('Error fetching logs:', error);
        }
    }

    // Poll for log updates during operations
    let logPollingInterval = null;

    function startLogPolling() {
        if (!logPollingInterval) {
            logPollingInterval = setInterval(fetchAndDisplayLogs, 500);
        }
    }

    function stopLogPolling() {
        if (logPollingInterval) {
            clearInterval(logPollingInterval);
            logPollingInterval = null;
        }
    }

    // Process Files functionality
    const processFilesBtn = document.getElementById('processFilesBtn');
    if (processFilesBtn) {
        processFilesBtn.addEventListener('click', async () => {
            // Show warning dialog
            if (!confirm('Warning: Processing files will reset all existing data. Do you want to continue?')) {
                return;
            }

            try {
                // Reset log state and start polling
                lastLogTimestamp = '';
                loadingIndicator.classList.remove('hidden');
                startLogPolling();
                processFilesBtn.disabled = true;
                
                // Clear any previous content
                const statusElement = document.getElementById('operationStatus');
                statusElement.textContent = 'Starting file processing...';

                const response = await fetch('/process-documents', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        input_dir: 'data/raw_documents',
                        save_chunks: true,
                        save_embeddings: true
                    })
                });

                const data = await response.json();

                if (response.ok) {
                    // Wait 5 seconds to show final logs
                    await new Promise(resolve => setTimeout(resolve, 5000));
                    
                    // Update stats and navigate to search
                    document.getElementById('docCount').textContent = data.vector_count || 0;
                    document.getElementById('nodeCount').textContent = data.node_count || 0;
                    document.querySelector('[data-view="search"]').click();
                } else {
                    throw new Error(data.error || 'Failed to process files');
                }
            } catch (error) {
                console.error('[Process Files] Error:', error);
                displayError(error.message);
            } finally {
                stopLogPolling();
                processFilesBtn.disabled = false;
                loadingIndicator.classList.add('hidden');
            }
        });
    }

    // Error display
    function displayError(message) {
        const errorMessage = errorDiv.querySelector('.error-message');
        errorMessage.textContent = message;
        errorDiv.classList.remove('hidden');
        resultsDiv.classList.add('hidden');
    }
});