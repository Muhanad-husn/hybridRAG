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
                
                // Only update if it's a new message
                if (latestLog !== statusElement.textContent) {
                    statusElement.textContent = latestLog;
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

    function updateOperationStatus(message) {
        const statusElement = document.getElementById('operationStatus');
        const timestamp = new Date().toLocaleTimeString();
        statusElement.textContent = `[${timestamp}] ${message}`;
    }

    // Confidence indicator management
    function updateConfidence(confidence) {
        const confidenceIndicator = document.getElementById('confidenceIndicator');
        const confidenceFill = confidenceIndicator.querySelector('.confidence-fill');
        const confidencePercentage = confidenceIndicator.querySelector('.confidence-percentage');
        
        if (confidenceFill && confidencePercentage) {
            // Convert confidence to percentage if needed
            const confidenceValue = typeof confidence === 'number' ? confidence :
                                  typeof confidence === 'string' ? parseFloat(confidence) : 0;
            
            // Ensure value is between 0-100
            const percentage = Math.min(Math.max(confidenceValue, 0), 100);
            
            confidenceFill.style.width = `${percentage}%`;
            confidencePercentage.textContent = `${Math.round(percentage)}%`;
        }
    }

    // View switching
    const navButtons = document.querySelectorAll('.nav-btn');
    const views = document.querySelectorAll('.view');

    navButtons.forEach(button => {
        button.addEventListener('click', () => {
            const viewName = button.dataset.view;
            
            // Update navigation buttons
            navButtons.forEach(btn => btn.classList.toggle('active', btn === button));
            
            // Update views
            views.forEach(view => {
                view.classList.toggle('active', view.id === `${viewName}View`);
                view.classList.toggle('hidden', view.id !== `${viewName}View`);
            });

            // Load view-specific content
            if (viewName === 'history') {
                loadSearchHistory();
            } else if (viewName === 'saved') {
                loadSavedResults();
            }
        });
    });

    // Load and display search history
    async function loadSearchHistory() {
        try {
            updateOperationStatus('Loading search history...');
            const response = await fetch('/search-history');
            const data = await response.json();
            
            if (historyList) {
                historyList.innerHTML = '';
                data.history.forEach(query => {
                    const item = document.createElement('div');
                    item.className = 'history-item';
                    
                    const queryText = document.createElement('span');
                    queryText.className = 'query-text';
                    queryText.textContent = query;
                    
                    const rerunButton = document.createElement('button');
                    rerunButton.className = 'rerun-btn';
                    rerunButton.textContent = 'Rerun';
                    rerunButton.onclick = () => {
                        queryInput.value = query;
                        searchForm.dispatchEvent(new Event('submit'));
                        document.querySelector('[data-view="search"]').click();
                    };
                    
                    item.appendChild(queryText);
                    item.appendChild(rerunButton);
                    historyList.appendChild(item);
                });
                updateOperationStatus('Search history loaded successfully');
            }
        } catch (error) {
            console.error('[History] Error:', error);
            updateOperationStatus('Error loading search history');
        }
    }

    // Filter history based on search input
    if (historySearch) {
        historySearch.addEventListener('input', (e) => {
            const searchTerm = e.target.value.toLowerCase();
            const historyItems = document.querySelectorAll('.history-item');
            
            historyItems.forEach(item => {
                const queryText = item.querySelector('.query-text').textContent.toLowerCase();
                item.style.display = queryText.includes(searchTerm) ? 'flex' : 'none';
            });
        });
    }

    // Load saved results
    async function loadSavedResults() {
        try {
            updateOperationStatus('Loading saved results...');
            const response = await fetch('/saved-results');
            const data = await response.json();
            
            const savedResultsGrid = document.querySelector('.saved-results-grid');
            if (savedResultsGrid) {
                savedResultsGrid.innerHTML = '';
                
                if (data.results && data.results.length > 0) {
                    data.results.forEach(filename => {
                        const resultCard = document.createElement('div');
                        resultCard.className = 'saved-result-card';
                        
                        const title = document.createElement('h3');
                        title.textContent = decodeURIComponent(filename.replace('.html', ''));
                        
                        const openButton = document.createElement('button');
                        openButton.className = 'open-result-btn';
                        openButton.textContent = 'Open Result';
                        openButton.onclick = () => {
                            window.location.href = `/results/${filename}`;
                        };
                        
                        resultCard.appendChild(title);
                        resultCard.appendChild(openButton);
                        savedResultsGrid.appendChild(resultCard);
                    });
                    updateOperationStatus('Saved results loaded successfully');
                } else {
                    savedResultsGrid.innerHTML = '<p>No saved results yet.</p>';
                    updateOperationStatus('No saved results found');
                }
            }
        } catch (error) {
            console.error('[Saved] Error:', error);
            updateOperationStatus('Error loading saved results');
            const savedResultsGrid = document.querySelector('.saved-results-grid');
            if (savedResultsGrid) {
                savedResultsGrid.innerHTML = '<p>Error loading saved results.</p>';
            }
        }
    }

    // Initialize response tabs
    function initializeResponseTabs() {
        const tabs = document.querySelectorAll('.response-tabs .tab-btn');
        const englishSection = document.getElementById('englishResponse');
        const arabicSection = document.getElementById('arabicResponse');
        const rawDataSection = document.getElementById('rawDataResponse');
        
        function switchTab(type) {
            if (!['en', 'ar', 'raw'].includes(type)) return;
            
            // Update tab buttons
            tabs.forEach(t => t.classList.toggle('active', t.dataset.type === type));
            
            // Update section visibility
            englishSection.classList.toggle('hidden', type !== 'en');
            englishSection.classList.toggle('active', type === 'en');
            arabicSection.classList.toggle('hidden', type !== 'ar');
            arabicSection.classList.toggle('active', type === 'ar');
            rawDataSection.classList.toggle('hidden', type !== 'raw');
            rawDataSection.classList.toggle('active', type === 'raw');
            
            // Update RTL/LTR and styles
            if (type === 'ar') {
                arabicSection.dir = 'rtl';
                arabicSection.style.textAlign = 'right';
                arabicSection.querySelector('.response-content').style.fontFamily = "'Noto Naskh Arabic', Arial, sans-serif";
            } else if (type === 'en') {
                englishSection.dir = 'ltr';
                englishSection.style.textAlign = 'left';
            }
        }

        tabs.forEach(tab => {
            tab.addEventListener('click', () => {
                const type = tab.dataset.type;
                if (type) switchTab(type);
            });
        });

        return switchTab;
    }

    const switchTab = initializeResponseTabs();
    
    // Save functionality
    function setupSaveButtons(data) {
        updateOperationStatus('Setting up save buttons...');
        const saveButtons = document.querySelectorAll('.save-btn');
        
        saveButtons.forEach(btn => {
            const lang = btn.dataset.lang;
            const filename = lang === 'en' ? data.english_file : data.arabic_file;
            
            // Remove existing listeners
            const newBtn = btn.cloneNode(true);
            btn.parentNode.replaceChild(newBtn, btn);
            
            if (filename) {
                newBtn.disabled = false;
                newBtn.title = lang === 'en' ? "Click to download HTML file" : "انقر للتحميل بصيغة HTML";
                
                newBtn.addEventListener('click', async (e) => {
                    e.preventDefault();
                    updateOperationStatus('Saving result...');
                    
                    try {
                        // First save the result to our saved results list
                        const saveResponse = await fetch('/save-result', {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json',
                            },
                            body: JSON.stringify({ filename })
                        });
                        
                        if (!saveResponse.ok) {
                            throw new Error('Failed to save result');
                        }
                        
                        // Then initiate the download
                        const downloadUrl = `/results/${filename}`;
                        window.location.href = downloadUrl;
                        updateOperationStatus('Result saved successfully');
                        
                        // Show success message
                        const successMessage = document.createElement('div');
                        successMessage.className = 'success-message';
                        successMessage.textContent = 'Result saved successfully';
                        document.body.appendChild(successMessage);
                        
                        setTimeout(() => {
                            successMessage.remove();
                        }, 3000);
                        
                    } catch (error) {
                        console.error('[Save] Error:', error);
                        updateOperationStatus('Error saving result');
                        const errorMessage = document.createElement('div');
                        errorMessage.className = 'error-message';
                        errorMessage.textContent = 'Failed to save result';
                        document.body.appendChild(errorMessage);
                        
                        setTimeout(() => {
                            errorMessage.remove();
                        }, 3000);
                    }
                });
            } else {
                newBtn.disabled = true;
                newBtn.title = lang === 'en' ? "No file available" : "الملف غير متوفر";
            }
        });
    }

    // Search form handler
    searchForm.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        const query = queryInput.value.trim();
        if (!query) return;

        errorDiv.classList.add('hidden');
        resultsDiv.classList.add('hidden');
        loadingIndicator.classList.remove('hidden');
        searchButton.disabled = true;

        // Start polling for logs
        lastLogTimestamp = '';
        startLogPolling();

        try {
            const response = await fetch('/search', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    query: query,
                    mode: 'hybrid',
                    translate: document.getElementById('translateToggle').checked,
                    rerank_count: Math.min(Math.max(parseInt(document.getElementById('resultsCount').value) || 15, 5), 80)
                })
            });

            const data = await response.json();

            if (response.ok) {
                // Update confidence if available
                if (data.confidence !== undefined) {
                    updateConfidence(data.confidence); // Already in percentage form
                }
                displayResults(data);
            } else {
                console.error('[Search] Error:', data.error);
                throw new Error(data.error || 'An error occurred while processing your query');
            }
        } catch (error) {
            console.error('[Search] Error:', error.message);
            updateOperationStatus(`Error: ${error.message}`);
            displayError(error.message);
        } finally {
            // Stop polling for logs
            stopLogPolling();
            setTimeout(() => {
                loadingIndicator.classList.add('hidden');
                searchButton.disabled = false;
            }, 500);
        }
    });

    // Display results
    async function displayResults(data) {
        if (data.error) {
            displayError(data.error);
            return;
        }

        // Clear previous content
        document.querySelectorAll('.response-content').forEach(content => {
            content.textContent = '';
        });
        sourcesList.innerHTML = '';

        // Display English response
        const englishContent = document.querySelector('#englishResponse .response-content');
        if (englishContent && data.answer) {
            englishContent.textContent = data.answer;
        }

        // Handle Arabic tab visibility and content
        const arabicTab = document.querySelector('.tab-btn[data-type="ar"]');
        const arabicContent = document.querySelector('#arabicResponse .response-content');
        const translateEnabled = document.getElementById('translateToggle').checked;

        // Show/hide Arabic tab based on translation setting
        arabicTab.style.display = translateEnabled ? 'block' : 'none';
        
        if (arabicContent) {
            if (translateEnabled && data.arabic_answer) {
                arabicContent.textContent = data.arabic_answer;
                arabicContent.dir = 'rtl';
                arabicContent.style.textAlign = 'right';
                arabicContent.style.fontFamily = "'Noto Naskh Arabic', Arial, sans-serif";
                arabicContent.style.color = 'var(--text-primary)';
            } else if (!translateEnabled) {
                arabicContent.textContent = 'الترجمة العربية معطلة';
                arabicContent.dir = 'rtl';
                arabicContent.style.textAlign = 'right';
                arabicContent.style.color = '#666';
            } else {
                arabicContent.textContent = 'المحتوى العربي غير متوفر';
                arabicContent.dir = 'rtl';
                arabicContent.style.textAlign = 'right';
                arabicContent.style.color = '#666';
            }
        }

        // Switch to English tab if Arabic is disabled and we're on Arabic tab
        if (!translateEnabled && document.querySelector('.tab-btn[data-type="ar"]').classList.contains('active')) {
            switchTab('en');
        }

        // Setup save buttons with file information
        setupSaveButtons(data);

        // Display raw data
        const vectorContent = document.querySelector('.vector-content');
        if (vectorContent && data.llm_input) {
            try {
                // Extract only the context section
                const context = data.llm_input.context || 'No context available';
                const formattedInput = context.split('\n')
                    .filter(line => line.trim()) // Remove empty lines
                    .join('\n');
                vectorContent.textContent = formattedInput;
            } catch (error) {
                console.error('[Display] Error:', error);
                vectorContent.textContent = 'Error displaying LLM input data';
            }
        } else {
            vectorContent.textContent = 'No LLM input data available';
        }

        // Display sources
        if (data.sources && data.sources.length > 0) {
            data.sources.forEach(source => {
                const li = document.createElement('li');
                li.textContent = source;
                sourcesList.appendChild(li);
            });
        }

        // Update stats if available
        if (data.stats) {
            document.getElementById('docCount').textContent = data.stats.documents || 0;
            document.getElementById('nodeCount').textContent = data.stats.nodes || 0;
        }

        // Show results and switch to appropriate tab
        resultsDiv.classList.remove('hidden');
        errorDiv.classList.add('hidden');
        switchTab(data.language === 'ar' ? 'ar' : 'en');

        // Continue polling for logs during result processing
        await new Promise(resolve => setTimeout(resolve, 2000));
    }

    // Error display
    function displayError(message) {
        const errorMessage = errorDiv.querySelector('.error-message');
        errorMessage.textContent = message;
        errorDiv.classList.remove('hidden');
        resultsDiv.classList.add('hidden');
        updateOperationStatus(`Error: ${message}`);
    }

    // Process Files functionality
    const processFilesBtn = document.getElementById('processFilesBtn');
    if (processFilesBtn) {
        processFilesBtn.addEventListener('click', async () => {
            // Show warning dialog
            if (!confirm('Warning: Processing files will reset all existing data in both vector store and graph store. This action cannot be undone. Do you want to continue?')) {
                return;
            }

            try {
                // Show loading state
                loadingIndicator.classList.remove('hidden');
                resultsDiv.classList.add('hidden');
                errorDiv.classList.add('hidden');
                processFilesBtn.disabled = true;

                // Add progress bar if it doesn't exist
                let progressBar = document.getElementById('processProgress');
                if (!progressBar) {
                    progressBar = document.createElement('div');
                    progressBar.id = 'processProgress';
                    progressBar.className = 'progress-bar';
                    progressBar.innerHTML = '<div class="progress-fill"></div>';
                    loadingIndicator.insertBefore(progressBar, document.getElementById('operationStatus'));
                }

                // Reset progress
                const progressFill = progressBar.querySelector('.progress-fill');
                progressFill.style.width = '0%';
                
                updateOperationStatus('Initializing file processing...');
                startLogPolling();

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
                    progressFill.style.width = '100%';
                    updateOperationStatus('Files processed successfully!');

                    // Show stats
                    const statsDiv = document.createElement('div');
                    statsDiv.className = 'process-stats';
                    statsDiv.innerHTML = `
                        <h3>Processing Complete</h3>
                        <p>Documents in Vector Store: ${data.vector_count || 0}</p>
                        <p>Graph Nodes: ${data.node_count || 0}</p>
                        <p>Graph Edges: ${data.edge_count || 0}</p>
                    `;
                    loadingIndicator.appendChild(statsDiv);

                    // Auto-navigate to search after 5 seconds
                    setTimeout(() => {
                        // Remove stats and progress bar
                        statsDiv.remove();
                        progressBar.remove();
                        loadingIndicator.classList.add('hidden');
                        // Navigate to search view
                        document.querySelector('[data-view="search"]').click();
                    }, 5000);
                } else {
                    throw new Error(data.error || 'Failed to process files');
                }
            } catch (error) {
                console.error('[Process Files] Error:', error);
                displayError(error.message);
                processFilesBtn.disabled = false;
                loadingIndicator.classList.add('hidden');
            } finally {
                stopLogPolling();
            }
        });
    }

    // Set initial tab
    switchTab('en');
});