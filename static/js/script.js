document.addEventListener('DOMContentLoaded', function() {
    // DOM Elements
    const searchForm = document.getElementById('searchForm');
    const apiKeyForm = document.getElementById('apiKeyForm');
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

    // File Upload functionality
    const loadFilesBtn = document.getElementById('loadFilesBtn');
    const processFilesBtn = document.getElementById('processFilesBtn');
    const fileInput = document.getElementById('fileInput');

    if (loadFilesBtn && fileInput) {
        // Trigger file input when Load Files button is clicked
        loadFilesBtn.addEventListener('click', () => {
            fileInput.click();
        });

        // Handle file selection
        fileInput.addEventListener('change', async (event) => {
            if (!event.target.files.length) return;

            try {
                // Reset log state and start polling
                lastLogTimestamp = '';
                loadingIndicator.classList.remove('hidden');
                startLogPolling();
                loadFilesBtn.disabled = true;
                processFilesBtn.disabled = true;
                
                // Clear any previous content
                const statusElement = document.getElementById('operationStatus');
                statusElement.textContent = 'Uploading files...';

                // Upload the files
                const formData = new FormData();
                for (const file of event.target.files) {
                    formData.append('files', file);
                }

                const uploadResponse = await fetch('/upload-files', {
                    method: 'POST',
                    body: formData
                });

                if (!uploadResponse.ok) {
                    const uploadData = await uploadResponse.json();
                    throw new Error(uploadData.error || 'Failed to upload files');
                }

                statusElement.textContent = 'Files uploaded successfully';
            } catch (error) {
                console.error('[Load Files] Error:', error);
                displayError(error.message);
            } finally {
                stopLogPolling();
                loadFilesBtn.disabled = false;
                processFilesBtn.disabled = false;
                loadingIndicator.classList.add('hidden');
                fileInput.value = ''; // Clear file input
            }
        });
    }

    // Process Files functionality
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
                loadFilesBtn.disabled = true;
                processFilesBtn.disabled = true;
                
                // Clear any previous content
                const statusElement = document.getElementById('operationStatus');
                statusElement.textContent = 'Processing files...';

                const processResponse = await fetch('/process-documents', {
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

                const processData = await processResponse.json();

                if (processResponse.ok) {
                    // Wait 5 seconds to show final logs
                    await new Promise(resolve => setTimeout(resolve, 5000));
                    
                    // Update stats and navigate to search
                    document.getElementById('docCount').textContent = processData.vector_count || 0;
                    document.getElementById('nodeCount').textContent = processData.node_count || 0;
                    document.querySelector('[data-view="search"]').click();
                } else {
                    throw new Error(processData.error || 'Failed to process files');
                }
            } catch (error) {
                console.error('[Process Files] Error:', error);
                displayError(error.message);
            } finally {
                stopLogPolling();
                loadFilesBtn.disabled = false;
                processFilesBtn.disabled = false;
                loadingIndicator.classList.add('hidden');
            }
        });
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
        }

        tabs.forEach(tab => {
            tab.addEventListener('click', () => {
                const type = tab.dataset.type;
                if (type) switchTab(type);
            });
        });

        // Set initial tab
        switchTab('en');
    }
// Initialize tabs
initializeResponseTabs();

// View switching
const navButtons = document.querySelectorAll('.nav-btn');
const views = document.querySelectorAll('.view');

// Function to fetch and display saved results
async function updateSavedResultsView() {
    const savedResultsGrid = document.querySelector('.saved-results-grid');
    if (!savedResultsGrid) return;

    try {
        const response = await fetch('/saved-results');
        const data = await response.json();

        savedResultsGrid.innerHTML = '';

        if (data.results && data.results.length > 0) {
            data.results.forEach(filename => {
                const resultCard = document.createElement('div');
                resultCard.className = 'result-card';
                
                const downloadLink = document.createElement('a');
                downloadLink.href = `/results/${filename}`;
                downloadLink.textContent = filename;
                downloadLink.download = filename;
                
                resultCard.appendChild(downloadLink);
                savedResultsGrid.appendChild(resultCard);
            });
        } else {
            savedResultsGrid.innerHTML = '<p>No saved results yet</p>';
        }
    } catch (error) {
        console.error('[Saved Results] Error:', error);
        savedResultsGrid.innerHTML = '<p>Error loading saved results</p>';
    }
}

// Function to fetch and display search history
async function updateSearchHistoryView() {
    const historyList = document.querySelector('.history-list');
    const historySearch = document.querySelector('.history-search');
    if (!historyList) return;

    try {
        const response = await fetch('/search-history');
        const data = await response.json();

        function renderHistory(queries) {
            historyList.innerHTML = '';
            if (queries && queries.length > 0) {
                queries.forEach(query => {
                    const historyItem = document.createElement('div');
                    historyItem.className = 'history-item';
                    
                    const queryText = document.createElement('span');
                    queryText.textContent = query;
                    
                    const rerunButton = document.createElement('button');
                    rerunButton.textContent = 'Search Again';
                    rerunButton.className = 'rerun-btn';
                    rerunButton.onclick = () => {
                        // Fill the search input and submit
                        queryInput.value = query;
                        document.querySelector('[data-view="search"]').click();
                        setTimeout(() => searchForm.dispatchEvent(new Event('submit')), 100);
                    };
                    
                    historyItem.appendChild(queryText);
                    historyItem.appendChild(rerunButton);
                    historyList.appendChild(historyItem);
                });
            } else {
                historyList.innerHTML = '<p>No search history yet</p>';
            }
        }

        // Initial render
        renderHistory(data.history);

        // Setup search filter
        if (historySearch) {
            historySearch.oninput = (e) => {
                const searchTerm = e.target.value.toLowerCase();
                const filteredHistory = data.history.filter(query =>
                    query.toLowerCase().includes(searchTerm)
                );
                renderHistory(filteredHistory);
            };
        }

    } catch (error) {
        console.error('[Search History] Error:', error);
        historyList.innerHTML = '<p>Error loading search history</p>';
    }
}

navButtons.forEach(button => {
    button.addEventListener('click', () => {
        const viewName = button.dataset.view;
        if (!viewName) return;

        // Update active states
        navButtons.forEach(btn => btn.classList.toggle('active', btn === button));
        views.forEach(view => {
            view.classList.toggle('active', view.id === `${viewName}View`);
            view.classList.toggle('hidden', view.id !== `${viewName}View`);
        });

        // Update content based on view
        if (viewName === 'saved') {
            updateSavedResultsView();
        } else if (viewName === 'history') {
            updateSearchHistoryView();
        }
    });
});

// API Key form handler
if (apiKeyForm) {
    apiKeyForm.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        const apiKeyInput = document.getElementById('apiKeyInput');
        const messageDiv = document.getElementById('apiKeyMessage');
        const submitButton = apiKeyForm.querySelector('button[type="submit"]');
        
        const apiKey = apiKeyInput.value.trim();
        if (!apiKey) return;

        try {
            submitButton.disabled = true;
            messageDiv.textContent = 'Updating API key...';
            messageDiv.className = 'settings-message';

            const response = await fetch('/update-api-key', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ api_key: apiKey })
            });

            const data = await response.json();

            if (response.ok) {
                messageDiv.textContent = 'API key updated successfully';
                messageDiv.className = 'settings-message success';
                apiKeyInput.value = ''; // Clear the input
            } else {
                throw new Error(data.error || 'Failed to update API key');
            }
        } catch (error) {
            console.error('[API Key Update] Error:', error);
            messageDiv.textContent = error.message;
            messageDiv.className = 'settings-message error';
        } finally {
            submitButton.disabled = false;
        }
    });
}

// Search form handler
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
                    translate: document.getElementById('translateToggle').checked,
                    rerank_count: Math.min(Math.max(parseInt(document.getElementById('resultsCount').value) || 15, 5), 80)
                })
            });

            const data = await response.json();

            if (response.ok) {
                loadingIndicator.classList.add('hidden');
                resultsDiv.classList.remove('hidden');

                // Update confidence if available
                if (data.confidence !== undefined) {
                    updateConfidence(data.confidence);
                }
                
                // Display English response
                const englishContent = document.querySelector('#englishResponse .response-content');
                if (englishContent && data.answer) {
                    englishContent.textContent = data.answer;
                }

                // Display Arabic response
                const arabicContent = document.querySelector('#arabicResponse .response-content');
                if (arabicContent) {
                    if (data.arabic_answer) {
                        arabicContent.textContent = data.arabic_answer;
                        arabicContent.dir = 'rtl';
                        arabicContent.style.textAlign = 'right';
                        arabicContent.style.fontFamily = "'Noto Naskh Arabic', Arial, sans-serif";
                    } else {
                        arabicContent.textContent = 'المحتوى العربي غير متوفر';
                        arabicContent.dir = 'rtl';
                        arabicContent.style.textAlign = 'right';
                    }
                }

                // Display raw data
                const vectorContent = document.querySelector('.vector-content');
                if (vectorContent && data.llm_input) {
                    try {
                        const context = data.llm_input.context || 'No context available';
                        const formattedInput = context.split('\n')
                            .filter(line => line.trim())
                            .join('\n');
                        vectorContent.textContent = formattedInput;
                    } catch (error) {
                        console.error('[Display] Error:', error);
                        vectorContent.textContent = 'Error displaying raw data';
                    }
                }

                // Update sources
                sourcesList.innerHTML = '';
                if (data.sources) {
                    data.sources.forEach(source => {
                        const li = document.createElement('li');
                        li.textContent = source;
                        sourcesList.appendChild(li);
                    });
                }
            } else {
                throw new Error(data.error || 'An error occurred while processing your query');
            }
        } catch (error) {
            console.error('[Search] Error:', error);
            displayError(error.message);
        } finally {
            // Stop polling for logs
            stopLogPolling();
            searchButton.disabled = false;
        }
    });

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

    // Error display
    function displayError(message) {
        const errorMessage = errorDiv.querySelector('.error-message');
        errorMessage.textContent = message;
        errorDiv.classList.remove('hidden');
        resultsDiv.classList.add('hidden');
    }

    // Save button handlers
    document.querySelectorAll('.save-btn').forEach(button => {
        button.addEventListener('click', async () => {
            const lang = button.dataset.lang;
            const responseSection = document.getElementById(`${lang === 'en' ? 'english' : 'arabic'}Response`);
            const content = responseSection.querySelector('.response-content').textContent;
            
            try {
                // Get the query and translated query
                const query = queryInput.value.trim();
                const translatedQuery = lang === 'ar' ? query : ''; // For Arabic, use original query
                
                // Get sources
                const sources = Array.from(sourcesList.children).map(li => li.textContent);
                
                // Generate HTML content
                const result = await fetch('/generate-result', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        content: content,
                        query: query,
                        translatedQuery: translatedQuery,
                        sources: sources,
                        isArabic: lang === 'ar'
                    })
                });
                
                const htmlData = await result.json();
                
                if (!result.ok) {
                    throw new Error(htmlData.error || 'Failed to generate HTML');
                }

                // Save to server for saved results panel
                const saveResponse = await fetch('/save-result', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        content: content,
                        filename: htmlData.filename,
                        html: htmlData.html
                    })
                });
                
                const saveData = await saveResponse.json();
                
                if (!saveResponse.ok) {
                    throw new Error(saveData.error || 'Failed to save result');
                }

                // Create a blob from the HTML content for browser download
                const blob = new Blob([htmlData.html], { type: 'text/html;charset=utf-8' });
                
                // Create a temporary link to trigger download
                const downloadLink = document.createElement('a');
                downloadLink.href = URL.createObjectURL(blob);
                downloadLink.download = htmlData.filename;
                
                // Append link to body, click it, and remove it
                document.body.appendChild(downloadLink);
                downloadLink.click();
                document.body.removeChild(downloadLink);
                
                // Clean up the URL object
                URL.revokeObjectURL(downloadLink.href);

                // Update saved results list if on saved results view
                const savedView = document.getElementById('savedView');
                if (!savedView.classList.contains('hidden')) {
                    // Refresh saved results view
                    document.querySelector('[data-view="saved"]').click();
                }
            } catch (error) {
                console.error('[Save Result] Error:', error);
                alert('Failed to save result: ' + error.message);
            }
        });
    });
});