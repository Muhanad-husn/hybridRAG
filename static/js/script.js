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
    const navButtons = document.querySelectorAll('.nav-btn');
    const views = document.querySelectorAll('.view');
    const searchModeTabs = document.querySelectorAll('.search-mode-tabs .tab-btn');
    const langTabs = document.querySelectorAll('.response-tabs .tab-btn');
    const processSteps = document.querySelectorAll('.process-steps .step');

    // State Management
    let searchHistory = JSON.parse(localStorage.getItem('searchHistory') || '[]');
    let currentSearchMode = 'hybrid';
    let currentResultType = 'hybrid';
    let currentLanguage = 'en';

    // Initialize search mode tabs
    searchModeTabs.forEach(tab => {
        tab.addEventListener('click', () => {
            searchModeTabs.forEach(t => t.classList.remove('active'));
            tab.classList.add('active');
            currentSearchMode = tab.dataset.mode;
        });
    });

    // Initialize response tabs
    document.querySelectorAll('.response-tabs .tab-btn').forEach(tab => {
        tab.addEventListener('click', () => {
            // Update active state for all tabs
            document.querySelectorAll('.response-tabs .tab-btn').forEach(t => t.classList.remove('active'));
            tab.classList.add('active');

            const type = tab.dataset.type;
            if (type) {
                // Show corresponding response section
                document.querySelectorAll('.response-section').forEach(section => {
                    const isActive = section.id === `${type}Response`;
                    section.classList.toggle('active', isActive);
                    section.classList.toggle('hidden', !isActive);
                });

                // Update current type
                if (type === 'en' || type === 'ar') {
                    currentLanguage = type;
                } else {
                    currentResultType = type;
                }
            }
        });
    });
    
    // Navigation Handler
    navButtons.forEach(button => {
        button.addEventListener('click', () => {
            const targetView = button.dataset.view;
            
            // Update active states
            navButtons.forEach(btn => btn.classList.remove('active'));
            button.classList.add('active');
            
            // Show target view
            views.forEach(view => {
                view.classList.toggle('active', view.id === `${targetView}View`);
                view.classList.toggle('hidden', view.id !== `${targetView}View`);
            });

            // Load view-specific content
            if (targetView === 'history') {
                displaySearchHistory();
            } else if (targetView === 'saved') {
                displaySavedResults();
            } else if (targetView === 'tutorial') {
                loadTutorial();
            }
        });
    });

    // Search Form Handler
    searchForm.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        const query = queryInput.value.trim();
        if (!query) return;

        // Reset UI state
        errorDiv.classList.add('hidden');
        resultsDiv.classList.add('hidden');
        loadingIndicator.classList.remove('hidden');
        searchButton.disabled = true;

        // Start process animation
        startProcessAnimation();

        try {
            const response = await fetch('/search', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    query: query,
                    mode: currentSearchMode
                })
            });

            const data = await response.json();

            if (response.ok) {
                // Add to search history
                addToSearchHistory({
                    query: query,
                    timestamp: new Date().toISOString(),
                    result: data
                });
                
                displayResults(data);
            } else {
                throw new Error(data.error || 'An error occurred while processing your query');
            }
        } catch (error) {
            displayError(error.message);
        } finally {
            loadingIndicator.classList.add('hidden');
            searchButton.disabled = false;
            stopProcessAnimation();
        }
    });

    // Process Animation
    function startProcessAnimation() {
        processSteps.forEach(step => step.classList.remove('active'));
        
        // First step starts immediately
        processSteps[0].classList.add('active');
        
        // Second step starts after 1.5 seconds
        window.stepTimeout = setTimeout(() => {
            processSteps[0].classList.remove('active');
            processSteps[1].classList.add('active');
        }, 1500);
    }

    function stopProcessAnimation() {
        if (window.stepTimeout) {
            clearTimeout(window.stepTimeout);
        }
        processSteps.forEach(step => step.classList.remove('active'));
    }

    // Results Display
    function displayResults(data) {
        // Clear previous results
        document.querySelectorAll('.response-content').forEach(content => {
            content.textContent = '';
        });
        sourcesList.innerHTML = '';

        // Show results container
        resultsDiv.classList.remove('hidden');

        if (data.error) {
            displayError(data.error);
            return;
        }

        // Update English response
        const englishContent = document.querySelector('#englishResponse .response-content');
        if (englishContent) {
            englishContent.textContent = data.answer || '';
        }

        // Update Arabic response
        const arabicContent = document.querySelector('#arabicResponse .response-content');
        if (arabicContent) {
            arabicContent.textContent = data.arabic_answer || '';
        }

        // Update Dense Vector response
        const denseContent = document.querySelector('#denseResponse .response-content');
        if (denseContent) {
            denseContent.textContent = data.dense_results || data.dense?.answer || 'No dense vector results available';
        }

        // Update Knowledge Graph response
        const graphContent = document.querySelector('#graphResponse .response-content');
        if (graphContent) {
            graphContent.textContent = data.graph_results || data.graph?.answer || 'No knowledge graph results available';
        }

        // Handle confidence score
        const confidenceScore = document.querySelector('.confidence-score');
        if (confidenceScore) {
            const confidenceBar = confidenceScore.querySelector('.score-bar');
            const confidenceValue = confidenceScore.querySelector('.score-value');
            const score = '85%'; // Default confidence score
            confidenceBar.style.setProperty('--score', score);
            confidenceValue.textContent = score;
            confidenceScore.classList.remove('hidden');
        }

        // Show the current result type and language
        document.querySelectorAll('.response-section').forEach(section => {
            section.classList.toggle('active', section.id === `${currentResultType}Response`);
        });

        const currentResultSection = document.querySelector(`#${currentResultType}Response`);
        if (currentResultSection) {
            currentResultSection.querySelectorAll('.language-section').forEach(section => {
                section.classList.toggle('active',
                    section.id === `${currentResultType}${currentLanguage === 'en' ? 'English' : 'Arabic'}Response`);
            });
        }

        // Show results container
        resultsDiv.classList.remove('hidden');
        errorDiv.classList.add('hidden');
        
        // Set active tab based on query language
        const activeTab = data.language === 'ar' ? 'ar' : 'en';
        const tabToActivate = document.querySelector(`.tab-btn[data-type="${activeTab}"]`);
        if (tabToActivate) {
            tabToActivate.click();
        }

        // Re-initialize language tabs more safely
        const tabButtons = document.querySelectorAll('.response-tabs .tab-btn');
        tabButtons.forEach(tab => {
            // Remove old event listeners by cloning and replacing
            if (tab && tab.parentNode) {
                const newTab = tab.cloneNode(true);
                tab.parentNode.replaceChild(newTab, tab);
            }
        });

        // Add new event listeners
        document.querySelectorAll('.response-tabs .tab-btn').forEach(tab => {
            tab.addEventListener('click', () => {
                // Update active states for tabs
                document.querySelectorAll('.response-tabs .tab-btn').forEach(t => {
                    t.classList.remove('active');
                });
                tab.classList.add('active');
                
                // Show/hide content sections
                const type = tab.dataset.type;
                document.querySelectorAll('.response-section').forEach(section => {
                    const isActive = section.id === `${type}Response`;
                    section.classList.toggle('active', isActive);
                    section.classList.toggle('hidden', !isActive);
                });
                
                // Reinitialize save buttons when switching tabs
                initializeSaveButtons();
            });
        });

        // Display sources
        if (data.sources && data.sources.length > 0) {
            data.sources.forEach(source => {
                const li = document.createElement('li');
                li.textContent = source;
                sourcesList.appendChild(li);
            });
        }

        // Hide empty sections
        const querySuggestions = document.querySelector('.query-suggestions');
        const relatedTopics = document.querySelector('.learning-resources');
        querySuggestions.classList.add('hidden');
        relatedTopics.classList.add('hidden');

        // Update stats with reasonable defaults
        document.getElementById('docCount').textContent = '1';
        document.getElementById('nodeCount').textContent = '42';

        // Display related topics
        if (data.related_topics) {
            displayRelatedTopics(data.related_topics);
        }

        // Display suggested queries
        if (data.suggested_queries) {
            displaySuggestedQueries(data.suggested_queries);
        }

        // Show results
        resultsDiv.classList.remove('hidden');
        errorDiv.classList.add('hidden');

        // Update stats
        updateStats(data.stats);

        // Reinitialize save buttons after displaying results
        initializeSaveButtons();
    }

    // Related Topics Display
    function displayRelatedTopics(topics) {
        const topicChips = document.querySelector('.topic-chips');
        topicChips.innerHTML = '';
        
        topics.forEach(topic => {
            const chip = document.createElement('div');
            chip.className = 'chip';
            chip.textContent = topic;
            chip.addEventListener('click', () => {
                queryInput.value = `Tell me about ${topic}`;
                searchForm.dispatchEvent(new Event('submit'));
            });
            topicChips.appendChild(chip);
        });
    }

    // Suggested Queries Display
    function displaySuggestedQueries(queries) {
        const queryList = document.querySelector('.query-list');
        queryList.innerHTML = '';
        
        queries.forEach(query => {
            const li = document.createElement('li');
            li.textContent = query;
            li.addEventListener('click', () => {
                queryInput.value = query;
                searchForm.dispatchEvent(new Event('submit'));
            });
            queryList.appendChild(li);
        });
    }

    // Error Display
    function displayError(message) {
        const errorMessage = errorDiv.querySelector('.error-message');
        const errorSuggestions = errorDiv.querySelector('.error-suggestions');
        
        errorMessage.textContent = message;
        errorSuggestions.innerHTML = `
            <p>Suggestions:</p>
            <ul>
                <li>Try rephrasing your query</li>
                <li>Check for spelling mistakes</li>
                <li>Use more specific terms</li>
            </ul>
        `;
        
        errorDiv.classList.remove('hidden');
        resultsDiv.classList.add('hidden');
    }

    // History Management
    function addToSearchHistory(entry) {
        searchHistory.unshift(entry);
        if (searchHistory.length > 50) searchHistory.pop();
        localStorage.setItem('searchHistory', JSON.stringify(searchHistory));
    }

    function displaySearchHistory() {
        const historyList = document.querySelector('.history-list');
        historyList.innerHTML = '';
        
        searchHistory.forEach(entry => {
            const historyItem = document.createElement('div');
            historyItem.className = 'history-item';
            historyItem.innerHTML = `
                <h3>${entry.query}</h3>
                <p>${new Date(entry.timestamp).toLocaleString()}</p>
                <button class="rerun-btn">Re-run Query</button>
            `;
            
            historyItem.querySelector('.rerun-btn').addEventListener('click', () => {
                queryInput.value = entry.query;
                navButtons.find(btn => btn.dataset.view === 'search').click();
                searchForm.dispatchEvent(new Event('submit'));
            });
            
            historyList.appendChild(historyItem);
        });
    }

    // Save Result Handlers
    function initializeSaveButtons() {
        // Remove any existing event listeners
        document.querySelectorAll('.control-btn').forEach(btn => {
            const newBtn = btn.cloneNode(true);
            btn.parentNode.replaceChild(newBtn, btn);
        });

        // Add new event listeners
        const types = ['en', 'ar', 'dense', 'graph'];
        types.forEach(type => {
            const saveBtn = document.getElementById(`save${type === 'en' ? 'English' : type === 'ar' ? 'Arabic' : type}Result`);
            if (saveBtn) {
                saveBtn.addEventListener('click', () => {
                    console.log(`${type} save button clicked`);
                    saveResult(type);
                });
            }
        });
    }

    // Initialize save buttons when DOM is loaded
    document.addEventListener('DOMContentLoaded', initializeSaveButtons);
    
    // Re-initialize save buttons after displaying results and when switching tabs
    document.querySelectorAll('.tab-btn').forEach(tab => {
        tab.addEventListener('click', () => {
            setTimeout(initializeSaveButtons, 100); // Small delay to ensure DOM is updated
        });
    });

    async function saveResult(type) {
        console.log(`saveResult called with type: ${type}`);
        try {
            console.log('Starting HTML generation process...');
            showNotification(`Preparing to save ${type} content...`);

            // Get content based on type
            const contentElement = document.querySelector(`#${type}Response .response-content`);

            if (!contentElement || !contentElement.textContent.trim()) {
                console.error(`No ${type} content available`);
                showNotification(`No ${type} content available to save.`);
                return;
            }

            const content = contentElement.textContent;
            const sources = Array.from(sourcesList.children).map(li => li.textContent);
            
            console.log('Sending request to generate HTML...');
            const isArabic = type === 'ar';
            
            // Only use original query, no translation needed
            const originalQuery = queryInput.value;

            try {
                const response = await fetch('/generate-result', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        content: content,
                        query: originalQuery,
                        translatedQuery: '', // No translation needed
                        sources: sources,
                        isArabic: type === 'ar'
                    })
                });

                if (!response.ok) {
                    const error = await response.json();
                    throw new Error(error.details || error.error || 'Failed to generate HTML');
                }

                // Get the HTML blob
                const blob = await response.blob();
                if (blob.size === 0) {
                    throw new Error('Generated HTML is empty');
                }
                
                // Create a unique filename with timestamp
                const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
                // Create a unique filename with timestamp
                let typeName;
                switch (type) {
                    case 'en':
                        typeName = 'English';
                        break;
                    case 'ar':
                        typeName = 'Arabic';
                        break;
                    case 'dense':
                        typeName = 'DenseVector';
                        break;
                    case 'graph':
                        typeName = 'KnowledgeGraph';
                        break;
                }
                const filename = `HybridRAG_${typeName}_${timestamp}.html`;
                
                // Create a link to download the HTML
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.style.display = 'none';
                a.href = url;
                a.download = filename;
                
                // Add to document and click
                document.body.appendChild(a);
                a.click();
                
                // Clean up
                setTimeout(() => {
                    window.URL.revokeObjectURL(url);
                    document.body.removeChild(a);
                }, 100);

                console.log('HTML downloaded successfully');
                showNotification(`${typeName} result saved as HTML!`);
            } catch (htmlError) {
                console.error('HTML generation error:', htmlError);
                showNotification('Error generating HTML: ' + htmlError.message);
                throw htmlError;
            }
        } catch (error) {
            console.error('Save error:', error);
            showNotification('Error saving result: ' + error.message);
        }
    }

    // Utility Functions
    function showNotification(message) {
        const notification = document.createElement('div');
        notification.className = 'notification';
        notification.textContent = message;
        document.body.appendChild(notification);
        
        setTimeout(() => {
            notification.remove();
        }, 3000);
    }

    function updateStats(stats) {
        if (stats) {
            document.getElementById('docCount').textContent = stats.documents || 0;
            document.getElementById('nodeCount').textContent = stats.nodes || 0;
        }
    }

    // Tutorial Content
    function loadTutorial() {
        const tutorialSteps = document.querySelector('.tutorial-steps');
        const tutorialExamples = document.querySelector('.tutorial-examples');
        
        tutorialSteps.innerHTML = `
            <div class="tutorial-step">
                <h3>1. Choose Your Search Mode</h3>
                <p>Select between Hybrid, Dense Vector, or Knowledge Graph search modes for different types of queries.</p>
            </div>
            <div class="tutorial-step">
                <h3>2. Enter Your Query</h3>
                <p>Type your question in English or Arabic. The system supports both languages!</p>
            </div>
            <div class="tutorial-step">
                <h3>3. Explore Results</h3>
                <p>Review the answer, check sources, and explore related topics to deepen your understanding.</p>
            </div>
        `;
        
        tutorialExamples.innerHTML = `
            <h3>Example Queries:</h3>
            <ul>
                <li>"What are the main causes of climate change?"</li>
                <li>"Explain the theory of relativity"</li>
                <li>"Compare different types of renewable energy"</li>
            </ul>
        `;
    }
});