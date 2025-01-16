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
    const langTabs = document.querySelectorAll('.response-tabs .tab-btn');
    const processSteps = document.querySelectorAll('.process-steps .step');

    // State Management
    let searchHistory = JSON.parse(localStorage.getItem('searchHistory') || '[]');
    let currentLanguage = 'en';

    // Debug wrapper for tab operations
    function debugTabOperation(operation, details) {
        console.group(`Tab Operation: ${operation}`);
        console.log('Details:', details);
        console.log('Current State:', {
            activeTab: document.querySelector('.tab-btn.active')?.dataset.type,
            englishContent: document.querySelector('#englishResponse .response-content')?.textContent.length,
            arabicContent: document.querySelector('#arabicResponse .response-content')?.textContent.length,
            englishVisible: !document.querySelector('#englishResponse')?.classList.contains('hidden'),
            arabicVisible: !document.querySelector('#arabicResponse')?.classList.contains('hidden'),
            direction: document.body.dir
        });
        console.groupEnd();
    }

    function initializeResponseTabs() {
        console.group('Tab Initialization');
        
        const tabs = document.querySelectorAll('.response-tabs .tab-btn');
        const englishSection = document.getElementById('englishResponse');
        const arabicSection = document.getElementById('arabicResponse');

        console.log('Found elements:', {
            tabCount: tabs.length,
            hasEnglishSection: !!englishSection,
            hasArabicSection: !!arabicSection
        });
        
        function switchTab(type) {
            console.group('Tab Switch');
            console.log('Switching to tab:', type);

            // Validate type
            if (type !== 'en' && type !== 'ar') {
                console.error('Invalid tab type:', type);
                console.groupEnd();
                return;
            }

            // Log initial state
            console.log('Initial state:', {
                englishVisible: !englishSection.classList.contains('hidden'),
                arabicVisible: !arabicSection.classList.contains('hidden'),
                currentLang: currentLanguage
            });

            // Update tab states
            tabs.forEach(t => {
                const isActive = t.dataset.type === type;
                t.classList.toggle('active', isActive);
                console.log(`Tab ${t.dataset.type}:`, isActive ? 'active' : 'inactive');
            });
            
            // Update section visibility
            englishSection.classList.toggle('hidden', type !== 'en');
            englishSection.classList.toggle('active', type === 'en');
            arabicSection.classList.toggle('hidden', type !== 'ar');
            arabicSection.classList.toggle('active', type === 'ar');

            // Log content state
            const activeSection = type === 'en' ? englishSection : arabicSection;
            console.log('Active section state:', {
                id: activeSection.id,
                hasContent: !!activeSection.querySelector('.response-content')?.textContent.trim(),
                contentLength: activeSection.querySelector('.response-content')?.textContent.trim().length || 0,
                isHidden: activeSection.classList.contains('hidden'),
                isActive: activeSection.classList.contains('active')
            });

            // Update current language and direction
            currentLanguage = type;
            document.body.dir = type === 'ar' ? 'rtl' : 'ltr';
            console.log('Language and direction updated:', {
                language: currentLanguage,
                direction: document.body.dir
            });

            console.groupEnd();
        }

        // Add click handlers
        tabs.forEach(tab => {
            tab.addEventListener('click', () => {
                const type = tab.dataset.type;
                if (type) {
                    switchTab(type);
                }
            });
        });

        // Export for use in other functions
        return switchTab;
    }

    // Initialize tabs and get the switchTab function
    const switchTab = initializeResponseTabs();
    
    // Save Response Handler
    function saveResponseAsHtml(lang) {
        const responseSection = document.getElementById(`${lang}Response`);
        const content = responseSection.querySelector('.response-content').textContent;
        
        if (!content) {
            showNotification('No content to save');
            return;
        }

        // Get first line for filename (limited to 50 chars)
        const firstLine = content.split('\n')[0].trim().substring(0, 50)
            .replace(/[^a-zA-Z0-9\u0600-\u06FF\s]/g, '') // Keep English, Arabic, numbers, and spaces
            .replace(/\s+/g, '_'); // Replace spaces with underscores
        
        const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
        const filename = `${firstLine}_${lang}_${timestamp}.html`;

        // Create HTML content
        const htmlContent = `
<!DOCTYPE html>
<html lang="${lang}">
<head>
    <meta charset="UTF-8">
    <title>${firstLine}</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            margin: 40px;
            direction: ${lang === 'ar' ? 'rtl' : 'ltr'};
        }
        .content {
            max-width: 800px;
            margin: 0 auto;
        }
        .timestamp {
            color: #666;
            font-size: 0.9em;
            margin-bottom: 20px;
        }
    </style>
</head>
<body>
    <div class="content">
        <div class="timestamp">Generated on: ${new Date().toLocaleString()}</div>
        <div class="response-content">
            ${content.split('\n').map(line => `<p>${line}</p>`).join('')}
        </div>
    </div>
</body>
</html>`;

        // Create blob and download
        const blob = new Blob([htmlContent], { type: 'text/html;charset=utf-8' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);

        showNotification(`Saved as ${filename}`);
    }

    // Add event listeners to save buttons
    document.querySelectorAll('.save-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const lang = btn.dataset.lang;
            saveResponseAsHtml(lang);
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
                    mode: 'hybrid'  // Always use hybrid mode
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

        // Show results container and hide error
        resultsDiv.classList.remove('hidden');
        errorDiv.classList.add('hidden');

        if (data.error) {
            displayError(data.error);
            return;
        }

        console.group('Display Results');
        console.log('Raw response data:', data);

        // Update content for both languages
        const responses = {
            'en': {
                element: document.querySelector('#englishResponse .response-content'),
                content: data.answer,
                dir: 'ltr'
            },
            'ar': {
                element: document.querySelector('#arabicResponse .response-content'),
                content: data.arabic_answer,
                dir: 'rtl'
            }
        };

        // Log response data
        console.log('English content available:', !!responses.en.content);
        console.log('Arabic content available:', !!responses.ar.content);
        console.log('English element found:', !!responses.en.element);
        console.log('Arabic element found:', !!responses.ar.element);

        // Update content for both languages
        Object.entries(responses).forEach(([type, info]) => {
            if (info.element) {
                console.log(`Updating ${type} content:`, {
                    contentLength: info.content?.length || 0,
                    hasContent: !!info.content,
                    elementId: info.element.id,
                    parentId: info.element.parentElement?.id
                });

                info.element.textContent = info.content || `No ${type} response available`;
                info.element.setAttribute('dir', info.dir);
                
                // Ensure parent section has correct direction
                const parentSection = info.element.closest('.response-section');
                if (parentSection) {
                    parentSection.setAttribute('dir', info.dir);
                    console.log(`Set direction for ${type} section:`, info.dir);
                }
            } else {
                console.warn(`${type} response element not found`);
            }
        });

        console.groupEnd();

        // Update confidence score if available
        const confidenceScore = document.querySelector('.confidence-score');
        if (confidenceScore) {
            const score = data.confidence || '85%';
            const confidenceBar = confidenceScore.querySelector('.score-bar');
            const confidenceValue = confidenceScore.querySelector('.score-value');
            
            confidenceBar.style.setProperty('--score', score);
            confidenceValue.textContent = score;
            confidenceScore.classList.remove('hidden');
        }

        // Switch to appropriate tab based on query language
        const activeTab = data.language === 'ar' ? 'ar' : 'en';
        switchTab(activeTab);

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

    // Set initial tab
    switchTab('en');

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
                <h3>1. Enter Your Query</h3>
                <p>Type your question in English or Arabic. The system supports both languages!</p>
            </div>
            <div class="tutorial-step">
                <h3>2. Explore Results</h3>
                <p>Review the answers in different formats: English, Arabic, Dense Vector, and Knowledge Graph representations.</p>
            </div>
            <div class="tutorial-step">
                <h3>3. Check Sources</h3>
                <p>Review sources and evidence, explore related topics to deepen your understanding.</p>
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