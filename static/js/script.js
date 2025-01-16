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

    function initializeResponseTabs() {
        const tabs = document.querySelectorAll('.response-tabs .tab-btn');
        const englishSection = document.getElementById('englishResponse');
        const arabicSection = document.getElementById('arabicResponse');
        
        function switchTab(type) {
            if (type !== 'en' && type !== 'ar') {
                return;
            }

            tabs.forEach(t => {
                const isActive = t.dataset.type === type;
                t.classList.toggle('active', isActive);
            });
            
            englishSection.classList.toggle('hidden', type !== 'en');
            englishSection.classList.toggle('active', type === 'en');
            arabicSection.classList.toggle('hidden', type !== 'ar');
            arabicSection.classList.toggle('active', type === 'ar');

            currentLanguage = type;
            document.body.dir = type === 'ar' ? 'rtl' : 'ltr';
        }

        tabs.forEach(tab => {
            tab.addEventListener('click', () => {
                const type = tab.dataset.type;
                if (type) {
                    switchTab(type);
                }
            });
        });

        return switchTab;
    }

    // Initialize tabs and get the switchTab function
    const switchTab = initializeResponseTabs();
    
    // Initialize save buttons
    function initializeSaveButtons() {
        document.querySelectorAll('.save-btn').forEach(btn => {
            // Remove existing listeners to prevent duplicates
            const newBtn = btn.cloneNode(true);
            btn.parentNode.replaceChild(newBtn, btn);
            
            newBtn.addEventListener('click', (e) => {
                e.preventDefault();
                e.stopPropagation();
                const lang = newBtn.dataset.lang;
                saveResponseAsHtml(lang);
            });
        });
    }

    // Initialize save buttons on page load
    initializeSaveButtons();
    
    // Navigation Handler
    navButtons.forEach(button => {
        button.addEventListener('click', () => {
            const targetView = button.dataset.view;
            
            navButtons.forEach(btn => btn.classList.remove('active'));
            button.classList.add('active');
            
            views.forEach(view => {
                view.classList.toggle('active', view.id === `${targetView}View`);
                view.classList.toggle('hidden', view.id !== `${targetView}View`);
            });

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

        errorDiv.classList.add('hidden');
        resultsDiv.classList.add('hidden');
        loadingIndicator.classList.remove('hidden');
        searchButton.disabled = true;

        startProcessAnimation();

        try {
            const response = await fetch('/search', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    query: query,
                    mode: 'hybrid'
                })
            });

            const data = await response.json();

            if (response.ok) {
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
        processSteps[0].classList.add('active');
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
        document.querySelectorAll('.response-content').forEach(content => {
            content.textContent = '';
        });
        sourcesList.innerHTML = '';

        resultsDiv.classList.remove('hidden');
        errorDiv.classList.add('hidden');

        if (data.error) {
            displayError(data.error);
            return;
        }

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

        Object.entries(responses).forEach(([type, info]) => {
            if (info.element) {
                info.element.textContent = info.content || `No ${type} response available`;
                info.element.setAttribute('dir', info.dir);
                
                const parentSection = info.element.closest('.response-section');
                if (parentSection) {
                    parentSection.setAttribute('dir', info.dir);
                }
            }
        });

        const confidenceScore = document.querySelector('.confidence-score');
        if (confidenceScore) {
            const score = data.confidence || '85%';
            const confidenceBar = confidenceScore.querySelector('.score-bar');
            const confidenceValue = confidenceScore.querySelector('.score-value');
            
            confidenceBar.style.setProperty('--score', score);
            confidenceValue.textContent = score;
            confidenceScore.classList.remove('hidden');
        }

        const activeTab = data.language === 'ar' ? 'ar' : 'en';
        switchTab(activeTab);

        if (data.sources && data.sources.length > 0) {
            data.sources.forEach(source => {
                const li = document.createElement('li');
                li.textContent = source;
                sourcesList.appendChild(li);
            });
        }

        const querySuggestions = document.querySelector('.query-suggestions');
        const relatedTopics = document.querySelector('.learning-resources');
        querySuggestions.classList.add('hidden');
        relatedTopics.classList.add('hidden');

        if (data.related_topics) {
            displayRelatedTopics(data.related_topics);
        }

        if (data.suggested_queries) {
            displaySuggestedQueries(data.suggested_queries);
        }

        resultsDiv.classList.remove('hidden');
        errorDiv.classList.add('hidden');

        // Reinitialize save buttons after content is loaded
        initializeSaveButtons();

        if (data.stats) {
            document.getElementById('docCount').textContent = data.stats.documents || 0;
            document.getElementById('nodeCount').textContent = data.stats.nodes || 0;
        }
    }

    // Save Response Handler
    function saveResponseAsHtml(lang) {
        const responseSection = document.getElementById(`${lang}Response`);
        const content = responseSection.querySelector('.response-content').textContent;
        
        if (!content) {
            showNotification('No content to save');
            return;
        }

        const firstLine = content.split('\n')[0].trim().substring(0, 50)
            .replace(/[^a-zA-Z0-9\u0600-\u06FF\s]/g, '')
            .replace(/\s+/g, '_');
        
        const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
        const filename = `${firstLine}_${lang}_${timestamp}.html`;

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
                <p>Review the answers in both English and Arabic.</p>
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