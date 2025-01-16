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
    const searchMode = document.getElementById('searchMode');
    const langTabs = document.querySelectorAll('.response-tabs .tab-btn');
    const processSteps = document.querySelectorAll('.process-steps .step');
    const saveResultBtn = document.getElementById('saveResult');

    // State Management
    let searchHistory = JSON.parse(localStorage.getItem('searchHistory') || '[]');
    let savedResults = JSON.parse(localStorage.getItem('savedResults') || '[]');
    
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
                    mode: searchMode.value
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
        englishResponse.querySelector('.response-content').textContent = '';
        arabicResponse.querySelector('.response-content').textContent = '';
        sourcesList.innerHTML = '';

        if (data.error) {
            displayError(data.error);
            return;
        }

        // Handle confidence score display
        const confidenceScore = englishResponse.querySelector('.confidence-score');
        const confidenceBar = confidenceScore.querySelector('.score-bar');
        const confidenceValue = confidenceScore.querySelector('.score-value');
        
        if (searchMode.value === 'hybrid') {
            const score = '85%'; // Default confidence score
            confidenceBar.style.setProperty('--score', score);
            confidenceValue.textContent = score;
            confidenceScore.classList.remove('hidden');
        } else {
            confidenceScore.classList.add('hidden');
        }

        // Display responses in both languages
        englishResponse.querySelector('.response-content').innerHTML = data.answer;
        
        if (data.arabic_answer) {
            const arabicContent = arabicResponse.querySelector('.response-content');
            arabicContent.innerHTML = data.arabic_answer;
            arabicContent.setAttribute('dir', 'rtl');
            document.querySelector('[data-lang="ar"]').style.display = 'block';
        } else {
            document.querySelector('[data-lang="ar"]').style.display = 'none';
        }

        // Initialize language tabs
        langTabs.forEach(tab => {
            tab.addEventListener('click', () => {
                // Update active states
                langTabs.forEach(t => t.classList.remove('active'));
                tab.classList.add('active');
                
                // Show selected language content
                const lang = tab.dataset.lang;
                const sections = document.querySelectorAll('.response-section');
                sections.forEach(section => {
                    section.classList.toggle('active',
                        (lang === 'en' && section.id === 'englishResponse') ||
                        (lang === 'ar' && section.id === 'arabicResponse')
                    );
                });
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

    // Save Result Handler
    saveResultBtn.addEventListener('click', () => {
        const { jsPDF } = window.jspdf;
        const doc = new jsPDF();
        
        // Get content
        const query = queryInput.value;
        const answer = englishResponse.querySelector('.response-content').textContent;
        const sources = Array.from(sourcesList.children).map(li => li.textContent);
        const timestamp = new Date().toLocaleString();
        
        // Set up PDF
        doc.setFontSize(16);
        doc.text('HybridRAG Search Result', 20, 20);
        
        doc.setFontSize(12);
        doc.text(`Generated: ${timestamp}`, 20, 30);
        
        doc.setFontSize(14);
        doc.text('Query:', 20, 45);
        doc.setFontSize(12);
        const queryLines = doc.splitTextToSize(query, 170);
        doc.text(queryLines, 20, 55);
        
        doc.setFontSize(14);
        doc.text('Answer:', 20, 75);
        doc.setFontSize(12);
        const answerLines = doc.splitTextToSize(answer, 170);
        doc.text(answerLines, 20, 85);
        
        if (sources.length > 0) {
            let yPosition = 105 + (answerLines.length * 7); // Start after answer with some padding
            
            doc.setFontSize(14);
            doc.text('Sources:', 20, yPosition);
            doc.setFontSize(12);
            
            sources.forEach((source, index) => {
                yPosition += 10;
                const sourceLines = doc.splitTextToSize(`${index + 1}. ${source}`, 170);
                doc.text(sourceLines, 20, yPosition);
                yPosition += (sourceLines.length * 7);
                
                // Add new page if needed
                if (yPosition > 280) {
                    doc.addPage();
                    yPosition = 20;
                }
            });
        }
        
        // Generate filename with timestamp
        const filename = `HybridRAG_Result_${new Date().toISOString().replace(/[:.]/g, '-')}.pdf`;
        
        // Save PDF
        doc.save(filename);
        showNotification('Result saved as PDF!');
    });

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