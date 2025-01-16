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

    // State Management
    let searchHistory = JSON.parse(localStorage.getItem('searchHistory') || '[]');
    
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

        // Show results container
        resultsDiv.classList.remove('hidden');

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

        // Always show both language tabs
        document.querySelector('[data-lang="ar"]').style.display = 'block';
        document.querySelector('[data-lang="en"]').style.display = 'block';

        // Show results container first
        resultsDiv.classList.remove('hidden');

        // Show both response sections and tabs
        englishResponse.classList.remove('hidden');
        arabicResponse.classList.remove('hidden');
        document.querySelector('[data-lang="en"]').style.display = 'block';
        document.querySelector('[data-lang="ar"]').style.display = 'block';

        // Get content elements
        const englishContent = englishResponse.querySelector('.response-content');
        const arabicContent = arabicResponse.querySelector('.response-content');

        // Clear previous content
        englishContent.textContent = '';
        arabicContent.textContent = '';

        // Always set English content
        englishContent.textContent = data.answer;
        englishResponse.classList.remove('hidden');
        englishResponse.classList.add('active');
        
        // Set Arabic content if available
        if (data.arabic_answer) {
            arabicContent.textContent = data.arabic_answer;
            arabicContent.setAttribute('dir', 'rtl');
            arabicResponse.classList.remove('hidden');
            arabicResponse.classList.add('active');
        }
        
        // Always show both tabs
        document.querySelector('[data-lang="en"]').style.display = 'block';
        document.querySelector('[data-lang="ar"]').style.display = 'block';
        
        // Set active tab based on query language
        if (data.language === 'ar') {
            document.querySelector('[data-lang="ar"]').click();
        } else {
            document.querySelector('[data-lang="en"]').click();
        }

        // Remove previous tab click handlers
        langTabs.forEach(tab => {
            const newTab = tab.cloneNode(true);
            tab.parentNode.replaceChild(newTab, tab);
        });

        // Re-initialize language tabs
        document.querySelectorAll('.response-tabs .tab-btn').forEach(tab => {
            tab.addEventListener('click', () => {
                // Update active states for tabs
                document.querySelectorAll('.response-tabs .tab-btn').forEach(t => {
                    t.classList.remove('active');
                });
                tab.classList.add('active');
                
                // Show/hide content sections
                const lang = tab.dataset.lang;
                if (lang === 'en') {
                    englishResponse.classList.add('active');
                    englishResponse.classList.remove('hidden');
                    arabicResponse.classList.remove('active');
                    arabicResponse.classList.add('hidden');
                } else {
                    arabicResponse.classList.add('active');
                    arabicResponse.classList.remove('hidden');
                    englishResponse.classList.remove('active');
                    englishResponse.classList.add('hidden');
                }
                
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
        const saveEnglishBtn = document.getElementById('saveEnglishResult');
        const saveArabicBtn = document.getElementById('saveArabicResult');
        
        if (saveEnglishBtn) {
            saveEnglishBtn.addEventListener('click', () => {
                console.log('English save button clicked');
                saveResult('en');
            });
        }
        
        if (saveArabicBtn) {
            saveArabicBtn.addEventListener('click', () => {
                console.log('Arabic save button clicked');
                saveResult('ar');
            });
        }
    }

    // Initialize save buttons when DOM is loaded
    document.addEventListener('DOMContentLoaded', initializeSaveButtons);
    
    // Re-initialize save buttons after displaying results and when switching tabs
    document.querySelectorAll('.tab-btn').forEach(tab => {
        tab.addEventListener('click', () => {
            setTimeout(initializeSaveButtons, 100); // Small delay to ensure DOM is updated
        });
    });

    async function saveResult(lang) {
        console.log(`saveResult called with lang: ${lang}`);
        try {
            console.log('Starting PDF generation process...');
            showNotification(`Preparing to save ${lang === 'ar' ? 'Arabic' : 'English'} content...`);
            
            if (!window.jspdf) {
                console.error('jsPDF library not found in window object');
                showNotification('PDF library not loaded. Please try again.');
                return;
            }
            console.log('jsPDF library found');

            // Get content based on language
            const contentElement = lang === 'ar' ?
                arabicResponse.querySelector('.response-content') :
                englishResponse.querySelector('.response-content');

            if (!contentElement || !contentElement.textContent.trim()) {
                console.error(`No ${lang} content available`);
                showNotification(`No ${lang === 'ar' ? 'Arabic' : 'English'} content available to save.`);
                return;
            }
            const content = contentElement.textContent;
            console.log(`${lang} content length:`, content.length);

            const sources = Array.from(sourcesList.children).map(li => li.textContent);
            console.log('Number of sources:', sources.length);

            // Initialize PDF
            console.log('Initializing PDF document...');
            const { jsPDF } = window.jspdf;
            const doc = new jsPDF({
                orientation: 'portrait',
                unit: 'mm',
                format: 'a4'
            });

            try {
                // Configure PDF settings
                console.log('Setting up PDF...');
                doc.setFont('helvetica', 'normal');
                doc.setFontSize(12);

                // Common settings
                const margin = 20;
                const pageWidth = doc.internal.pageSize.width;
                let currentY = margin;

                // Add metadata
                doc.text('Generated: ' + new Date().toLocaleString(), margin, currentY);
                currentY += 15;

                // Add query
                doc.text('Query:', margin, currentY);
                currentY += 10;
                const queryLines = doc.splitTextToSize(queryInput.value, pageWidth - 2 * margin);
                doc.text(queryLines, margin, currentY);
                currentY += queryLines.length * 7 + 10;

                // Add answer
                doc.text('Answer:', margin, currentY);
                currentY += 10;
                const contentLines = doc.splitTextToSize(content, pageWidth - 2 * margin);
                doc.text(contentLines, margin, currentY);
                currentY += contentLines.length * 7 + 10;

                // Add sources if available
                if (sources.length > 0) {
                    if (currentY > doc.internal.pageSize.height - margin) {
                        doc.addPage();
                        currentY = margin;
                    }
                    doc.text('Sources:', margin, currentY);
                    currentY += 10;
                    sources.forEach((source, i) => {
                        const sourceText = `${i + 1}. ${source}`;
                        const sourceLines = doc.splitTextToSize(sourceText, pageWidth - 2 * margin);
                        
                        if (currentY + (sourceLines.length * 7) > doc.internal.pageSize.height - margin) {
                            doc.addPage();
                            currentY = margin;
                        }
                        
                        doc.text(sourceLines, margin, currentY);
                        currentY += sourceLines.length * 7;
                    });
                }

                console.log('Content added to PDF successfully');

                // Save with language-specific filename
                const langSuffix = lang === 'ar' ? 'Arabic' : 'English';
                const filename = `HybridRAG_Result_${langSuffix}_${new Date().toISOString().replace(/[:.]/g, '-')}.pdf`;
                console.log('Saving PDF as:', filename);
                doc.save(filename);
                console.log('PDF saved successfully');
                showNotification(`${langSuffix} result saved as PDF!`);
            } catch (error) {
                console.error('Error adding content to PDF:', error);
                throw error;
            }
        } catch (error) {
            console.error('PDF generation error:', error);
            console.error('Error stack:', error.stack);
            
            // Provide more specific error messages based on the error type
            let errorMessage = 'Error saving PDF. ';
            if (error.message.includes('font')) {
                errorMessage += 'Font loading failed. Please try again.';
            } else if (error.message.includes('VFS')) {
                errorMessage += 'Font system initialization failed.';
            } else if (error.message.includes('rendering')) {
                errorMessage += 'Text rendering failed. Please check the content.';
            } else {
                errorMessage += 'Please try again or contact support if the issue persists.';
            }
            
            showNotification(errorMessage);
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