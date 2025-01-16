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
            } else if (viewName === 'tutorial') {
                loadTutorial();
            }
        });
    });

    // Load and display search history
    async function loadSearchHistory() {
        try {
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
                        // Switch to search view
                        document.querySelector('[data-view="search"]').click();
                    };
                    
                    item.appendChild(queryText);
                    item.appendChild(rerunButton);
                    historyList.appendChild(item);
                });
            }
        } catch (error) {
            console.error('Error loading search history:', error);
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
    function loadSavedResults() {
        const savedResultsGrid = document.querySelector('.saved-results-grid');
        if (savedResultsGrid) {
            // This would typically load from localStorage or a backend endpoint
            savedResultsGrid.innerHTML = '<p>Feature coming soon: Save and manage your search results.</p>';
        }
    }

    // Load tutorial content
    function loadTutorial() {
        const tutorialSteps = document.querySelector('.tutorial-steps');
        const tutorialExamples = document.querySelector('.tutorial-examples');
        
        if (tutorialSteps && tutorialExamples) {
            tutorialSteps.innerHTML = `
                <div class="tutorial-step">
                    <h3>1. Enter Your Query</h3>
                    <p>Type your question or search query in English or Arabic.</p>
                </div>
                <div class="tutorial-step">
                    <h3>2. Review Results</h3>
                    <p>View responses in both languages and check sources.</p>
                </div>
                <div class="tutorial-step">
                    <h3>3. Save and Share</h3>
                    <p>Download results as HTML files for future reference.</p>
                </div>
            `;

            tutorialExamples.innerHTML = `
                <h3>Example Queries:</h3>
                <ul>
                    <li>What are the main causes of the Syrian conflict?</li>
                    <li>How has the conflict affected education in Syria?</li>
                    <li>What humanitarian aid efforts are ongoing in Syria?</li>
                </ul>
            `;
        }
    }

    // Initialize response tabs
    function initializeResponseTabs() {
        const tabs = document.querySelectorAll('.response-tabs .tab-btn');
        const englishSection = document.getElementById('englishResponse');
        const arabicSection = document.getElementById('arabicResponse');
        
        function switchTab(type) {
            if (type !== 'en' && type !== 'ar') return;
            
            console.log(`Switching to ${type} tab`);
            
            // Update tab buttons
            tabs.forEach(t => t.classList.toggle('active', t.dataset.type === type));
            
            // Update section visibility
            englishSection.classList.toggle('hidden', type !== 'en');
            englishSection.classList.toggle('active', type === 'en');
            arabicSection.classList.toggle('hidden', type !== 'ar');
            arabicSection.classList.toggle('active', type === 'ar');
            
            // Update RTL/LTR
            if (type === 'ar') {
                arabicSection.dir = 'rtl';
                arabicSection.style.textAlign = 'right';
                arabicSection.querySelector('.response-content').style.fontFamily = "'Noto Naskh Arabic', Arial, sans-serif";
            } else {
                englishSection.dir = 'ltr';
                englishSection.style.textAlign = 'left';
            }
            
            console.log(`Tab switched to ${type}`);
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
        console.log('Setting up save buttons with data:', {
            hasEnglishFile: Boolean(data.english_file),
            hasArabicFile: Boolean(data.arabic_file)
        });

        const saveButtons = document.querySelectorAll('.save-btn');
        console.log(`Found ${saveButtons.length} save buttons`);

        saveButtons.forEach(btn => {
            const lang = btn.dataset.lang;
            const filename = lang === 'en' ? data.english_file : data.arabic_file;
            
            console.log(`Configuring ${lang} save button:`, {
                hasFile: Boolean(filename),
                filename: filename
            });
            
            // Remove existing listeners
            const newBtn = btn.cloneNode(true);
            btn.parentNode.replaceChild(newBtn, btn);
            
            if (filename) {
                console.log(`Setting up save button for ${lang}:`, {
                    filename: filename,
                    buttonId: newBtn.id,
                    buttonLang: newBtn.dataset.lang
                });
                
                newBtn.disabled = false;
                newBtn.title = lang === 'en' ? "Click to download HTML file" : "انقر للتحميل بصيغة HTML";
                
                newBtn.addEventListener('click', (e) => {
                    e.preventDefault();
                    console.log('\nSave button clicked:');
                    console.log('Language:', lang);
                    console.log('Filename:', filename);
                    console.log('Button state:', {
                        disabled: newBtn.disabled,
                        dataset: newBtn.dataset,
                        title: newBtn.title
                    });
                    
                    const downloadUrl = `/results/${filename}`;
                    console.log('Download URL:', downloadUrl);
                    
                    try {
                        window.location.href = downloadUrl;
                        console.log('Download initiated');
                    } catch (error) {
                        console.error('Error initiating download:', error);
                    }
                });
                
                console.log(`${lang} save button enabled and configured`);
            } else {
                console.log(`${lang} save button disabled:`, {
                    reason: 'No filename available',
                    buttonId: newBtn.id,
                    buttonLang: newBtn.dataset.lang
                });
                
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
            console.log('\nSearch response received:', {
                status: response.status,
                ok: response.ok,
                hasEnglishAnswer: Boolean(data.answer),
                hasArabicAnswer: Boolean(data.arabic_answer),
                englishFile: data.english_file,
                arabicFile: data.arabic_file,
                language: data.language
            });

            if (response.ok) {
                console.log('Processing search response data:', {
                    englishContentLength: data.answer?.length,
                    arabicContentLength: data.arabic_answer?.length,
                    hasEnglishFile: Boolean(data.english_file),
                    hasArabicFile: Boolean(data.arabic_file),
                    sourceCount: data.sources?.length
                });
                displayResults(data);
            } else {
                console.error('Search response error:', data.error);
                throw new Error(data.error || 'An error occurred while processing your query');
            }
        } catch (error) {
            displayError(error.message);
        } finally {
            loadingIndicator.classList.add('hidden');
            searchButton.disabled = false;
        }
    });

    // Display results
    function displayResults(data) {
        if (data.error) {
            displayError(data.error);
            return;
        }

        // Clear previous content
        document.querySelectorAll('.response-content').forEach(content => {
            content.textContent = '';
        });
        sourcesList.innerHTML = '';

        console.log('Displaying responses with data:', {
            hasEnglish: Boolean(data.answer),
            hasArabic: Boolean(data.arabic_answer),
            englishFile: data.english_file,
            arabicFile: data.arabic_file
        });

        console.log('Setting up responses with data:', {
            hasEnglish: Boolean(data.answer),
            hasArabic: Boolean(data.arabic_answer),
            englishLength: data.answer?.length,
            arabicLength: data.arabic_answer?.length
        });

        // Update confidence indicator
        const confidenceFill = document.querySelector('.confidence-fill');
        const confidencePercentage = document.querySelector('.confidence-percentage');
        if (confidenceFill && confidencePercentage) {
            const confidence = data.confidence || 0;
            confidenceFill.style.width = `${confidence}%`;
            confidencePercentage.textContent = `${confidence}%`;
            console.log('Confidence updated:', confidence);
        }

        // Display English response
        const englishContent = document.querySelector('#englishResponse .response-content');
        if (englishContent && data.answer) {
            englishContent.textContent = data.answer;
            console.log('English content set');
        }

        // Display Arabic response
        const arabicContent = document.querySelector('#arabicResponse .response-content');
        if (arabicContent) {
            if (data.arabic_answer) {
                arabicContent.textContent = data.arabic_answer;
                arabicContent.dir = 'rtl';
                arabicContent.style.textAlign = 'right';
                arabicContent.style.fontFamily = "'Noto Naskh Arabic', Arial, sans-serif";
                console.log('Arabic content set');
            } else {
                console.error('No Arabic content available');
                arabicContent.textContent = 'المحتوى العربي غير متوفر';
                arabicContent.dir = 'rtl';
                arabicContent.style.textAlign = 'right';
                arabicContent.style.color = '#666';
            }
        }

        // Setup save buttons with file information
        setupSaveButtons(data);

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
    }

    // Error display
    function displayError(message) {
        const errorMessage = errorDiv.querySelector('.error-message');
        errorMessage.textContent = message;
        errorDiv.classList.remove('hidden');
        resultsDiv.classList.add('hidden');
    }

    // Set initial tab
    switchTab('en');
});