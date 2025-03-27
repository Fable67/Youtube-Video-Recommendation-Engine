const searchBtn = document.getElementById('searchBtn');
const searchInput = document.getElementById('searchInput');
const errorMsg = document.getElementById('errorMsg');

// Listen for Enter on the entire document
document.addEventListener('keydown', event => {
    if (event.key === 'Enter') {
        event.preventDefault();
        startSearch();
    }
});

searchBtn.addEventListener('click', async () => {
    startSearch();
});

async function startSearch() {
    const query = searchInput.value.trim();
    if (!query) {
        errorMsg.textContent = 'Please enter a query.';
        errorMsg.style.display = 'block';
        return;
    } else {
        errorMsg.style.display = 'none';
    }
    searchBtn.classList.add('loading');

    try {
        const response = await fetch('http://localhost:4999/api/search', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query })
        });
        if (!response.ok) {
            const { error } = await response.json();
            throw new Error(error || 'Search request failed');
        }
        const results = await response.json();
        // Store results in sessionStorage
        sessionStorage.setItem('searchResults', JSON.stringify(results));
        sessionStorage.setItem('query', query)
        // Redirect to results page
        window.location.href = 'results.html';
    } catch (err) {
        searchBtn.classList.remove('loading');
        errorMsg.textContent = err.message;
        errorMsg.style.display = 'block';
    }
}