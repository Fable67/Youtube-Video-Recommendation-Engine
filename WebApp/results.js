function toggleTopics(id) {
    const list = document.getElementById(id);
    if (list.style.display === "none" || list.style.display === "") {
    list.style.display = "block";
    } else {
    list.style.display = "none";
    }
}

// Load and display results on DOMContentLoaded
document.addEventListener('DOMContentLoaded', () => {
    const query = sessionStorage.getItem('query')
    const resultsJson = sessionStorage.getItem('searchResults');
    if (!resultsJson) {
    // No results found in sessionStorage, redirect or show error
    document.querySelector('.results').innerHTML = '<p>No results found. Please return to search.</p>';
    return;
    }
    const results = JSON.parse(resultsJson);
    const resultsContainer = document.querySelector('.results');

    // Clear any placeholder or existing content
    resultsContainer.innerHTML = '';

    // Create a header for the page
    if (query) {
        document.querySelector('h1').textContent = `Search Results for "${query}"`;
    } else {
        document.querySelector('h1').textContent = 'Search Results';
    }

    // For each result, build the HTML structure
    results.forEach((res, i) => {
    // Create the result container
    const resultDiv = document.createElement('div');
    resultDiv.className = 'result';

    // Create the iframe
    const iframe = document.createElement('iframe');
    iframe.src = res.embed_url;
    iframe.allowFullscreen = true;
    resultDiv.appendChild(iframe);

    // Details div
    const detailsDiv = document.createElement('div');
    detailsDiv.className = 'details';
    // Title
    const titleEl = document.createElement('h2');
    titleEl.textContent = res.title;
    detailsDiv.appendChild(titleEl);

    // Similarity
    const simEl = document.createElement('p');
    simEl.className = 'similarity';
    // multiply res.similarity by 100, round to 2 decimal places
    simEl.textContent = `Similarity Score: ${(res.similarity * 100).toFixed(2)}%`;
    detailsDiv.appendChild(simEl);

    // Topics
    const topicsDiv = document.createElement('div');
    topicsDiv.className = 'topics';

    const topicsBtn = document.createElement('button');
    topicsBtn.textContent = 'Topics';
    const topicsId = `topics-${i}`;
    topicsBtn.addEventListener('click', () => toggleTopics(topicsId));
    topicsDiv.appendChild(topicsBtn);

    const topicsList = document.createElement('div');
    topicsList.className = 'topics-list';
    topicsList.id = topicsId;

    // Build each topic
    res.topics.forEach((topic, idx) => {
        const span = document.createElement('span');
        if (idx === res.topic_index) {
        span.classList.add('highlight-topic');
        }
        // Replace newline with space
        span.textContent = topic.replace(/\n/g, ' ');
        topicsList.appendChild(span);
    });

    topicsDiv.appendChild(topicsList);
    detailsDiv.appendChild(topicsDiv);

    resultDiv.appendChild(detailsDiv);
    resultsContainer.appendChild(resultDiv);
    });
});
