const API = '';

const searchInput = document.getElementById('searchInput');
const searchBtn = document.getElementById('searchBtn');
const resultsGrid = document.getElementById('resultsGrid');
const loadingIndicator = document.getElementById('loadingIndicator');
const welcomeMessage = document.getElementById('welcomeMessage');

let isSearching = false;

async function executeSearch() {
  const query = searchInput.value.trim();
  if (!query || isSearching) return;

  isSearching = true;
  searchBtn.textContent = 'Searching...';
  searchBtn.disabled = true;
  
  if (welcomeMessage) welcomeMessage.style.display = 'none';
  resultsGrid.innerHTML = '';
  loadingIndicator.style.display = 'block';

  try {
    const res = await fetch(`${API}/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: query })
    });
    
    loadingIndicator.style.display = 'none';
    
    if (res.ok) {
      const data = await res.json();
      renderProfiles(data.answer);
    } else {
      resultsGrid.innerHTML = `<div style="grid-column: 1/-1; color: red;">Error: ${res.statusText}</div>`;
    }
  } catch (err) {
    loadingIndicator.style.display = 'none';
    resultsGrid.innerHTML = `<div style="grid-column: 1/-1; color: red;">Network Error: ${err.message}. Ensure the backend is running.</div>`;
  }

  isSearching = false;
  searchBtn.textContent = 'Search';
  searchBtn.disabled = false;
}

if (searchBtn) searchBtn.addEventListener('click', executeSearch);
if (searchInput) {
  searchInput.addEventListener('keydown', e => {
    if (e.key === 'Enter') executeSearch();
  });
}

function renderProfiles(jsonString) {
  try {
    // Try mapping LLM response to a JSON array. Sometimes LLMs use markdown blocks around JSON.
    const cleanString = jsonString.replace(/```json/g, '').replace(/```/g, '').trim();
    const profiles = JSON.parse(cleanString);

    if (!Array.isArray(profiles) || profiles.length === 0) {
      resultsGrid.innerHTML = `<div style="grid-column: 1/-1; text-align: center; color: var(--text-light); padding: 2rem;">No matching profiles found for these requirements.</div>`;
      return;
    }

    profiles.forEach(p => {
      const card = document.createElement('div');
      card.className = 'profile-card';
      card.style.cssText = `
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s, box-shadow 0.2s;
        display: flex;
        flex-direction: column;
        gap: 0.5rem;
      `;
      
      const avatarSVG = `<svg viewBox="0 0 24 24" fill="var(--surface-2)" stroke="var(--primary)" stroke-width="1.5" style="width: 48px; height: 48px; border-radius: 50%;"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path><circle cx="12" cy="7" r="4"></circle></svg>`;
      
      card.innerHTML = `
        <div style="display: flex; align-items: center; gap: 1rem; border-bottom: 1px solid var(--border-light); padding-bottom: 1rem; margin-bottom: 0.5rem;">
          ${avatarSVG}
          <div>
            <h3 style="margin: 0; color: var(--text-dark); font-size: 1.25rem;">${p.name || 'Unknown Actor'}</h3>
            <span style="background: var(--surface-2); padding: 2px 8px; border-radius: 12px; font-size: 0.8rem; color: var(--primary); font-weight: 500;">
              ⭐ ${parseFloat(p.rating) ? parseFloat(p.rating).toFixed(1) : p.rating || 'N/A'}
            </span>
          </div>
        </div>
        <div style="font-size: 0.9rem; color: var(--text);">
          <strong style="color: var(--text-dark);">Age:</strong> ${p.age || 'N/A'}
        </div>
        <div style="font-size: 0.9rem; color: var(--text);">
          <strong style="color: var(--text-dark);">Height:</strong> ${p.height || 'N/A'}
        </div>
        <div style="font-size: 0.9rem; color: var(--text); margin-top: 0.5rem;">
          <strong style="color: var(--text-dark);">Skills:</strong>
          <p style="margin: 0.25rem 0 0 0; color: var(--text-light); line-height: 1.4;">${p.skills || 'None'}</p>
        </div>
      `;
      
      // Match Reason
      if (p.reason) {
        card.innerHTML += `
          <div style="font-size: 0.85rem; padding-top: 1rem; margin-top: auto; color: var(--primary); font-style: italic; border-top: 1px solid var(--border-light);">
            ${p.reason}
          </div>
        `;
      }
      
      // Add hover effect
      card.addEventListener('mouseover', () => { card.style.transform = 'translateY(-2px)'; card.style.boxShadow = '0 10px 15px -3px rgba(0, 0, 0, 0.1)'; });
      card.addEventListener('mouseout', () => { card.style.transform = 'translateY(0)'; card.style.boxShadow = '0 4px 6px -1px rgba(0, 0, 0, 0.1)'; });

      resultsGrid.appendChild(card);
    });

  } catch (e) {
    console.error("Failed to parse AI response as JSON", jsonString, e);
    resultsGrid.innerHTML = `
        <div style="grid-column: 1/-1; padding: 2rem; background: var(--surface); border-radius: 12px; border: 1px solid var(--border);">
            <h3 style="color: var(--text-dark); margin-bottom: 1rem;">Raw AI Output (Fallback):</h3>
            <pre style="white-space: pre-wrap; color: var(--text-light); font-family: inherit;">${jsonString}</pre>
        </div>`;
  }
}

// Sidebar toggle
const sidebarToggle = document.getElementById('sidebarToggle');
const sidebar = document.getElementById('sidebar');
if (sidebarToggle && sidebar) {
  sidebarToggle.addEventListener('click', () => {
    sidebar.classList.toggle('collapsed');
    sidebar.classList.toggle('open');
  });
}

// Health hook
const statusDot = document.getElementById('statusDot');
const statusText = document.getElementById('statusText');
async function checkHealth() {
  if (!statusDot || !statusText) return;
  try {
    const res = await fetch(`${API}/health`);
    if (res.ok) {
      statusDot.className  = 'status-dot online';
      statusText.textContent = 'DB Online'; // Changed from doc length since docs api is dead
    } else {
      throw new Error('offline');
    }
  } catch {
    statusDot.className   = 'status-dot offline';
    statusText.textContent = 'Server offline';
  }
}
setInterval(checkHealth, 30000);
checkHealth();
