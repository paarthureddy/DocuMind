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
  searchBtn.textContent = 'Processing...';
  searchBtn.disabled = true;
  
  if (welcomeMessage) welcomeMessage.style.display = 'none';
  resultsGrid.innerHTML = '';
  loadingIndicator.style.display = 'block';

  try {
    // First classify intent locally to determine routing
    const intent = classifyIntentLocally(query);
    
    if (intent === 'calculator') {
      // Mathematical query - use chat endpoint
      const res = await fetch(`${API}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: query })
      });
      
      loadingIndicator.style.display = 'none';
      
      if (res.ok) {
        const data = await res.json();
        renderCalculationResult(data);
      } else {
        resultsGrid.innerHTML = `<div style="grid-column: 1/-1; color: red;">Error: ${res.statusText}</div>`;
      }
    } else if (intent === 'profile_search') {
      // Profile search - go directly to search API for structured data
      const searchRes = await fetch(`${API}/search_profiles`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          query: query,
          max_results: 10,
          enable_reranking: true
        })
      });
      
      loadingIndicator.style.display = 'none';
      
      if (searchRes.ok) {
        const searchData = await searchRes.json();
        renderProfiles(searchData);
      } else {
        resultsGrid.innerHTML = `<div style="grid-column: 1/-1; color: red;">Search error: ${searchRes.statusText}</div>`;
      }
    } else {
      // General chat - use chat endpoint
      const res = await fetch(`${API}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: query })
      });
      
      loadingIndicator.style.display = 'none';
      
      if (res.ok) {
        const data = await res.json();
        renderGeneralResponse(data);
      } else {
        resultsGrid.innerHTML = `<div style="grid-column: 1/-1; color: red;">Error: ${res.statusText}</div>`;
      }
    }
  } catch (err) {
    loadingIndicator.style.display = 'none';
    resultsGrid.innerHTML = `<div style="grid-column: 1/-1; color: red;">Network Error: ${err.message}. Ensure the backend is running.</div>`;
  }

  isSearching = false;
  searchBtn.textContent = 'Ask AI';
  searchBtn.disabled = false;
}

function classifyIntentLocally(query) {
  // Simple local intent classification to match backend
  const queryLower = query.toLowerCase().trim();
  
  // Check for pure math expressions
  if (/^[\d+\-*/().^ ]+$/.test(query) && !/[a-zA-Z]/.test(query)) {
    return 'calculator';
  }
  
  // Check for talent keywords
  const talentKeywords = [
    "actor", "actress", "model", "singer", "dancer", "performer", "artist",
    "talent", "casting", "cast", "role", "character", "villain", "hero",
    "protagonist", "antagonist", "lead", "supporting", "comic", "romantic",
    "male", "female", "gender", "height", "feet", "ft", "inch", "in", "cm",
    "tall", "short", "complexion", "brown", "fair", "dark", "light",
    "wheatish", "handsome", "beautiful", "appearance", "looks",
    "skills", "experience", "craft", "profession", "search", "find",
    "looking for", "need", "want", "intense", "charming", "aggressive",
    "soft", "dominant", "comic", "funny", "serious", "dramatic"
  ];
  
  if (talentKeywords.some(keyword => queryLower.includes(keyword))) {
    return 'profile_search';
  }
  
  return 'chat';
}

function isTalentSearchQuery(query) {
  const talentKeywords = [
    "actor", "actress", "model", "talent", "casting", "cast", "role",
    "character", "villain", "hero", "lead", "supporting", "comic",
    "male", "female", "height", "complexion", "appearance", "skills",
    "experience", "profile", "search", "find", "looking for", "need",
    "feet", "ft", "inch", "in", "cm", "tall", "short", "brown", "fair",
    "dark", "light", "wheatish", "handsome", "beautiful", "intense",
    "charming", "aggressive", "soft", "dominant", "comic", "funny"
  ];
  
  const queryLower = query.toLowerCase();
  return talentKeywords.some(keyword => queryLower.includes(keyword));
}

function renderCalculationResult(data) {
  const card = document.createElement('div');
  card.className = 'profile-card';
  card.style.cssText = `
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 2rem;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    text-align: center;
    grid-column: 1/-1;
  `;
  
  card.innerHTML = `
    <div style="font-size: 3rem; margin-bottom: 1rem;">🧮</div>
    <h3 style="color: var(--text-dark); margin-bottom: 1rem;">Calculation Result</h3>
    <div style="font-size: 1.5rem; color: var(--primary); font-weight: bold; margin-bottom: 1rem;">
      ${data.answer}
    </div>
    <div style="color: var(--text-light); font-size: 0.9rem;">
      Intent: ${data.intent} | ✓ Calculated using AI calculator
    </div>
  `;
  
  resultsGrid.appendChild(card);
}

function renderGeneralResponse(data) {
  const card = document.createElement('div');
  card.className = 'profile-card';
  card.style.cssText = `
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 2rem;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    grid-column: 1/-1;
  `;
  
  const toolIcon = data.tools_used && data.tools_used.includes('web_search') ? '🌐' : 
                   data.tools_used && data.tools_used.includes('profile_search') ? '🎭' : '🤖';
  
  card.innerHTML = `
    <div style="display: flex; align-items: flex-start; gap: 1rem;">
      <div style="font-size: 2rem;">${toolIcon}</div>
      <div style="flex: 1;">
        <h3 style="color: var(--text-dark); margin-bottom: 1rem;">AI Assistant Response</h3>
        <div style="color: var(--text); line-height: 1.6; white-space: pre-wrap;">
          ${data.answer}
        </div>
        <div style="margin-top: 1rem; display: flex; gap: 1rem; align-items: center; flex-wrap: wrap;">
          ${data.tools_used && data.tools_used.length > 0 ? `
          <div style="color: var(--text-light); font-size: 0.8rem;">
            Tools used: ${data.tools_used.join(', ')}
          </div>
          ` : ''}
          <div style="color: var(--text-light); font-size: 0.8rem;">
            Intent: ${data.intent || 'chat'}
          </div>
        </div>
      </div>
    </div>
  `;
  
  resultsGrid.appendChild(card);
}

if (searchBtn) searchBtn.addEventListener('click', executeSearch);
if (searchInput) {
  searchInput.addEventListener('keydown', e => {
    if (e.key === 'Enter') executeSearch();
  });
}

function renderProfiles(searchData) {
  try {
    if (!searchData.success || !searchData.results || searchData.results.length === 0) {
      resultsGrid.innerHTML = `
        <div class="col-span-full py-20 text-center text-slate-500">
          <div class="text-6xl mb-4">🎭</div>
          <h3 class="text-2xl font-bold text-slate-700">No matching profiles found</h3>
          <p class="mt-2">Try adjusting your search terms or broadening your criteria.</p>
        </div>
      `;
      return;
    }

    const profiles = searchData.results;

    profiles.forEach(p => {
      const card = document.createElement('div');
      card.className = 'bg-white rounded-2xl border border-slate-200 p-6 flex flex-col shadow hover:shadow-xl transition-all duration-300 transform hover:-translate-y-1 relative group';
      
      const genderIcon = p.metadata?.gender === 'female' ? '♀️' : p.metadata?.gender === 'male' ? '♂️' : '⚪';
      const genderBg = p.metadata?.gender === 'female' ? 'bg-pink-500' : p.metadata?.gender === 'male' ? 'bg-blue-500' : 'bg-slate-500';

      // Past roles extraction and badges
      const pastRoles = p.metadata?.past_roles || [];
      const pastRolesHTML = pastRoles.length > 0 
        ? pastRoles.slice(0, 4).map(role => `<span class="bg-indigo-50 text-indigo-700 font-medium px-2.5 py-0.5 rounded-full text-xs border border-indigo-100 flex-shrink-0 whitespace-nowrap">${role}</span>`).join('') + (pastRoles.length > 4 ? `<span class="text-xs text-slate-400 italic">...</span>` : '')
        : '<span class="text-xs text-slate-400 italic">No past roles listed</span>';

      // Craft & Subcrafts
      const mainCraft = p.metadata?.craft || 'Talent';
      const subcrafts = p.metadata?.subcrafts || [];
      const subcraftText = subcrafts.length > 0 ? `<br><span class="text-xs font-normal text-slate-500">${subcrafts.slice(0, 2).join(', ')}</span>` : '';

      const scorePct = typeof p.score === 'number' ? p.score : 0;
      const genderLabel = p.metadata?.gender ? p.metadata.gender.charAt(0).toUpperCase() + p.metadata.gender.slice(1) : 'N/A';
      const expYears = p.metadata?.experience_years;
      const expLabel = expYears ? (expYears >= 10 ? `${expYears}y (Senior)` : expYears >= 5 ? `${expYears}y (Mid)` : `${expYears}y`) : 'N/A';

      card.innerHTML = `
        <div class="absolute top-4 right-4 bg-gradient-to-r from-emerald-400 to-green-500 text-white text-sm font-bold px-3 py-1 rounded-full shadow-sm">
          ${scorePct}% Match
        </div>

        <div class="flex items-start gap-4 mb-5 pb-5 border-b border-slate-100">
          <div class="relative flex-shrink-0">
            <div class="w-16 h-16 rounded-full bg-slate-100 border-2 border-slate-200 flex items-center justify-center text-slate-300 overflow-hidden">
               <svg class="w-10 h-10" fill="currentColor" viewBox="0 0 24 24"><path d="M24 20.993V24H0v-2.996A14.977 14.977 0 0112.004 15c4.904 0 9.26 2.354 11.996 5.993zM16.002 8.999a4 4 0 11-8 0 4 4 0 018 0z" /></svg>
            </div>
            <div class="absolute -bottom-1 -right-1 ${genderBg} text-white w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold border-2 border-white shadow-sm">${genderIcon}</div>
          </div>
          
          <div class="flex-1 pr-14 min-w-0">
            <h3 class="text-xl font-bold text-slate-800 truncate" title="${p.name || 'Unknown'}">${p.name || 'Unknown'}</h3>
            <div class="inline-block mt-1 bg-slate-100 text-slate-700 px-2.5 py-1 rounded-md text-sm font-semibold uppercase tracking-wide border border-slate-200">
              ${mainCraft} ${subcraftText}
            </div>
          </div>
        </div>
        
        <div class="grid grid-cols-2 gap-3 mb-5 text-sm text-slate-600">
          <div class="flex items-center gap-2">
            <span class="text-slate-400">${genderIcon}</span>
            <span class="font-medium truncate"><strong class="font-semibold text-slate-700">Gender:</strong> ${genderLabel}</span>
          </div>
          <div class="flex items-center gap-2">
            <span class="text-slate-400">📏</span>
            <span class="font-medium truncate"><strong class="font-semibold text-slate-700">Height:</strong> ${p.metadata?.height_cm ? p.metadata.height_cm+'cm' : 'N/A'}</span>
          </div>
          <div class="flex items-center gap-2">
            <span class="text-slate-400">🎨</span>
            <span class="font-medium truncate"><strong class="font-semibold text-slate-700">Skin:</strong> <span class="capitalize">${p.metadata?.complexion || 'N/A'}</span></span>
          </div>
          <div class="flex items-center gap-2">
            <span class="text-slate-400">💼</span>
            <span class="font-medium truncate"><strong class="font-semibold text-slate-700">Experience:</strong> ${expLabel}</span>
          </div>
          <div class="flex items-center gap-2">
            <span class="text-slate-400">⭐</span>
            <span class="font-medium truncate"><strong class="font-semibold text-slate-700">Rating:</strong> ${p.metadata?.rating_average ? parseFloat(p.metadata.rating_average).toFixed(1) : 'N/A'}</span>
          </div>
          ${p.metadata?.age ? `<div class="flex items-center gap-2">
            <span class="text-slate-400">🎂</span>
            <span class="font-medium truncate"><strong class="font-semibold text-slate-700">Age:</strong> ${p.metadata.age} yrs</span>
          </div>` : ''}
        </div>

        <div class="mb-5">
          <h4 class="text-xs uppercase font-bold text-slate-400 tracking-wider mb-2 flex items-center gap-1">
            <svg class="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 4v16M17 4v16M3 8h4m10 0h4M3 12h18M3 16h4m10 0h4M4 20h16a1 1 0 001-1V5a1 1 0 00-1-1H4a1 1 0 00-1 1v14a1 1 0 001 1z" /></svg>
            Past Roles
          </h4>
          <div class="flex flex-wrap gap-1.5 items-center">
            ${pastRolesHTML}
          </div>
        </div>
        
        <div class="mt-auto pt-4 border-t border-slate-100">
          <div class="flex flex-wrap gap-2 mb-3">
            ${p.metadata?.rating_average ? `<span class="bg-yellow-50 text-yellow-700 px-2 py-0.5 rounded font-medium text-xs border border-yellow-200">⭐ ${parseFloat(p.metadata.rating_average).toFixed(1)} Rating</span>` : ''}
            ${p.metadata?.gender ? `<span class="bg-pink-50 text-pink-700 px-2 py-0.5 rounded font-medium text-xs border border-pink-200">${genderIcon} ${genderLabel}</span>` : ''}
            ${expYears ? `<span class="bg-green-50 text-green-700 px-2 py-0.5 rounded font-medium text-xs border border-green-200">💼 ${expYears}y Exp</span>` : ''}
            ${p.metadata?.age ? `<span class="bg-blue-50 text-blue-700 px-2 py-0.5 rounded font-medium text-xs border border-blue-200">👤 ${p.metadata.age} yrs</span>` : ''}
            ${pastRoles.length > 0 ? `<span class="bg-purple-50 text-purple-700 px-2 py-0.5 rounded font-medium text-xs border border-purple-200">🎭 ${pastRoles[0]}</span>` : ''}
          </div>
          <div class="bg-indigo-50/50 rounded-lg p-3 border-l-4 border-indigo-400 text-sm italic text-slate-600">
            <span class="font-bold text-indigo-700 not-italic block mb-0.5">💡 Why this match?</span>
            ${p.explanation || 'Semantic and physical traits matched your query.'}
          </div>
        </div>
      `;
      
      resultsGrid.appendChild(card);
    });

    if (profiles.length > 0) {
      const summaryCard = document.createElement('div');
      summaryCard.className = 'col-span-full text-center py-6 text-slate-400 font-medium text-sm border-t border-slate-200 mt-4';
      summaryCard.innerHTML = `Found ${profiles.length} matching profiles • Sorted by relevance score`;
      resultsGrid.appendChild(summaryCard);
    }
  } catch (e) {
    console.error("Failed to render", e);
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
      statusText.textContent = 'AI Assistant Ready';
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
