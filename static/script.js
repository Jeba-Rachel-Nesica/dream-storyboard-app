// Crisis modal
const crisisBtn = document.getElementById('crisisBtn');
const crisisModal = document.getElementById('crisisModal');
const closeCrisis = document.getElementById('closeCrisis');
if (crisisBtn) crisisBtn.onclick = ()=> crisisModal.classList.remove('hidden');
if (closeCrisis) closeCrisis.onclick = ()=> crisisModal.classList.add('hidden');

// First-time consent gate (gentle)
if (location.pathname !== '/consent') {
  const given = localStorage.getItem('consentGiven') === 'true';
  if (!given) window.location.replace('/consent');
}
