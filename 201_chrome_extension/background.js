// ===== Background Service Worker =====

// Open Side Panel when extension icon is clicked
chrome.action.onClicked.addListener((tab) => {
    chrome.sidePanel.open({ windowId: tab.windowId });
});

// Set Side Panel behavior - open on action click
chrome.sidePanel.setPanelBehavior({ openPanelOnActionClick: true })
    .catch((error) => console.error('Error setting panel behavior:', error));

// Log when service worker starts
console.log('LLM Extractor background service worker started');
