const APP_URL = "http://localhost:8501/";   // summarizer web UI address

function canonicalYouTubeUrl(raw) {
  try {
    const u = new URL(raw);
    let id = null;
    if (u.hostname === "youtu.be") id = u.pathname.slice(1);
    else if (/(^|\.)youtube\.com$/.test(u.hostname)) {
      if (u.pathname === "/watch") id = u.searchParams.get("v");
      else {
        const m = u.pathname.match(/^\/(shorts|embed|live)\/([\w-]{11})/);
        if (m) id = m[2];
      }
    }
    return id && /^[\w-]{11}$/.test(id) ? `https://www.youtube.com/watch?v=${id}` : null;
  } catch { return null; }
}

function registerMenu() {
  chrome.contextMenus.removeAll().then(() => {
    chrome.contextMenus.create({
      id: "yt-transcribe",
      title: "YouTube transcribe",
      contexts: ["video", "page", "link"],
      documentUrlPatterns: ["*://*.youtube.com/*"]
    }, () => {
      if (chrome.runtime.lastError) {
        console.error("registerMenu(): creation FAILED:", chrome.runtime.lastError.message);
      } else {
        console.log("registerMenu(): menu item created OK");
      }
    });
  });
}

chrome.contextMenus.onClicked.addListener((info) => {
  if (info.menuItemId !== "yt-transcribe") return;
  const url = canonicalYouTubeUrl(info.linkUrl) || canonicalYouTubeUrl(info.pageUrl);
  if (!url) return;
  chrome.windows.create({ url: `${APP_URL}?url=${encodeURIComponent(url)}` });
});

chrome.runtime.onInstalled.addListener(registerMenu);
registerMenu();
