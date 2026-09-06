mergeInto(LibraryManager.library, {
  CopyToClipboard: function (strPtr) {
    var str = UTF8ToString(strPtr);
    function fallback() {
      // execCommand works inside itch.io's cross-origin iframe where
      // navigator.clipboard is usually blocked.
      var ta = document.createElement('textarea');
      ta.value = str;
      ta.style.position = 'fixed';
      ta.style.top = '0';
      ta.style.left = '0';
      ta.style.opacity = '0';
      document.body.appendChild(ta);
      ta.focus();
      ta.select();
      try { document.execCommand('copy'); } catch (e) {}
      document.body.removeChild(ta);
    }
    if (navigator.clipboard && navigator.clipboard.writeText) {
      // Try the modern API first; if the iframe rejects it, fall back.
      navigator.clipboard.writeText(str).catch(fallback);
    } else {
      fallback();
    }
  }
});
