(function () {
  var storageKey = 'feynrl-theme';
  var root = document.documentElement;

  function getStoredTheme() {
    try {
      return localStorage.getItem(storageKey);
    } catch (error) {
      return null;
    }
  }

  function storeTheme(theme) {
    try {
      localStorage.setItem(storageKey, theme);
    } catch (error) {
      // Ignore storage failures and keep the current theme in-memory only.
    }
  }

  function applyTheme(theme) {
    root.setAttribute('data-theme', theme);
    if (toggle) {
      toggle.setAttribute('aria-label', theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode');
      toggle.title = theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode';
    }
  }

  function nextTheme() {
    return root.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
  }

  function makeOrb(button, theme) {
    var rect = button.getBoundingClientRect();
    var orb = document.createElement('span');
    var startX = rect.left;
    var startY = rect.top;
    var endX = window.innerWidth * 0.12;
    var endY = window.innerHeight * 0.18;
    var diameter = Math.max(window.innerWidth, window.innerHeight) * 2.4;

    orb.className = 'theme-orb';
    orb.setAttribute('aria-hidden', 'true');
    orb.style.setProperty('--theme-orb-color', theme === 'dark' ? '#111111' : '#f5f1ea');
    orb.style.setProperty('--theme-orb-start-x', startX + 'px');
    orb.style.setProperty('--theme-orb-start-y', startY + 'px');
    orb.style.setProperty('--theme-orb-arc-x', Math.max(startX - 56, 18) + 'px');
    orb.style.setProperty('--theme-orb-arc-y', Math.max(startY - 84, 18) + 'px');
    orb.style.setProperty('--theme-orb-end-x', endX + 'px');
    orb.style.setProperty('--theme-orb-end-y', endY + 'px');
    orb.style.setProperty('--theme-orb-scale', String(diameter / rect.width));

    orb.addEventListener('animationend', function () {
      orb.remove();
    });

    document.body.appendChild(orb);
  }

  var storedTheme = getStoredTheme();
  var initialTheme = storedTheme === 'dark' ? 'dark' : 'light';
  var toggle = null;

  applyTheme(initialTheme);

  document.addEventListener('DOMContentLoaded', function () {
    toggle = document.createElement('button');
    toggle.className = 'theme-toggle';
    toggle.type = 'button';
    toggle.setAttribute('aria-live', 'polite');
    applyTheme(root.getAttribute('data-theme') || initialTheme);

    toggle.addEventListener('click', function () {
      var theme = nextTheme();
      toggle.classList.add('is-hidden');
      makeOrb(toggle, theme);

      window.setTimeout(function () {
        applyTheme(theme);
        storeTheme(theme);
      }, 200);

      window.setTimeout(function () {
        toggle.classList.remove('is-hidden');
      }, 760);
    });

    document.body.appendChild(toggle);
  });
})();
