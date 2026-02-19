// Clone Hero Manager - Main JavaScript
// Utility functions and global initialization

// ---------------------------------------------------------------------------
// Initialization
// ---------------------------------------------------------------------------
document.addEventListener('DOMContentLoaded', function () {
    console.log('🎸 Clone Hero Manager loaded');

    // Fade-in animation for main content
    const mainContent = document.querySelector('.main-content');
    if (mainContent) {
        mainContent.style.opacity = '0';
        setTimeout(() => {
            mainContent.style.transition = 'opacity 0.4s ease-in';
            mainContent.style.opacity = '1';
        }, 50);
    }

    // Auto-dismiss flash messages after 6 seconds
    const alerts = document.querySelectorAll('.main-content > .alert');
    alerts.forEach(alert => {
        setTimeout(() => {
            alert.style.transition = 'opacity 0.4s, transform 0.4s';
            alert.style.opacity = '0';
            alert.style.transform = 'translateY(-10px)';
            setTimeout(() => alert.remove(), 400);
        }, 6000);
    });

    // Highlight the active sidebar link based on the current path
    highlightActiveSidebarLink();
});

// ---------------------------------------------------------------------------
// Notifications
// ---------------------------------------------------------------------------

/**
 * Show a temporary floating notification in the top-right corner.
 *
 * @param {string} message - The notification text.
 * @param {string} type    - One of: 'info', 'success', 'warning', 'danger'.
 * @param {number} duration - How long to show the notification (ms). Default 4000.
 */
function showNotification(message, type, duration) {
    type = type || 'info';
    duration = duration || 4000;

    // Create the container if it doesn't exist yet
    var container = document.getElementById('notification-container');
    if (!container) {
        container = document.createElement('div');
        container.id = 'notification-container';
        container.style.cssText =
            'position:fixed;top:20px;right:20px;z-index:9999;' +
            'display:flex;flex-direction:column;gap:10px;' +
            'max-width:400px;width:90%;pointer-events:none;';
        document.body.appendChild(container);
    }

    var alertDiv = document.createElement('div');
    alertDiv.className = 'alert alert-' + type;
    alertDiv.style.cssText =
        'pointer-events:auto;opacity:0;transform:translateX(30px);' +
        'transition:opacity 0.3s,transform 0.3s;' +
        'box-shadow:0 4px 12px rgba(0,0,0,0.15);margin:0;';
    alertDiv.textContent = message;

    container.appendChild(alertDiv);

    // Trigger entrance animation
    requestAnimationFrame(function () {
        requestAnimationFrame(function () {
            alertDiv.style.opacity = '1';
            alertDiv.style.transform = 'translateX(0)';
        });
    });

    // Schedule removal
    setTimeout(function () {
        alertDiv.style.opacity = '0';
        alertDiv.style.transform = 'translateX(30px)';
        setTimeout(function () {
            alertDiv.remove();
            // Remove the container if empty
            if (container && container.children.length === 0) {
                container.remove();
            }
        }, 300);
    }, duration);
}

// ---------------------------------------------------------------------------
// Sidebar
// ---------------------------------------------------------------------------

/**
 * Highlight the sidebar link that matches the current URL path.
 */
function highlightActiveSidebarLink() {
    var path = window.location.pathname;
    var links = document.querySelectorAll('.sidebar-nav a');

    links.forEach(function (link) {
        link.classList.remove('active');
        var href = link.getAttribute('href');
        if (!href) return;

        // Exact match for home, prefix match for everything else
        if (href === '/' && path === '/') {
            link.classList.add('active');
        } else if (href !== '/' && path.startsWith(href)) {
            link.classList.add('active');
        }
    });
}

// ---------------------------------------------------------------------------
// Confirm helpers
// ---------------------------------------------------------------------------

/**
 * Show a confirmation dialog and call the callback if confirmed.
 *
 * @param {string}   message  - The confirmation prompt.
 * @param {Function} callback - Called (no args) if the user confirms.
 */
function confirmAction(message, callback) {
    if (confirm(message)) {
        callback();
    }
}

// ---------------------------------------------------------------------------
// Fetch helpers
// ---------------------------------------------------------------------------

/**
 * Make a JSON API call and return the parsed response.
 * Throws on network errors or non-OK responses.
 *
 * @param {string} url     - The endpoint URL.
 * @param {object} options - Fetch options (method, body, headers, etc.).
 * @returns {Promise<object>}
 */
async function apiFetch(url, options) {
    options = options || {};

    // Default to JSON content-type for non-FormData bodies
    if (options.body && !(options.body instanceof FormData)) {
        options.headers = Object.assign(
            { 'Content-Type': 'application/json' },
            options.headers || {}
        );
        if (typeof options.body === 'object') {
            options.body = JSON.stringify(options.body);
        }
    }

    var response = await fetch(url, options);
    var data;
    try {
        data = await response.json();
    } catch (e) {
        data = { detail: 'Invalid server response' };
    }

    if (!response.ok) {
        var errMsg = data.detail || data.error || 'Request failed';
        throw new Error(errMsg);
    }

    return data;
}

// ---------------------------------------------------------------------------
// Formatting utilities
// ---------------------------------------------------------------------------

/**
 * Format a byte count into a human-readable string (e.g. "3.2 MB").
 *
 * @param {number} bytes - The byte count.
 * @returns {string}
 */
function formatFileSize(bytes) {
    if (!bytes || bytes === 0) return '0 B';
    var k = 1024;
    var sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
    var i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

/**
 * Format seconds into a MM:SS string.
 *
 * @param {number} seconds
 * @returns {string}
 */
function formatDuration(seconds) {
    if (!seconds && seconds !== 0) return '--:--';
    var mins = Math.floor(seconds / 60);
    var secs = Math.round(seconds % 60);
    return mins + ':' + (secs < 10 ? '0' : '') + secs;
}

/**
 * Truncate a string to a maximum length, adding an ellipsis if needed.
 *
 * @param {string} str
 * @param {number} maxLen
 * @returns {string}
 */
function truncate(str, maxLen) {
    maxLen = maxLen || 60;
    if (!str) return '';
    return str.length > maxLen ? str.substring(0, maxLen - 1) + '…' : str;
}

// ---------------------------------------------------------------------------
// Keyboard shortcuts
// ---------------------------------------------------------------------------
document.addEventListener('keydown', function (e) {
    // Ctrl/Cmd + K  →  focus the search input (if one exists on the page)
    if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
        var searchInput = document.querySelector('.search-input, input[name="search"]');
        if (searchInput) {
            e.preventDefault();
            searchInput.focus();
            searchInput.select();
        }
    }
});

// ---------------------------------------------------------------------------
// Exports (make utilities available globally)
// ---------------------------------------------------------------------------
window.showNotification = showNotification;
window.confirmAction = confirmAction;
window.apiFetch = apiFetch;
window.formatFileSize = formatFileSize;
window.formatDuration = formatDuration;
window.truncate = truncate;
