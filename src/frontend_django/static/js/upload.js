// Generic upload form handler

function setupUploadForm(formId, actionUrl, statusDivId) {
    const form = document.getElementById(formId);
    const statusDiv = document.getElementById(statusDivId);
    
    if (!form) return;
    
    form.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        const formData = new FormData(form);
        const fileInput = form.querySelector('input[type="file"]');
        
        if (!fileInput.files[0]) {
            statusDiv.innerHTML = '<div class="alert alert-danger">Please select a file</div>';
            statusDiv.style.display = 'block';
            return;
        }
        
        // Show upload status
        statusDiv.innerHTML = '<div class="alert alert-info">Uploading...</div>';
        statusDiv.style.display = 'block';
        
        try {
            const response = await fetch(actionUrl, {
                method: 'POST',
                body: formData,
            });
            
            const result = await response.json();
            
            if (response.ok && result.success) {
                statusDiv.innerHTML = '<div class="alert alert-success">✅ Upload successful!</div>';
                fileInput.value = '';
                
                // Reload page after 1.5 seconds to show updated list
                setTimeout(() => {
                    window.location.reload();
                }, 1500);
            } else {
                statusDiv.innerHTML = `<div class="alert alert-danger">❌ Error: ${result.error || 'Upload failed'}</div>`;
            }
        } catch (error) {
            statusDiv.innerHTML = `<div class="alert alert-danger">❌ Error: ${error.message}</div>`;
        }
    });
}

// Export for use in templates
window.setupUploadForm = setupUploadForm;
