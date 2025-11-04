async function postJSON(url, body) {
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body || {})
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || 'Request failed');
  return data;
}

async function uploadFile(url, file, onProgress) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    const formData = new FormData();
    formData.append('file', file);

    xhr.upload.addEventListener('progress', (event) => {
      if (event.lengthComputable && onProgress) {
        const percentComplete = Math.round((event.loaded / event.total) * 100);
        onProgress(percentComplete);
      }
    });

    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        try {
          resolve(JSON.parse(xhr.responseText));
        } catch (e) {
          resolve(xhr.responseText);
        }
      } else {
        let error;
        try {
          const err = JSON.parse(xhr.responseText);
          error = new Error(err.error || 'Upload failed');
        } catch (e) {
          error = new Error('Upload failed');
        }
        reject(error);
      }
    };

    xhr.onerror = () => {
      reject(new Error('Network error during upload'));
    };

    xhr.open('POST', url, true);
    xhr.send(formData);
  });
}

// Tab functionality
function openTab(evt, tabName) {
  // Hide all tab content
  const tabContents = document.getElementsByClassName('tabcontent');
  for (let i = 0; i < tabContents.length; i++) {
    tabContents[i].style.display = 'none';
  }

  // Remove active class from all tab buttons
  const tabButtons = document.getElementsByClassName('tablinks');
  for (let i = 0; i < tabButtons.length; i++) {
    tabButtons[i].className = tabButtons[i].className.replace(' active', '');
  }

  // Show the current tab and add active class to the button that opened the tab
  document.getElementById(tabName).style.display = 'block';
  evt.currentTarget.className += ' active';
}

const $ = (id) => document.getElementById(id);

$('btn-process').addEventListener('click', async () => {
  const url = $('yt-url').value.trim();
  $('process-status').textContent = 'Processing...';
  try {
    const data = await postJSON('/process', { url });
    $('process-status').textContent = `Done. ${data.chunks} chunks indexed.`;
  } catch (e) {
    $('process-status').textContent = 'Error: ' + e.message;
  }
});

$('btn-summarize').addEventListener('click', async () => {
  $('summary').textContent = 'Summarizing...';
  try {
    const data = await postJSON('/summarize', {});
    $('summary').textContent = data.summary || '';
  } catch (e) {
    $('summary').textContent = 'Error: ' + e.message;
  }
});

$('btn-process-text').addEventListener('click', async () => {
  const text = $('manual-text').value.trim();
  $('process-text-status').textContent = 'Processing...';
  try {
    const data = await postJSON('/process_text', { text });
    $('process-text-status').textContent = `Done. ${data.chunks} chunks indexed.`;
  } catch (e) {
    $('process-text-status').textContent = 'Error: ' + e.message;
  }
});

$('btn-ask').addEventListener('click', async () => {
  const question = $('question').value.trim();
  if (!question) return;
  $('answer').textContent = 'Thinking...';
  $('sources').textContent = '';
  try {
    const data = await postJSON('/ask', { question });
    $('answer').textContent = data.answer || '';
    if (data.sources && data.sources.length) {
      $('sources').textContent = 'Grounded in chunks: ' + data.sources.join(', ');
    }
  } catch (e) {
    $('answer').textContent = 'Error: ' + e.message;
  }
});

// File upload handling
const videoUpload = $('video-upload');
const btnUpload = $('btn-upload');
const fileName = $('file-name');
const uploadStatus = $('upload-status');
const progressContainer = $('progress-container');
const progressBarFill = $('progress-bar-fill');
const progressText = $('progress-text');

// Update file name when a file is selected
videoUpload.addEventListener('change', (e) => {
  const file = e.target.files[0];
  if (file) {
    fileName.textContent = file.name;
    btnUpload.disabled = false;
  } else {
    fileName.textContent = 'No file chosen';
    btnUpload.disabled = true;
  }
});

// Handle file upload
btnUpload.addEventListener('click', async () => {
  const file = videoUpload.files[0];
  if (!file) return;

  // Show progress bar
  progressContainer.style.display = 'block';
  progressBarFill.style.width = '0%';
  progressText.textContent = '0%';
  uploadStatus.textContent = 'Uploading and processing video...';
  btnUpload.disabled = true;

  try {
    // Update progress during upload
    const updateProgress = (percent) => {
      progressBarFill.style.width = `${percent}%`;
      progressText.textContent = `${percent}%`;
    };

    // Upload the file
    const data = await uploadFile('/upload_video', file, updateProgress);
    
    // Update UI
    uploadStatus.textContent = `Done. ${data.chunks} chunks indexed.`;
    progressBarFill.style.width = '100%';
    progressText.textContent = '100%';
    
    // Reset form after successful upload
    videoUpload.value = '';
    fileName.textContent = 'No file chosen';
    
    // Hide progress bar after a delay
    setTimeout(() => {
      progressContainer.style.display = 'none';
    }, 2000);
    
  } catch (e) {
    uploadStatus.textContent = 'Error: ' + e.message;
    progressContainer.style.display = 'none';
  } finally {
    btnUpload.disabled = false;
  }
});
