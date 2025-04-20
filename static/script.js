document.addEventListener('DOMContentLoaded', () => {
  const adVideo = document.getElementById('adVideo');
  const adSource = document.getElementById('adSource');
  const ipInput = document.getElementById('ipInput');
  const submitBtn = document.getElementById('submitBtn');

  let currentVideoPath = ''; // Keep track of the current video path

  submitBtn.addEventListener('click', () => {
    const ip = ipInput.value.trim();

    if (ip !== '') {
      fetch('http://localhost:5000/receive-ip', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ ip: ip })
      })
      .then(response => response.json())
      .then(data => {
        if (data.video_path) {
          const newVideoPath = data.video_path.replace(/\\/g, '/');
          if (currentVideoPath !== newVideoPath) {
            adSource.src = `/static/${newVideoPath}`;
            adVideo.load();
            adVideo.play();
            currentVideoPath = newVideoPath;
          }
        } else {
          console.log('No video_path returned yet.');
        }
      })
      .catch(error => {
        console.error('Error sending IP to backend:', error);
      });
    } else {
      alert('Please enter a valid IP address.');
    }
  });

  // Polling logic
  setInterval(() => {
    fetch('http://localhost:5000/current-ad')
      .then(response => response.json())
      .then(data => {
        if (data.video_path) {
          const newVideoPath = data.video_path.replace(/\\/g, '/');
          if (currentVideoPath !== newVideoPath) {
            console.log('Switching to new ad:', newVideoPath);
            adSource.src = `/static/${newVideoPath}`;
            adVideo.load();
            adVideo.play();
            currentVideoPath = newVideoPath;
          } else {
            // Same video path, no action needed
            console.log('Same ad, continuing playback...');
          }
        }
      })
      .catch(error => {
        console.error('Error fetching current ad:', error);
      });
  }, 5000);
});
