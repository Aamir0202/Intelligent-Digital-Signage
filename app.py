from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import windows1  # Your custom module

from multiprocessing import Manager, Process

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["http://127.0.0.1:5000", "http://localhost:5000"]}})

shared_data = None  # Declare globally
windows_process = None  # Optional: to track your spawned process

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/receive-ip', methods=['POST'])
def receive_ip():
    global shared_data, windows_process

    data = request.get_json()
    ip = data.get('ip')
    print(f"Received IP: {ip}")

    if shared_data is not None:
        shared_data['ip'] = ip

        # Start process if not already running
        if windows_process is None or not windows_process.is_alive():
            windows_process = Process(target=windows1.start_windows, args=(shared_data,))
            windows_process.start()

        # Poll for up to 10 seconds to wait for video path
        import time
        timeout = 10
        start_time = time.time()
        while time.time() - start_time < timeout:
            video_path = shared_data['video_path'].value
            #print(f"In app.py shared data: {shared_data}")
            if video_path:  # If path is non-empty
                #print(f"app.py received video path: {video_path}")
                return jsonify({"video_path": video_path})
            time.sleep(0.5)  # Wait before checking again

        # If no video after timeout
        return jsonify({"message": "No video playing yet"}), 404
    else:
        return jsonify({"status": "error", "message": "Shared manager not initialized"}), 500


@app.route('/current-ad', methods=['GET'])
def current_ad():
    if shared_data is not None and 'video_path' in shared_data:
        return jsonify({"video_path": shared_data['video_path'].value})
    return jsonify({"video_path": ""})
    

if __name__ == '__main__':
    manager = Manager()
    shared_data = manager.dict()
    shared_data['video_path'] = manager.Value('u', '')  # Initialize shared video path

    # Start Flask server
    app.run(debug=True)
