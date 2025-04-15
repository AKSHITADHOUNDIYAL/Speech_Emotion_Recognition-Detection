import multiprocessing
import subprocess
import os
import time
import uvicorn

def run_streamlit():
    """Run the Streamlit frontend app"""
    command = ["streamlit", "run", "frontend/app.py", "--server.port", "5000", "--server.address", "0.0.0.0"]
    subprocess.run(command)

def run_fastapi():
    """Run the FastAPI backend server"""
    uvicorn.run("backend.api:app", host="0.0.0.0", port=8000, reload=False)

if __name__ == "__main__":
    # Create separate processes for each application
    streamlit_process = multiprocessing.Process(target=run_streamlit)
    fastapi_process = multiprocessing.Process(target=run_fastapi)
    
    # Start processes
    print("Starting Speech Emotion Recognition System...")
    print(" - Starting Streamlit frontend on port 5000")
    streamlit_process.start()
    
    print(" - Starting FastAPI backend on port 8000")
    fastapi_process.start()
    
    print("\nFrontend URL: http://localhost:5000")
    print("Backend API URL: http://localhost:8000")
    print("API documentation: http://localhost:8000/docs")
    
    try:
        # Keep the main process running
        while True:
            if not streamlit_process.is_alive() or not fastapi_process.is_alive():
                print("One or more processes stopped unexpectedly. Shutting down...")
                break
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        # Ensure all processes are terminated
        if streamlit_process.is_alive():
            streamlit_process.terminate()
        if fastapi_process.is_alive():
            fastapi_process.terminate()
            
        # Wait for processes to terminate
        streamlit_process.join()
        fastapi_process.join()
        
        print("Application shutdown complete.")
