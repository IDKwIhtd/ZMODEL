
from fastapi import FastAPI
import subprocess
import time
from threading import Thread, Lock
import uvicorn

app = FastAPI()

results_data = {}
results_lock = Lock()

@app.post("/update_results/")
def update_results(results: dict):
    global results_data
    with results_lock:
        results_data = results 
        print("Received results:", results_data)
    return {"status": "Success", "data": results_data}

@app.get("/get_results/")
def get_results():
    with results_lock:
        return {"status": "Success", "data": results_data}

@app.get("/")
def test():
    return {"message": "Server is running"}

def run_script():
    while True:
        subprocess.run(["python", "ZMODEL.py"])
        time.sleep(60)

if __name__ == "__main__":
    # FastAPI 서버를 별도의 스레드에서 실행
    server_thread = Thread(target=uvicorn.run, args=(app,), kwargs={"host": "0.0.0.0", "port": 8000})
    server_thread.start()

    # 스크립트 실행
    run_script()
