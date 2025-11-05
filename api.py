import os
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass
import json
import shutil
import subprocess
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from typing import Optional

# Optional MongoDB / GridFS
MONGODB_URI = os.environ.get('MONGODB_URI')
MONGODB_DB = os.environ.get('MONGODB_DB', 'droneTracking')
mongo_client = None
grid_fs = None
mongo_collection = None
try:
    if MONGODB_URI:
        from pymongo import MongoClient
        from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
        import gridfs
        mongo_client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
        # ping to verify connection
        try:
            mongo_client.admin.command('ping')
            print("✅ Connected to MongoDB Atlas successfully!")
        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            print(f"❌ Could not connect to MongoDB Atlas: {e}")
            mongo_client = None
        if mongo_client is not None:
            mongo_db = mongo_client[MONGODB_DB]
            grid_fs = gridfs.GridFS(mongo_db)
            mongo_collection = mongo_db["processingVideos"]
    else:
        print("ℹ️ MONGODB_URI not set; skipping MongoDB Atlas connection.")
except Exception as e:
    print(f"❌ MongoDB initialization error: {e}")
    mongo_client = None
    grid_fs = None
    mongo_collection = None
from fastapi.middleware.cors import CORSMiddleware

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, 'uploads')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

app = FastAPI(title="Drone Detection & Tracking API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    filename = file.filename
    if not filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    ext = os.path.splitext(filename)[1].lower()
    if ext not in [".mp4", ".avi", ".mov"]:
        raise HTTPException(status_code=400, detail="Invalid file type. Allowed: MP4, AVI, MOV")

    ts = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
    saved_name = f"video_{ts}{ext}"
    save_path = os.path.join(UPLOAD_DIR, saved_name)

    with open(save_path, "wb") as out:
        shutil.copyfileobj(file.file, out)

    # Store in GridFS if configured
    input_grid_id: Optional[str] = None
    if grid_fs is not None:
        try:
            # re-open to stream from disk to GridFS
            with open(save_path, 'rb') as fh:
                grid_id = grid_fs.put(
                    fh,
                    filename=saved_name,
                    content_type='video/mp4' if ext == '.mp4' else 'application/octet-stream',
                    metadata={"type": "input"}
                )
                input_grid_id = str(grid_id)
        except Exception:
            input_grid_id = None

    return {"filename": saved_name, "path": save_path, "input_id": input_grid_id}


@app.post("/process")
async def process_video(filename: str = Form(...), input_id: str = Form(None)):
    input_path = os.path.join(UPLOAD_DIR, filename)
    if not os.path.isfile(input_path):
        raise HTTPException(status_code=404, detail="Uploaded file not found")

    out_name = os.path.splitext(filename)[0] + "_processed.mp4"
    output_path = os.path.join(OUTPUT_DIR, out_name)

    script_path = os.path.join(BASE_DIR, 'demo_detect_track.py')
    if not os.path.isfile(script_path):
        raise HTTPException(status_code=500, detail="Processing script not found")

    # Call the script non-interactively, no GUI, with input/output paths
    try:
        python_exec = os.path.join(BASE_DIR, 'venv', 'Scripts', 'python.exe') if os.name == 'nt' else 'python'
        result = subprocess.run(
            [
                python_exec,
                script_path,
                '--input', input_path,
                '--output', output_path,
                '--no-gui'
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start processing: {e}")

    metrics_path = output_path + '.json'
    response = {
        "output": out_name,
        "metrics": None,
        "stdout": result.stdout[-2000:] if result.stdout else "",
        "stderr": result.stderr[-2000:] if result.stderr else "",
        "drone_detected": False,
        "detection_probability": 0,
        "alert": False
    }
    mode = "Unknown"
    if os.path.isfile(metrics_path):
        # return only the filename to avoid client path parsing issues across OSes
        response["metrics"] = os.path.basename(metrics_path)
        # compute simple alert signal from metrics contents
        try:
            with open(metrics_path, 'r', encoding='utf-8') as f:
                m = json.load(f)
            alert = False
            alert_reason = None
            mode = m.get('video_type', "Unknown") if isinstance(m, dict) else "Unknown"
            # Include detection status and probability from metrics
            drone_detected = m.get("drone_detected", False)
            detection_probability = m.get("detection_probability", 0)
            
            # Alert based on detection probability > 20%
            if detection_probability > 20:
                alert = True
                alert_reason = f"Drone detected with {detection_probability}% probability"
            
            response["alert"] = alert
            if alert_reason:
                response["alert_reason"] = alert_reason
            response["metrics_obj"] = m
            response["video_type"] = mode
            response["drone_detected"] = drone_detected
            response["detection_probability"] = detection_probability
        except Exception:
            # if metrics parsing fails, skip alert fields
            pass

    if not os.path.isfile(output_path):
        raise HTTPException(status_code=500, detail="Processing did not produce output video")

    # Save processed video to GridFS if configured
    if grid_fs is not None:
        try:
            with open(output_path, 'rb') as fh:
                out_id = grid_fs.put(
                    fh,
                    filename=out_name,
                    content_type='video/mp4',
                    metadata={"type": "output", "source": filename, "video_type": mode}
                )
                response["output_id"] = str(out_id)
        except Exception:
            pass

    # Persist a record linking input/output in collection
    if mongo_collection is not None:
        try:
            from bson import ObjectId
            doc = {
                "createdAt": datetime.utcnow(),
                "input": {
                    "filename": filename,
                    "id": ObjectId(input_id) if input_id else None
                },
                "output": {
                    "filename": out_name,
                    "id": ObjectId(response.get("output_id")) if response.get("output_id") else None
                },
                "metrics": response.get("metrics_obj"),
                "alert": response.get("alert", False),
                "alert_reason": response.get("alert_reason"),
                "video_type": mode
            }
            # Remove None IDs to avoid invalid ObjectId
            if doc["input"]["id"] is None:
                del doc["input"]["id"]
            if doc["output"]["id"] is None:
                del doc["output"]["id"]
            inserted = mongo_collection.insert_one(doc)
            response["record_id"] = str(inserted.inserted_id)
        except Exception:
            pass

    return JSONResponse(response)


@app.get("/download/video/{name}")
async def download_video(name: str, request: Request):
    path = os.path.join(OUTPUT_DIR, name)
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="File not found")

    range_header = request.headers.get('range')
    file_size = os.path.getsize(path)
    if range_header is None:
        # No range requested; serve full file
        headers = {
            "Accept-Ranges": "bytes",
            "Content-Disposition": f"inline; filename=\"{name}\"",
        }
        return FileResponse(path, media_type="video/mp4", filename=name, headers=headers)

    # Parse Range: bytes=start-end
    try:
        units, _, rng = range_header.partition("=")
        if units != "bytes":
            raise ValueError("Invalid units")
        start_str, _, end_str = rng.partition("-")
        start = int(start_str) if start_str else 0
        end = int(end_str) if end_str else file_size - 1
        start = max(0, start)
        end = min(end, file_size - 1)
        if start > end:
            raise ValueError("Invalid range")
    except Exception:
        # Malformed Range header
        raise HTTPException(status_code=416, detail="Invalid range header")

    chunk_size = (end - start) + 1

    def iter_file(file_path: str, offset: int, length: int, block_size: int = 1024 * 1024):
        with open(file_path, 'rb') as f:
            f.seek(offset)
            remaining = length
            while remaining > 0:
                read_size = min(block_size, remaining)
                data = f.read(read_size)
                if not data:
                    break
                remaining -= len(data)
                yield data

    headers = {
        "Content-Range": f"bytes {start}-{end}/{file_size}",
        "Accept-Ranges": "bytes",
        "Content-Length": str(chunk_size),
        "Content-Type": "video/mp4",
        "Content-Disposition": f"inline; filename=\"{name}\"",
    }
    return StreamingResponse(iter_file(path, start, chunk_size), status_code=206, headers=headers)

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from bson import ObjectId

@app.get("/download/video/byid/{file_id}")
async def download_video_by_id(file_id: str, request: Request):
    if grid_fs is None:
        raise HTTPException(status_code=503, detail="GridFS not configured")

    try:
        f = grid_fs.get(ObjectId(file_id))
    except Exception:
        raise HTTPException(status_code=404, detail="File not found")

    size = f.length
    range_header = request.headers.get("range")

    # Serve full video if no Range header
    if not range_header:
        headers = {
            "Accept-Ranges": "bytes",
            "Content-Disposition": f'inline; filename="{f.filename or "video.mp4"}"',
        }
        def full_iter():
            f.seek(0)
            chunk = f.read(1024 * 1024)
            while chunk:
                yield chunk
                chunk = f.read(1024 * 1024)
        return StreamingResponse(full_iter(), media_type=f.content_type or "video/mp4", headers=headers)

    # Serve partial content for range requests
    try:
        units, _, rng = range_header.partition("=")
        if units != "bytes":
            raise ValueError("Only bytes range supported")
        start_str, _, end_str = rng.partition("-")
        start = int(start_str) if start_str else 0
        end = int(end_str) if end_str else size - 1
        start = max(0, start)
        end = min(end, size - 1)
        if start > end:
            raise ValueError
    except Exception:
        raise HTTPException(status_code=416, detail="Invalid Range header")

    length = end - start + 1
    f.seek(start)

    def iter_bytes():
        remaining = length
        while remaining > 0:
            chunk = f.read(min(1024 * 1024, remaining))
            if not chunk:
                break
            remaining -= len(chunk)
            yield chunk

    headers = {
        "Content-Range": f"bytes {start}-{end}/{size}",
        "Accept-Ranges": "bytes",
        "Content-Length": str(length),
        "Content-Type": f.content_type or "video/mp4",
        "Content-Disposition": f'inline; filename="{f.filename or "video.mp4"}"',
    }

    return StreamingResponse(iter_bytes(), status_code=206, headers=headers)



@app.get("/videos")
async def list_videos(limit: int = 20):
    if mongo_collection is None:
        return []
    try:
        items = []
        for d in mongo_collection.find({}, sort=[("createdAt", -1)]).limit(max(1, min(100, limit))):
            input_id = d.get("input", {}).get("id")
            output_id = d.get("output", {}).get("id")
            items.append({
                "id": str(d.get("_id")),
                "createdAt": d.get("createdAt").isoformat() if d.get("createdAt") else None,
                "input": {
                    "filename": d.get("input", {}).get("filename"),
                    "id": str(input_id) if input_id else None
                },
                "output": {
                    "filename": d.get("output", {}).get("filename"),
                    "id": str(output_id) if output_id else None
                },
                "alert": d.get("alert", False),
                "alert_reason": d.get("alert_reason"),
                "video_type": d.get("video_type", "Unknown")
            })
        return items
    except Exception:
        return []


@app.get("/download/metrics/{name}")
async def download_metrics(name: str):
    path = os.path.join(OUTPUT_DIR, name)
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path, media_type="application/json", filename=name)

@app.get("/video/{file_id}")
async def get_video(file_id: str, request: Request):
    # Just call your existing byid endpoint logic
    return await download_video_by_id(file_id, request)


@app.get("/")
async def root():
    return {"status": "ok"}


@app.get("/status")
async def status():
    return {
        "gridfs_connected": bool(grid_fs),
        "collection_connected": bool(mongo_collection),
        "db": MONGODB_DB,
    }

