from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, Dict, List, Any
import uvicorn
import os
import random
from PIL import Image, ImageDraw, ImageFont
import io 
from datetime import datetime
import base64
import json

# Firebase Imports
import firebase_admin
from firebase_admin import credentials, firestore, storage

# ENV Setup
from dotenv import load_dotenv
load_dotenv()

app = FastAPI()

# Ensure drawings directory exists (Local fallback)
# os.makedirs("static/drawings", exist_ok=True) # REMOVED: Causes 500 on Vercel (Read-only root)

# --- Firebase Initialization ---
# 1. Try to get credentials from ENV (Vercel / Local .env)
firebase_creds = os.getenv("FIREBASE_CREDENTIALS")

if not firebase_admin._apps:
    try:
        if firebase_creds:
            # Parse JSON string from ENV
            if firebase_creds.startswith("{"):
                cred_dict = json.loads(firebase_creds)
                cred = credentials.Certificate(cred_dict)
            else:
                # Assuming it's a path if not JSON string (fallback)
                cred = credentials.Certificate(firebase_creds)
        else:
            # Fallback for local development if standard file exists
            if os.path.exists("serviceAccountKey.json"):
                cred = credentials.Certificate("serviceAccountKey.json")
            else:
                print("WARNING: No Firebase Credentials found. Set FIREBASE_CREDENTIALS env var.")
                cred = None # will crash if used

        if cred:
             # Initialize without storage bucket first if not needed immediately, 
             # or add 'storageBucket': 'YOUR-BUCKET.appspot.com' if using Storage.
             # For now, we use Firestore only for data.
             firebase_admin.initialize_app(cred)
             db = firestore.client()
             print("Firebase Initialized Successfully")
        else:
             db = None
             print("Firebase NOT Initialized")

    except Exception as e:
        print(f"Firebase Init Error: {e}")
        db = None

# --- Helpers for Firestore ---
def get_room_doc(room_id):
    if db:
        return db.collection('rooms').document(room_id)
    return None

def room_to_dict(doc_snapshot):
    if not doc_snapshot.exists: return None
    data = doc_snapshot.to_dict()
    # Convert sub-collections or handle structure changes here if needed
    # Flatten structure for frontend compatibility if necessary
    return data

# Serve static files (HTML, JS, CSS)
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static", html=True))
else:
    print("WARNING: static directory not found. Static files disabled.")

@app.get("/debug/dump")
def debug_dump():
    if not db: return {"error": "No Database"}
    # Only list first 5 for debug
    docs = db.collection('rooms').limit(5).stream()
    summary = {}
    for doc in docs:
        d = doc.to_dict()
        summary[doc.id] = {
            "host_name": d.get("host_name"),
            "slots_count": len(d.get("slots", {}))
        }
    return summary

# Data Models
class JoinRequest(BaseModel):
    user_name: str
    message: str
    image_data: Optional[str] = None # PNG preview
    strokes: Optional[List[Dict[str, Any]]] = None # Stroke Data
    
class Slot(BaseModel):
    position: int
    char: str
    user: Optional[str] = None
    message: Optional[str] = None
    is_filled: bool = False
    reserved_by: Optional[str] = None
    strokes: Optional[List[Dict[str, Any]]] = None
    image_path: Optional[str] = None
    hidden: bool = False

# Alphabet
ALPHABET = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

@app.get("/")
async def read_index():
    return FileResponse('index.html')

@app.get("/create")
async def read_create():
    return FileResponse('create.html')

@app.get("/host")
async def read_host():
    return FileResponse('create.html')

@app.post("/create_room")
async def create_room(data: dict):
    if not db: return {"error": "Database not connected"}
    
    room_id = str(random.randint(1000, 9999))
    
    # Clean up stale GIF if exists (Local Only / Tmp)
    try:
        gif_path = f"/tmp/result_{room_id}.gif" if os.getenv("VERCEL") else f"result_{room_id}.gif"
        if os.path.exists(gif_path):
            os.remove(gif_path)
    except: pass

    host_name = data.get('host_name', 'Host')
    text = data.get('text', '') 
    columns = data.get('columns', 5)
    open_time = data.get('open_time')

    # Create slots
    slots = {}
    for i, char in enumerate(text):
        s = Slot(position=i, char=char)
        if char == ' ':
            s.is_filled = True
            s.reserved_by = "SYSTEM"
        
        if hasattr(s, 'model_dump'): s_dump = s.model_dump()
        else: s_dump = s.dict()
        slots[str(i)] = s_dump # Firestore keys must be strings
    
    room_data = {
        "host_name": host_name,
        "is_open": False,
        "open_time": open_time,
        "slots": slots,
        "columns": columns,
        "users": {},
        "final_card": None,
        "host_message": "",
        "created_at": firestore.SERVER_TIMESTAMP
    }
    
    db.collection('rooms').document(room_id).set(room_data)
    return {"room_id": room_id}

@app.get("/status/{room_id}")
async def get_status(room_id: str):
    if not db: return {"error": "Database not connected"}
    
    doc_ref = db.collection('rooms').document(room_id)
    doc = doc_ref.get()
    
    if not doc.exists: return {"error": "Room not found"}
    room = doc.to_dict()
    
    slots = room.get('slots', {})
    
    # Calculate counts
    filled_count = 0
    total_slots = 0
    
    # Convert Dict[str, dict] back to formatted list for frontend
    slots_list = []
    # We need to sort by position because Firestore map order isn't guaranteed
    # But keys are "0", "1", ...
    sorted_keys = sorted(slots.keys(), key=lambda x: int(x))
    
    for k in sorted_keys:
        v = slots[k]
        if v.get('reserved_by') != "SYSTEM":
             total_slots += 1
             if v.get('is_filled'):
                 filled_count += 1
        slots_list.append(v)

    if total_slots == 0: total_slots = 1
    
    return {
        "host_name": room.get('host_name'),
        "is_open": room.get('is_open'),
        "open_time": room.get('open_time'),
        "filled_count": filled_count,
        "total_slots": total_slots,
        "columns": room.get('columns', 5),
        "slots": slots_list,
        "host_message": room.get('host_message')
    }

@app.post("/host-message/{room_id}")
async def save_host_message(room_id: str, data: dict):
    if not db: return {"error": "DB Error"}
    db.collection('rooms').document(room_id).update({
        "host_message": data.get('message', '')
    })
    return {"status": "SUCCESS"}

@app.post("/reserve/{room_id}")
async def reserve_slot(room_id: str, req: JoinRequest):
    if not db: return {"error": "DB Error"}

    # Use transaction to prevent race conditions
    # For simplicity in this demo, we use simple updates but check manually
    
    doc_ref = db.collection('rooms').document(room_id)
    doc = doc_ref.get()
    if not doc.exists: return {"error": "Room not found"}
    room = doc.to_dict()
    
    if room.get('is_open'):
        return {"status": "TIME_OVER", "message": "모집 시간이 마감되었습니다."}
    
    if room.get('open_time'):
         try:
             open_dt = datetime.fromisoformat(str(room['open_time']))
             if datetime.now() >= open_dt:
                 doc_ref.update({"is_open": True})
                 return {"status": "TIME_OVER", "message": "모집 시간이 마감되었습니다."}
         except: pass

    slots = room.get('slots', {})
    
    # 1. Try to find empty Text Slot
    # Sort keys to be deterministic
    sorted_keys = sorted(slots.keys(), key=lambda x: int(x))
    available_indices = []
    
    for k in sorted_keys:
        s = slots[k]
        if not s.get('is_filled') and not s.get('reserved_by'):
             available_indices.append(k)
    
    if available_indices:
        slot_idx = random.choice(available_indices) # Choose RANDOM as per original logic
        # Update Firestore
        # Construct update path: "slots.0.reserved_by"
        update_path = {f"slots.{slot_idx}.reserved_by": req.user_name}
        doc_ref.update(update_path)
        
        return {"status": "SUCCESS", "assigned_char": slots[slot_idx]['char'], "slot_idx": int(slot_idx)}
    
    # 2. If Full, Add Special Slot
    specials = ["★", "♥", "♪", "♣", "♠", "●", "▲", "◆", "✿"]
    char = random.choice(specials)
    
    new_pos = len(slots)
    new_slot = Slot(position=new_pos, char=char, hidden=True, reserved_by=req.user_name)
    
    if hasattr(new_slot, 'model_dump'): s_dump = new_slot.model_dump()
    else: s_dump = new_slot.dict()
    
    update_path = {f"slots.{new_pos}": s_dump}
    doc_ref.update(update_path)
    
    return {"status": "SUCCESS", "assigned_char": char, "slot_idx": new_pos}

@app.post("/join/{room_id}")
async def join_room(room_id: str, req: JoinRequest):
    if not db: return {"error": "DB Error"}
    
    doc_ref = db.collection('rooms').document(room_id)
    doc = doc_ref.get()
    if not doc.exists: return {"error": "Room not found"}
    room = doc.to_dict()
    
    # Find user's slot
    slots = room.get('slots', {})
    target_k = None
    
    for k, v in slots.items():
        if v.get('reserved_by') == req.user_name:
            target_k = k
            break
            
    if target_k:
        # Save strokes directly to Firestore
        updates = {
            f"slots.{target_k}.user": req.user_name,
            f"slots.{target_k}.message": req.message,
            f"slots.{target_k}.is_filled": True,
            f"slots.{target_k}.strokes": req.strokes
        }
        
        # We assume image saving is not critical or relying on Vercel temp fs for now
        # Ideally we upload image to Storage here if needed.
        # But stroke data is enough for the final GIF card.
        
        doc_ref.update(updates)
        return {"status": "SUCCESS"}
    
    return {"error": "No reservation found"}

@app.get("/result_card/{room_id}")
def get_result_card(room_id: str):
    # This remains file-based for local cache, which is tricky on Vercel.
    # On Vercel, this endpoint won't find the file created by another request (serverless).
    # Ideally, returns a cloud storage URL.
    # For MVP: We return local file IF it exists (generated in same instance), else error
    # Or simpler: Front-end triggers generation, back-end returns Base64 of GIF? (Might be heavy)
    
    # Generate path based on environment
    gif_path = f"/tmp/result_{room_id}.gif" if os.getenv("VERCEL") else f"result_{room_id}.gif"
    
    if os.path.exists(gif_path):
        return FileResponse(gif_path)
    return {"error": "Not generated yet or file lost (Serverless)"}

@app.post("/make-card/{room_id}")
def generate_card(room_id: str):
    if not db: return {"error": "DB Error"}
    
    doc_ref = db.collection('rooms').document(room_id)
    doc = doc_ref.get()
    if not doc.exists: return {"error": "Room not found"}
    room = doc.to_dict()
    
    # ... Drawing Logic (Redundant with old code, copying logic but using room data) ...
    # Grid Setup (Dynamic)
    cols = room.get('columns', 5)
    slots = room.get('slots', {})
    total_slots = len(slots)
    rows = (total_slots + cols - 1) // cols
    
    slot_w = 240
    slot_h = 240
    margin = 30
    
    width = cols * slot_w + (cols+1) * margin
    height = rows * slot_h + (rows+1) * margin
    
    bg_color = (161, 67, 67, 255) # #A14343
    
    def draw_slot(draw, x, y, slot, frame_idx):
        if slot.get('char', '').strip():
             draw.rectangle([x, y, x+slot_w, y+slot_h], fill=(255,255,255))
        
        if slot.get('is_filled') and slot.get('strokes'):
            make_card_draw_strokes(draw, x, y, slot['strokes'], slot_w, slot_h)
        else:
            # Draw Text Char
            try:
                try: 
                    # Use bundled font if available (for Vercel)
                    if os.path.exists("NanumGothic-Bold.ttf"):
                        font = ImageFont.truetype("NanumGothic-Bold.ttf", 160)
                    else:
                        font = ImageFont.truetype("arialbd.ttf", 160)
                except: 
                    font = ImageFont.load_default()
                
                bbox = draw.textbbox((0, 0), slot['char'], font=font)
                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]
                text_x = x + (slot_w - w) / 2
                text_y = y + (slot_h - h) / 2 - 20
                draw.text((text_x, text_y), slot['char'], fill="#A14343", font=font)
            except: pass

    frames = []
    for frame_idx in range(6): 
        img = Image.new('RGBA', (width, height), bg_color)
        draw = ImageDraw.Draw(img)
        
        # Iterate by index 0..N
        for i in range(total_slots):
            k = str(i)
            if k not in slots: continue
            slot = slots[k]
            
            r = i // cols
            c = i % cols
            
            sx = margin + c * (slot_w + margin)
            sy = margin + r * (slot_h + margin)
            
            draw_slot(draw, sx, sy, slot, frame_idx)
            
        frames.append(img)
        
    out_file = f"/tmp/result_{room_id}.gif" if os.getenv("VERCEL") else f"result_{room_id}.gif"
    frames[0].save(out_file, save_all=True, append_images=frames[1:], duration=120, loop=0, disposal=2)
    
    # Update Open Time to close room
    doc_ref.update({
        "is_open": True, 
        "open_time": datetime.now().isoformat()
    })
    
    return {"status": "SUCCESS", "file": out_file} # Note: File is local temp

# Include helper function make_card_draw_strokes (copied from original)
def make_card_draw_strokes(draw, ox, oy, strokes, w, h):
    scale = (w / 340.0) * 1.5 # Boost stroke thickness by 1.5x
    for stroke in strokes:
        color = stroke.get('color', '#000000')
        width = stroke.get('width', 5)
        points = stroke.get('points', [])
        s_type = stroke.get('type', 'line')
        
        if not points: continue
        scaled_width = max(1, int(width * scale))
        render_points = []
        for p in points:
            jx = (random.random()-0.5)*5
            jy = (random.random()-0.5)*5
            px = ox + (p['x'] * scale) + jx
            py = oy + (p['y'] * scale) + jy
            render_points.append((px, py))
            
        if s_type == 'line':
            if len(render_points) > 1:
                draw.line(render_points, fill=color, width=scaled_width, joint='curve')
            elif len(render_points) == 1:
                x, y = render_points[0]
                r = scaled_width/2
                draw.ellipse((x-r, y-r, x+r, y+r), fill=color)
        elif s_type == 'heart':
             # Simplified copy of original logic...
             f_size = int(scaled_width * 3)
             try: 
                 if os.path.exists("NanumGothic-Bold.ttf"):
                    font = ImageFont.truetype("NanumGothic-Bold.ttf", f_size)
                 else:
                    font = ImageFont.truetype("arial.ttf", f_size)
             except: font = ImageFont.load_default()
             draw.text(render_points[0], "♥", font=font, fill=color)
        elif s_type == 'v-stitch':
             # Simplified
             pass
        elif s_type == 'text':
             char = stroke.get('char', '?')
             draw.text(render_points[0], char, fill=color)

@app.get("/api/config")
def get_config():
    return {"kakao_key": os.getenv("KAKAO_JS_KEY")}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
