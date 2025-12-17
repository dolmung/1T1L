from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, FileResponse
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

app = FastAPI()

# Ensure drawings directory exists
os.makedirs("static/drawings", exist_ok=True)

# Force Reload trigger
# ENV Setup
from dotenv import load_dotenv
load_dotenv()

# Serve static files (HTML, JS, CSS)
# app.mount("/static", StaticFiles(directory=".", html=True)) # Exposes root - bad
app.mount("/static", StaticFiles(directory="static", html=True))

@app.get("/debug/dump")
def debug_dump():
    # Return string representation of rooms keys and sample data
    summary = {}
    for k, v in rooms.items():
        summary[k] = {
            "slots_count": len(v['slots']),
            "filled": sum(1 for s in v['slots'].values() if s.is_filled),
            "sample_slot_2": v['slots'].get(2).dict() if 2 in v['slots'] else None,
            "host_message": v.get("host_message")
        }
    return summary

# Data Models
class JoinRequest(BaseModel):
    user_name: str
    message: str
    image_data: Optional[str] = None # PNG preview
    strokes: Optional[List[Dict[str, Any]]] = None # Stroke Data: [{type, color, width, points:[{x,y}]}]

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

import json

DB_FILE = "rooms.json"

def save_rooms():
    data = {}
    for rid, room in rooms.items():
        r_data = room.copy()
        # Convert slots dict {int: Slot} to dict {str: dict}
        slots_dict = {}
        for k, v in room['slots'].items():
            if hasattr(v, 'model_dump'): s_dump = v.model_dump()
            else: s_dump = v.dict()
            slots_dict[str(k)] = s_dump
        r_data['slots'] = slots_dict
        data[rid] = r_data
    
    try:
        with open(DB_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Save failed: {e}")

def load_rooms():
    if not os.path.exists(DB_FILE): return {}
    try:
        with open(DB_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        loaded_rooms = {}
        for rid, r_data in data.items():
            # Reconstruct slots
            slots_raw = r_data.get('slots', {})
            slots_obj = {}
            for k, v in slots_raw.items():
                slots_obj[int(k)] = Slot(**v)
            r_data['slots'] = slots_obj
            loaded_rooms[rid] = r_data
        return loaded_rooms
    except Exception as e:
        print(f"Load failed: {e}")
        return {}

# In-memory DB (Initialized from file)
rooms = load_rooms()

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
    room_id = str(random.randint(1000, 9999))
    
    # Clean up stale GIF if exists (to avoid showing old images)
    gif_path = f"result_{room_id}.gif"
    if os.path.exists(gif_path):
        try:
            os.remove(gif_path)
        except:
            pass

    host_name = data.get('host_name', 'Host')
    text = data.get('text', '') 
    columns = data.get('columns', 5)
    open_time = data.get('open_time')

    # Create slots from text
    slots = {}
    for i, char in enumerate(text):
        s = Slot(position=i, char=char)
        if char == ' ':
            s.is_filled = True
            s.reserved_by = "SYSTEM"
        slots[i] = s
    
    rooms[room_id] = {
        "host_name": host_name,
        "is_open": False,
        "open_time": open_time,
        "slots": slots,
        "columns": columns,
        "users": {},
        "final_card": None,
        "host_message": ""
    }
    save_rooms()
    return {"room_id": room_id}

@app.get("/status/{room_id}")
async def get_status(room_id: str):
    if room_id not in rooms: return {"error": "Room not found"}
    room = rooms[room_id]
    
    # Count filled slots (excluding SYSTEM reserved spaces if we want effectively filled by users)
    # But usually simple count is fine
    filled_count = sum(1 for s in room['slots'].values() if s.is_filled and s.reserved_by != "SYSTEM")
    total_slots = sum(1 for s in room['slots'].values() if s.reserved_by != "SYSTEM") 
    
    if total_slots == 0: total_slots = 1 # Prevent div/0

    slots_data = []
    for s in room['slots'].values():
        if hasattr(s, 'model_dump'):
            slots_data.append(s.model_dump())
        else:
            slots_data.append(s.dict())

    return {
        "host_name": room['host_name'],
        "is_open": room['is_open'],
        "open_time": room['open_time'],
        "filled_count": filled_count,
        "total_slots": total_slots,
        "columns": room.get('columns', 5),
        "slots": slots_data,
        "host_message": room.get('host_message')
    }

@app.post("/host-message/{room_id}")
async def save_host_message(room_id: str, data: dict):
    if room_id not in rooms: return {"error": "Room not found"}
    rooms[room_id]['host_message'] = data.get('message', '')
    save_rooms()
    return {"status": "SUCCESS"}

@app.post("/reserve/{room_id}")
async def reserve_slot(room_id: str, req: JoinRequest):
    if room_id not in rooms: return {"error": "Room not found"}
    room = rooms[room_id]
    
    # Time Check
    if room['is_open']:
        return {"status": "TIME_OVER", "message": "모집 시간이 마감되었습니다."}
    
    if room['open_time']:
         open_dt = datetime.fromisoformat(str(room['open_time']))
         if datetime.now() >= open_dt:
             room['is_open'] = True # Sync status
             return {"status": "TIME_OVER", "message": "모집 시간이 마감되었습니다."}

    # 1. Try to find empty Text Slot
    available_indices = [i for i, s in room['slots'].items() if not s.is_filled and not s.reserved_by]
    
    if available_indices:
        # User requested Random filling again
        slot_idx = random.choice(available_indices)
        slot = room['slots'][slot_idx]
        slot.reserved_by = req.user_name
        save_rooms()
        return {"status": "SUCCESS", "assigned_char": slot.char, "slot_idx": slot_idx}
    
    # 2. If Full, Add Special Slot
    specials = ["★", "♥", "♪", "♣", "♠", "●", "▲", "◆", "✿"]
    char = random.choice(specials)
    
    new_pos = len(room['slots'])
    new_slot = Slot(position=new_pos, char=char, hidden=True) # Hidden from Host Grid
    new_slot.reserved_by = req.user_name
    
    new_slot.reserved_by = req.user_name
    
    room['slots'][new_pos] = new_slot
    save_rooms()
    
    return {"status": "SUCCESS", "assigned_char": char, "slot_idx": new_pos}

@app.post("/join/{room_id}")
async def join_room(room_id: str, req: JoinRequest):
    if room_id not in rooms: return {"error": "Room not found"}
    room = rooms[room_id]
    
    # Time Check (Strict)
    if room['is_open']:
         return {"error": "Time is up", "status": "TIME_OVER"}
    if room['open_time']:
         open_dt = datetime.fromisoformat(str(room['open_time']))
         if datetime.now() >= open_dt:
             room['is_open'] = True
             return {"error": "Time is up", "status": "TIME_OVER"}
    
    # Find user's reserved slot
    target_slot = None
    for s in room['slots'].values():
        if s.reserved_by == req.user_name:
            target_slot = s
            break
    
    if target_slot:
        target_slot.user = req.user_name
        target_slot.message = req.message
        target_slot.is_filled = True
        target_slot.strokes = req.strokes
        
        # Save Preview Image if provided
        if req.image_data:
            try:
                # expected format: "data:image/png;base64,....."
                parts = req.image_data.split(",", 1)
                encoded = parts[1] if len(parts) > 1 else parts[0]
                data = base64.b64decode(encoded)
                filename = f"static/drawings/{room_id}_{target_slot.position}.png"
                with open(filename, "wb") as f:
                    f.write(data)
                target_slot.image_path = filename
            except Exception as e:
                print(f"Failed to save image: {e}")
        
        save_rooms()
        # Check if full? Not strictly needed for logic but good for status
        return {"status": "SUCCESS"}
    
    return {"error": "No reservation found"}

@app.get("/result_card/{room_id}")
def get_result_card(room_id: str):
    # If GIF exists, return it
    gif_path = f"result_{room_id}.gif"
    if os.path.exists(gif_path):
        return FileResponse(gif_path)
    return {"error": "Not generated yet"}

@app.post("/make-card/{room_id}")
def generate_card(room_id: str):
    if room_id not in rooms: return {"error": "Room not found"}
    room = rooms[room_id]
    
    # Grid Setup (Dynamic)
    cols = room.get('columns', 5)
    total_slots = len(room['slots'])
    rows = (total_slots + cols - 1) // cols
    
    slot_w = 240
    slot_h = 240
    margin = 30
    
    width = cols * slot_w + (cols+1) * margin
    height = rows * slot_h + (rows+1) * margin
    
    # Solid Brown Background
    bg_color = (161, 67, 67, 255) # #A14343
    
    # Common function to draw a single slot
    # Common function to draw a single slot
    def draw_slot(draw, x, y, slot, frame_idx):
        # Draw background for slot ONLY if it has content (not a space)
        if slot.char.strip():
             draw.rectangle([x, y, x+slot_w, y+slot_h], fill=(255,255,255))
        else:
             pass
        
        # Draw stroke if filled
        if slot.is_filled and slot.strokes:
            make_card_draw_strokes(draw, x, y, slot.strokes, slot_w, slot_h)
        else:
            # Draw Autofill Char for empty slots
            try:
                # Attempt to load a nice bold font
                try:
                    font = ImageFont.truetype("arialbd.ttf", 160) # Windows Bold
                except:
                    try:
                        font = ImageFont.truetype("arial.ttf", 160)
                    except:
                        font = ImageFont.load_default()

                # Center text
                # getbbox returns (left, top, right, bottom)
                bbox = draw.textbbox((0, 0), slot.char, font=font)
                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]
                
                text_x = x + (slot_w - w) / 2
                text_y = y + (slot_h - h) / 2 - 20 # Slight offset adjustment
                
                draw.text((text_x, text_y), slot.char, fill="#A14343", font=font)
                
            except Exception as e:
                print(f"Font Error: {e}")
                pass

    frames = []
    
    # Create Animation
    for frame_idx in range(6): # 6 frames animation
        img = Image.new('RGBA', (width, height), bg_color)
        draw = ImageDraw.Draw(img)
        
        for i in range(total_slots):
            # Safe access
            if i not in room['slots']: continue
            slot = room['slots'][i]
            
            r = i // cols
            c = i % cols
            
            sx = margin + c * (slot_w + margin)
            sy = margin + r * (slot_h + margin)
            
            draw_slot(draw, sx, sy, slot, frame_idx)
            
        frames.append(img)
        
    out_file = f"result_{room_id}.gif"
    frames[0].save(out_file, save_all=True, append_images=frames[1:], duration=120, loop=0, disposal=2)
    
    room['is_open'] = True
    room['open_time'] = datetime.now().isoformat()
    save_rooms()
    
    return {"status": "SUCCESS", "file": out_file}

@app.post("/host-message/{room_id}")
async def save_host_message(room_id: str, data: dict):
    if room_id not in rooms: return {"error": "Room not found"}
    room = rooms[room_id]
    room['host_message'] = data.get('message', '')
    return {"status": "SUCCESS"}

def make_card_draw_strokes(draw, ox, oy, strokes, w, h):
    # Scale: Client(340) -> Server(120)
    scale = w / 340.0
    
    for stroke in strokes:
        color = stroke.get('color', '#000000')
        width = stroke.get('width', 5)
        points = stroke.get('points', [])
        s_type = stroke.get('type', 'line')
        
        if not points: continue
        
        scaled_width = max(1, int(width * scale))
        
        render_points = []
        for p in points:
            # Jitter
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
            # Draw Heart using Text "♥" to match client HTML5 Canvas
            f_size = int(scaled_width * 3) # Make it large enough
            try:
                font = ImageFont.truetype("arial.ttf", f_size)
            except:
                try:
                    font = ImageFont.truetype("Ndot-45.ttf", f_size) # Try custom font first if available? No, stick to system
                except:
                    font = ImageFont.load_default()
            
            interval = 5
            for i in range(0, len(render_points), interval):
                cx, cy = render_points[i]
                try:
                    draw.text((cx, cy), "♥", font=font, fill=color, anchor="mm")
                except:
                    w = f_size * 0.8
                    h = f_size * 0.8
                    draw.text((cx - w/2, cy - h/2), "♥", font=font, fill=color)

        elif s_type == 'v-stitch':
            # Draw V at intervals
            interval = 3
            v_size = scaled_width
            for i in range(0, len(render_points), interval):
                cx, cy = render_points[i]
                
                # Draw V lines
                # Left stroke
                draw.line([(cx - v_size/2, cy - v_size/2), (cx, cy + v_size/2)], fill=color, width=max(1, int(scaled_width/2)))
                # Right stroke
                draw.line([(cx, cy + v_size/2), (cx + v_size/2, cy - v_size/2)], fill=color, width=max(1, int(scaled_width/2)))

        elif s_type == 'text':
            # Draw Text/Char at intervals
            char = stroke.get('char', '?')
            # Load Font (Try Arial for Windows)
            f_size = int(scaled_width * 2) 
            try:
                font = ImageFont.truetype("arial.ttf", f_size)
            except:
                font = ImageFont.load_default()
            
            # Draw less frequently for text to avoid overlap mess
            interval = 5 
            for i in range(0, len(render_points), interval):
                cx, cy = render_points[i]
                # Draw centered
                try:
                    # anchor mm = middle middle
                    draw.text((cx, cy), char, font=font, fill=color, anchor="mm")
                except:
                    # Fallback for older Pillow
                    w, h = draw.textsize(char, font=font)
                    draw.text((cx - w/2, cy - h/2), char, font=font, fill=color)

@app.get("/api/config")
def get_config():
    return {"kakao_key": os.getenv("KAKAO_JS_KEY")}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
