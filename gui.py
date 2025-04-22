import numpy as np
import cv2
import dearpygui.dearpygui as dpg

# === CANVAS SETTINGS ===
canvas_width = 800
canvas_height = 800
scale = 10  # pixels per meter
danger_range = 30  # metri entro cui il pericolo è massimo

# === WORLD-TO-CANVAS MAPPING ===
def world_to_canvas(x, y):
    canvas_x = x * scale
    canvas_y = canvas_height - y * scale  # Y invertito
    return canvas_x, canvas_y

# === PLACEHOLDERS ===
points = np.empty((0, 3))
detections = []  # ogni detection: [x, y, w, h, vx, vy]
img1 = np.zeros((200, 300, 3), dtype=np.uint8)
img2 = np.zeros((200, 300, 3), dtype=np.uint8)
frame_count = 0  # Contatore del frame

# === TEXTURE HANDLING ===
def np_to_dpg_texture(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
    img = img.astype(np.float32) / 255.0
    return img.flatten()

def update_textures(img1_new, img2_new):
    dpg.set_value("img1_tex", np_to_dpg_texture(img1_new))
    dpg.set_value("img2_tex", np_to_dpg_texture(img2_new))

# === DRAW UTILS ===
def draw_axes():
    dpg.draw_line(world_to_canvas(0, 0), world_to_canvas(canvas_width/scale, 0), color=[255, 0, 0, 255], thickness=2, parent="BEV_drawlist")
    dpg.draw_text(world_to_canvas(canvas_width/scale - 2, 2), "X", color=[255, 0, 0, 255], size=18, parent="BEV_drawlist")
    dpg.draw_line(world_to_canvas(0, 0), world_to_canvas(0, canvas_height/scale), color=[0, 255, 0, 255], thickness=2, parent="BEV_drawlist")
    dpg.draw_text(world_to_canvas(2, canvas_height/scale - 2), "Y", color=[0, 255, 0, 255], size=18, parent="BEV_drawlist")

def draw_detection(x, y, w, h, vx, vy):
    distance = np.sqrt(x**2 + y**2)
    danger = max(0.0, 1 - distance / danger_range)
    color = [int(255 * danger), int(255 * (1 - danger)), 0, 255]

    # Box
    x0, y0 = world_to_canvas(x - w/2, y - h/2)
    x1, y1 = world_to_canvas(x + w/2, y + h/2)
    dpg.draw_rectangle([x0, y0], [x1, y1], color=color, thickness=2, parent="BEV_drawlist")
    dpg.draw_text([x0+5, y0+5], "Person", size=15, color=color, parent="BEV_drawlist")

    # Freccia della velocità
    cx, cy = world_to_canvas(x, y)
    vx_canvas = vx * scale
    vy_canvas = -vy * scale  # invertito Y
    dpg.draw_arrow([cx, cy], [cx + vx_canvas, cy + vy_canvas], color=[0, 150, 255, 255], thickness=2, size=5, parent="BEV_drawlist")
    speed = np.sqrt(vx**2 + vy**2)
    dpg.draw_text([cx + vx_canvas + 5, cy + vy_canvas], f"{speed:.1f} m/s", size=14, color=[0, 150, 255, 255], parent="BEV_drawlist")

    # Barra del pericolo
    bar_x = x1 + 10
    bar_top, bar_bottom = y0, y1
    danger_height = (y1 - y0) * danger
    dpg.draw_rectangle([bar_x, bar_top], [bar_x + 10, bar_bottom], color=[100, 100, 100, 255], thickness=1, parent="BEV_drawlist")
    dpg.draw_rectangle([bar_x, bar_bottom - danger_height], [bar_x + 10, bar_bottom], fill=color, parent="BEV_drawlist")

# === DRAW BEV ===
def draw_bev():
    dpg.delete_item("BEV_drawlist", children_only=True)
    draw_axes()

    # Punti LiDAR
    for pt in points:
        cx, cy = world_to_canvas(pt[0], pt[1])
        dpg.draw_circle([cx, cy], radius=1.5, color=[100, 100, 100, 150], fill=[150, 150, 150, 80], parent="BEV_drawlist")

    # Detections
    for det in detections:
        x, y, w, h, vx, vy = det  # Estrai anche vx e vy dalla detection
        draw_detection(x, y, w, h, vx, vy)  # Passa tutte le informazioni alla funzione


# === MOCK DATA GENERATOR ===
def generate_mock_data():
    global frame_count
    frame_count += 1

    new_points = np.random.uniform(0, 50, (1000, 3))  # Punti LiDAR simulati
    new_detections = [
        [np.random.uniform(5, 45), np.random.uniform(5, 45), 2 + np.random.rand()*2, 4 + np.random.rand()*2,
         np.random.uniform(-2, 2), np.random.uniform(-2, 2)]  # vx, vy
        for _ in range(np.random.randint(1, 4))  # Rilevamenti simulati
    ]
    
    # Crea nuove immagini per simulare il cambiamento
    new_img1 = np.zeros((200, 300, 3), dtype=np.uint8)
    new_img2 = np.zeros((200, 300, 3), dtype=np.uint8)

    # Aggiungi forme randomiche nelle immagini
    cv2.rectangle(new_img1, (50, 50), (250, 150), (0, np.random.randint(0,255), 0), -1)  # Rettangolo verde per simularlo
    cv2.putText(new_img1, f"Cam 1 - Frame {frame_count}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    cv2.circle(new_img2, (150, 100), 50, (0, 0, np.random.randint(0,255)), -1)  # Cerchio rosso per Cam 2
    cv2.putText(new_img2, f"Cam 2 - Frame {frame_count}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    return new_points, new_detections, new_img1, new_img2  # Ritorna le immagini aggiornate



# === UPDATE VIEW ===
def update_view():
    global points, detections, img1, img2
    points, detections, img1, img2 = generate_mock_data()
    update_textures(img1, img2)
    draw_bev()

# === GUI SETUP ===
dpg.create_context()

with dpg.texture_registry():
    dpg.add_static_texture(300, 200, np_to_dpg_texture(img1), tag="img1_tex")
    dpg.add_static_texture(300, 200, np_to_dpg_texture(img2), tag="img2_tex")

with dpg.window(label="LiDAR BEV Viewer", width=canvas_width+400, height=canvas_height+150):
    dpg.add_button(label="Next Frame", callback=update_view)
    dpg.add_text("BEV from LiDAR with Detections", bullet=True)

    # Pannello per le immagini delle camere sopra la BEV
    with dpg.child_window(width=canvas_width, height=200):
        dpg.add_text("Camera Views:")
        dpg.add_image("img1_tex")
        dpg.add_spacing(count=1)
        dpg.add_image("img2_tex")

    # Disegna la BEV sotto le immagini delle camere
    with dpg.group(horizontal=True):
        dpg.add_drawlist(width=canvas_width, height=canvas_height, tag="BEV_drawlist")

update_view()  # inizializza

dpg.create_viewport(title='BEV Viewer', width=canvas_width+420, height=canvas_height+380)
dpg.setup_dearpygui()
dpg.show_viewport()
dpg.start_dearpygui()
dpg.destroy_context()
