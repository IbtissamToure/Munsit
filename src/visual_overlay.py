import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image

COLORS = {
    "text_bg": (20, 20, 20),
    "sign_bg": (30, 0, 60),
    "audio_bg": (0, 40, 30),
    "text_color": (255, 255, 255),
    "sign_color": (180, 130, 255),
    "audio_color": (0, 220, 150),
    "border": (60, 60, 60)
}

AUDIO_ICONS = {
    "street_music": "MUSIC",
    "children_playing": "CROWD",
    "siren": "ALERT",
    "car_horn": "HORN",
    "dog_bark": "BARK",
    "": ""
}

def draw_rounded_rect(frame, x1, y1, x2, y2, color, radius=10, alpha=0.7):
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1 + radius, y1), (x2 - radius, y2), color, -1)
    cv2.rectangle(overlay, (x1, y1 + radius), (x2, y2 - radius), color, -1)
    cv2.circle(overlay, (x1 + radius, y1 + radius), radius, color, -1)
    cv2.circle(overlay, (x2 - radius, y1 + radius), radius, color, -1)
    cv2.circle(overlay, (x1 + radius, y2 - radius), radius, color, -1)
    cv2.circle(overlay, (x2 - radius, y2 - radius), radius, color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    return frame

def put_arabic_text(frame, text, position, font_size=32, color=(255, 255, 255)):
    if not text:
        return frame
    try:
        font = ImageFont.truetype(
            "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
            font_size
        )
    except:
        font = ImageFont.load_default()
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def render_overlay(frame, state):
    h, w = frame.shape[:2]

    # speech text panel - bottom
    if state.get("text"):
        text = state["text"]
        panel_h = 60
        draw_rounded_rect(frame, 10, h - panel_h - 10, w - 10, h - 10,
                         COLORS["text_bg"], alpha=0.8)
        frame = put_arabic_text(frame, text,
                                (20, h - panel_h),
                                font_size=28,
                                color=COLORS["text_color"])

    # sign panel - top right
    if state.get("sign"):
        sign = state["sign"]
        draw_rounded_rect(frame, w - 160, 10, w - 10, 80,
                         COLORS["sign_bg"], alpha=0.85)
        cv2.putText(frame, sign, (w - 145, 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                   COLORS["sign_color"], 2)
        cv2.putText(frame, "SIGN", (w - 145, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                   COLORS["sign_color"], 1)

    # audio event panel - top left
    audio_event = state.get("audio_event", "")
    if audio_event:
        icon = AUDIO_ICONS.get(audio_event, audio_event.upper())
        draw_rounded_rect(frame, 10, 10, 160, 80,
                         COLORS["audio_bg"], alpha=0.85)
        cv2.putText(frame, icon, (20, 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                   COLORS["audio_color"], 2)
        cv2.putText(frame, "SOUND", (20, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                   COLORS["audio_color"], 1)

    return frame

if __name__ == "__main__":
    cap = cv2.VideoCapture(1)
    print("Visual overlay test - Press Q to quit")

    test_state = {
        "text": "Hello",
        "sign": "ALIF",
        "audio_event": "street_music"
    }

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = render_overlay(frame, test_state)
        cv2.imshow("Munsit - Overlay Test", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()