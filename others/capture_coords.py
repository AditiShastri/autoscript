import cv2
import json
from pdf2image import convert_from_path
import numpy as np

PDF_PATH = "student_sample.pdf"
OUTPUT_JSON = "box_coords.json"
DPI = 200
TARGET_WIDTH = 800  # Force fit to 800px wide so it always fits on screen

pages = convert_from_path(PDF_PATH, dpi=DPI, fmt="png")
orig_img = cv2.cvtColor(np.array(pages[0]), cv2.COLOR_RGB2BGR)

# Calculate scale factor dynamically
scale_factor = TARGET_WIDTH / orig_img.shape[1]
display_img = cv2.resize(orig_img, None, fx=scale_factor, fy=scale_factor)

coords_map = {}
clicks = []
q_counter = 1
step = "number"

def click_event(event, x, y, flags, param):
    global clicks, q_counter, step, coords_map, scale_factor

    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"📍 Clicked: {x}, {y} (display coords)")
        clicks.append((x, y))

        if len(clicks) == 2:
            scaled = [(int(px / scale_factor), int(py / scale_factor)) for px, py in clicks]

            if q_counter not in coords_map:
                coords_map[q_counter] = {}
            coords_map[q_counter][step] = scaled

            print(f"✅ Saved Q{q_counter} - {step}: {scaled}")
            with open(OUTPUT_JSON, "w") as f:
                json.dump(coords_map, f, indent=4)
            print(f"💾 Saved to {OUTPUT_JSON}")

            clicks.clear()

            if step == "number":
                step = "answer"
            else:
                step = "number"
                q_counter += 1

            if q_counter > 3:
                cv2.destroyAllWindows()
                print("✅ Finished capturing Q1–Q3. Exiting...")
                exit(0)

cv2.namedWindow("Select Boxes", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Select Boxes", TARGET_WIDTH, int(display_img.shape[0]))
cv2.setMouseCallback("Select Boxes", click_event)

print("📌 Instructions:")
print("Click TOP-LEFT then BOTTOM-RIGHT for:")
print("Q1 number → Q1 answer → Q2 number → Q2 answer → Q3 number → Q3 answer\n")

while True:
    cv2.imshow("Select Boxes", display_img)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()
