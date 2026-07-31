import torch
from PIL import Image, ImageDraw
from transformers import Sam3Model, Sam3Processor

HERE = "/home/hound/CMU-VLN-Challenge-2026/experiments/sam3"

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

model = Sam3Model.from_pretrained("facebook/sam3").to(device)
processor = Sam3Processor.from_pretrained("facebook/sam3")

image = Image.open(f"{HERE}/japanese_room.png").convert("RGB")

def detect(prompt, threshold=0.4):
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    results = processor.post_process_instance_segmentation(
        outputs, threshold=threshold, mask_threshold=0.5,
        target_sizes=inputs.get("original_sizes").tolist()
    )[0]
    return results

paintings = detect("calligraphy painting")
ledge = detect("wooden display platform")

print(f"\ncalligraphy painting instances found: {len(paintings['masks'])}")
for i, box in enumerate(paintings["boxes"]):
    print(f"  painting {i}: box(x0,y0,x1,y1)={box.tolist()}, score={paintings['scores'][i].item():.3f}")

print(f"\ndisplay platform instances found: {len(ledge['masks'])}")
for i, box in enumerate(ledge["boxes"]):
    print(f"  ledge {i}: box(x0,y0,x1,y1)={box.tolist()}, score={ledge['scores'][i].item():.3f}")

if len(ledge["boxes"]) == 0:
    print("\nNo ledge/platform detected — cannot apply spatial filter.")
else:
    # take highest-scoring ledge instance, use its top edge (y0) as the reference line
    best_ledge_idx = int(torch.argmax(ledge["scores"]))
    ledge_top_y = ledge["boxes"][best_ledge_idx][1].item()
    print(f"\nUsing ledge top edge y = {ledge_top_y:.1f}")

    count_above = 0
    draw_img = image.copy()
    draw = ImageDraw.Draw(draw_img)
    for i, box in enumerate(paintings["boxes"]):
        x0, y0, x1, y1 = box.tolist()
        painting_bottom_y = y1  # bottom edge of the painting
        is_above = painting_bottom_y < ledge_top_y
        color = "lime" if is_above else "red"
        draw.rectangle([x0, y0, x1, y1], outline=color, width=4)
        if is_above:
            count_above += 1

    lx0, ly0, lx1, ly1 = ledge["boxes"][best_ledge_idx].tolist()
    draw.rectangle([lx0, ly0, lx1, ly1], outline="cyan", width=4)

    out_path = f"{HERE}/sam3_result.png"
    draw_img.save(out_path)
    print(f"\nANSWER: {count_above} calligraphy painting(s) above the display ledge")
    print(f"Visualization saved to {out_path}")
