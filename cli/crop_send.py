# cropping given img to 512x512
import argparse
from PIL import Image

def crop_resize_512(in_path: str, out_path: str):
    # open the img
    img = Image.open(in_path).convert("RGB")
    w, h = img.size
    side = min(w, h)
    img = img.crop(((w-side)//2, (h-side)//2, (w+side)//2, (h+side)//2))
    img = img.resize((512, 512), Image.LANCZOS)
    img.save(out_path, "PNG")
    # print(f"Successfully saved 512x512 image to {out_path}")

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("-image", required=True, help="Path to input image")
#     parser.add_argument("-out", default="out.png", help="Path to save 512x512 image")
#     args = parser.parse_args()

#     # input_image = "data/joy_cooke.jpg"
#     # output_image = "data/joy_cooke_512.png"
#     # crop_resize_512(input_image, output_image)

#     crop_resize_512(args.image, args.out)
    

# if __name__ == "__main__":
#     main()
