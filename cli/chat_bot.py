#chat bot
import shlex
from crop_send import crop_resize_512

BANNER = """
  /image PATH    — crop/resize an image to 512x512 and set it as current
  /help          — show help instructions
  /quit          — exit

Set the img -> type a prompt -> it will be send back with that img
"""

def main():
    print(BANNER)
    current_image = None
    while True:
        try:
            line = input("prompt> ").strip()
            current_image = line
        except (EOFError, KeyboardInterrupt):
            print("\nExiting")
            break

        if line == "":
            continue

        if line in ("/quit", "/q", "/exit"):
            print("Exiting")
            break

        if line == "/help":
            print(BANNER)
            continue

        if line.startswith("/image"):
            try:
                parts = shlex.split(line)
                path = parts[1]
                new_label = path.split('.')[0]
                resized = crop_resize_512(path, new_label + '_out.png')
                current_image = resized
                print(f"[image] prepared file: {resized}")
            except Exception as e:
                print(f"[error] {e}")
            continue

        if not current_image:
            print("[warn] Set an image first: /image PATH")
            continue

        try:
            answer = run_langchain(line, current_image)
            print(f"[AI] {answer}\n")
        except Exception as e:
            print(f"[error] {e}")

        print(f"[send] prompt: {line}")
        print(f"[send] image:  {current_image}")

if __name__ == "__main__":
    main()