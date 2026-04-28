import base64
import tkinter as tk

import cv2


def _ppm_photo_from_frame(frame) -> tk.PhotoImage:
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    height, width, _ = rgb_frame.shape
    ppm_header = f"P6\n{width} {height}\n255\n".encode("ascii")
    ppm_data = ppm_header + rgb_frame.tobytes()
    return tk.PhotoImage(data=base64.b64encode(ppm_data).decode("ascii"))


def _show_with_tkinter(cap: cv2.VideoCapture) -> None:
    root = tk.Tk()
    root.title("Webcam")

    label = tk.Label(root)
    label.pack()

    def close() -> None:
        cap.release()
        root.destroy()

    def update() -> None:
        ret, frame = cap.read()
        if not ret:
            root.after(10, update)
            return

        photo = _ppm_photo_from_frame(frame)
        label.configure(image=photo)
        label.image = photo
        root.after(15, update)

    root.protocol("WM_DELETE_WINDOW", close)
    root.bind("q", lambda _event: close())
    update()
    root.mainloop()


def main() -> None:
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame from webcam.")
                break

            try:
                cv2.imshow("Webcam", frame)
            except cv2.error as exc:
                if "The function is not implemented" in str(exc):
                    print("OpenCV GUI support is unavailable. Falling back to a Tkinter window.")
                    cv2.destroyAllWindows()
                    _show_with_tkinter(cap)
                    return
                raise

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
