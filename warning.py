import time
import win32gui, win32api, win32con

red = win32api.RGB(255, 0, 0)  # Red
green = win32api.RGB(0, 255, 0)  # Green
black = win32api.RGB(0, 0, 0)  # black


def draw(color, width):
    """Draw a rectangular frame around the screen"""
    w, h = win32api.GetSystemMetrics(0), win32api.GetSystemMetrics(1)
    dc = win32gui.GetDC(0)
    pen = win32gui.CreatePen(win32con.PS_SOLID, width, color)
    win32gui.SelectObject(dc, pen)
    m = width // 2 - 1
    win32gui.Polyline(dc, [(m, m), (m, h - m), (w - m, h - m), (w - m, m), (m, m)])


if __name__ == '__main__':
    start = time.time()
    while time.time() - start < 1:  # show frame for one seconds
        draw(red, 12)
    draw(green, 12)
