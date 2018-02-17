import pyautogui
x = 100
y = 100
pyautogui.moveTo(x,y)
for i in range(100):
    pyautogui.moveRel(10,10)
