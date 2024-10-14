import threading
import time
from pynput import keyboard
from functools import lru_cache

class Employee():
    name = ''
    isStop = False

    def __init__(self, name):
        self.name = name
        self.isStop = False

employees = [Employee('Van Anh'), Employee('Minh')]
listen = keyboard.Listener

@lru_cache(maxsize=None)
def showEmployee(employee, sleepTime):
    while True:
        if employee.isStop:
            print(f'{employee.name} is terminated')
            break
        print(employee.name)
        time.sleep(sleepTime)
    totalRunning = any(x.isStop==False for x in employees)
    print("Runing Thread: ", totalRunning)
    if totalRunning == False:
        print("All Thread is terminated")
        listen.stop()

def on_press(key):
    vk = key.vk if hasattr(key, 'vk') else key.value.vk
    if vk == None:
        return
    index = vk - 48 # index of 0 is 48
    if index < len(employees) and index >=0 and employees[index].isStop==False:
        employees[index].isStop = True
        print(f'Terminating employee {employees[index].name}')


if __name__ == "__main__":
    for employee in employees:
        p = threading.Thread(target=showEmployee, args=(employee, 2))
        p.start()

    with keyboard.Listener(on_press=on_press) as listener:
        listen = listener
        listener.join()
