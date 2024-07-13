import tkinter as tk
import serial
import time

# Replace  with the appropriate serial port for your Arduino.
SERIAL_PORT = 'COM9'
BAUD_RATE = 9600

class ArduinoControlApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Arduino Control App")

        # Initialize Arduino connection
        try:
            self.serial = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
            time.sleep(2)  # Allow time for the connection to be established
        except serial.SerialException:
            print(f"Failed to connect to Arduino on port {SERIAL_PORT}. Check the connection.")
            self.root.destroy()
            return

        # Initialize pin states
        self.pin_states = [0, 0, 0, 0]

        # Create toggle buttons
        self.buttons = []
        for i in range(4):
            button = tk.Button(self.root, text=f"Pin {i+2}", width=10, height=2, command=lambda idx=i: self.toggle_pin(idx))
            button.grid(row=0, column=i)
            self.buttons.append(button)

        # Timer variables
        self.start_time = 0
        self.timer_running = False

        self.root.mainloop()

    def toggle_pin(self, pin_idx):
        pin_state = 1 - self.pin_states[pin_idx]  # Toggle the state
        self.pin_states[pin_idx] = pin_state
        self.serial.write(f"{pin_state}{pin_state}{pin_state}{pin_state}".encode())

        if pin_state == 1:
            self.start_time = time.time()
            self.timer_running = True
            self.root.after(100, self.update_timer)

    def update_timer(self):
        if self.timer_running:
            elapsed_time = time.time() - self.start_time
            print(f"Timer: {elapsed_time:.2f} seconds")
            self.root.after(100, self.update_timer)
        else:
            print("Timer stopped.")

    def stop_timers(self):
        self.timer_running = False

    def __del__(self):
        self.stop_timers()
        if hasattr(self, 'serial'):
            self.serial.close()

if __name__ == "__main__":
    root = tk.Tk()
    app = ArduinoControlApp(root)