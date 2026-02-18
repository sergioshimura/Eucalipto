import lgpio
import time
import sys

VALVE_PIN = 17

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 gpio_handler.py <on|off>")
        return

    command = sys.argv[1]

    h = None
    try:
        h = lgpio.gpiochip_open(4)
        lgpio.gpio_claim_output(h, VALVE_PIN)
    except Exception as e:
        try:
            h = lgpio.gpiochip_open(0)
            lgpio.gpio_claim_output(h, VALVE_PIN)
        except Exception as e_inner:
            print(f"Error: Could not open GPIO: {e_inner}")
            return

    if command == "on":
        lgpio.gpio_write(h, VALVE_PIN, 1)
        time.sleep(0.2)
        lgpio.gpio_write(h, VALVE_PIN, 0)
    elif command == "off":
        lgpio.gpio_write(h, VALVE_PIN, 0)

    if h:
        lgpio.gpiochip_close(h)

if __name__ == "__main__":
    main()
