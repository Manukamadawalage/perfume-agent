# calibration.py
"""
Calibration tool for perfume dispenser valve flow rates
Measures how many grams per second each valve dispenses for each oil
"""

import serial
import json
import time
from pathlib import Path

def calibrate_valve(ser: serial.Serial, valve_id: int, oil_name: str, test_duration_sec: int = 10) -> float:
    """
    Calibrate flow rate for a single valve/oil combination

    Args:
        ser: Active serial connection to Arduino
        valve_id: Valve number (0-15)
        oil_name: Name of oil being calibrated
        test_duration_sec: How long to dispense for measurement (default 10s)

    Returns:
        Flow rate in grams per second
    """
    print("\n" + "="*60)
    print(f"CALIBRATING: Valve {valve_id} - {oil_name}")
    print("="*60)
    print(f"\nPreparation:")
    print(f"  1. Ensure valve {valve_id} reservoir contains {oil_name}")
    print(f"  2. Place empty collection container under dispense point")
    print(f"  3. Place container on scale and tare to 0.00g")
    print(f"\nThe system will dispense for {test_duration_sec} seconds")

    input("\nPress ENTER when ready to start...")

    # Send dispense command
    test_time_ms = test_duration_sec * 1000
    command = f"<{valve_id},{test_time_ms}>\n"

    print(f"\n→ Opening valve {valve_id} for {test_duration_sec} seconds...")
    print("  (Monitoring Arduino responses...)\n")

    ser.write(command.encode())

    # Monitor Arduino during dispensing
    start_time = time.time()
    done = False

    while not done and (time.time() - start_time) < (test_duration_sec + 30):
        if ser.in_waiting:
            response = ser.readline().decode().strip()
            if response:
                print(f"  Arduino: {response}")

                if response.startswith("DONE"):
                    done = True
                    break
                elif response.startswith("ERROR"):
                    print("\n✗ Arduino reported error during calibration")
                    return 0.0

    if not done:
        print("\n⚠ Warning: Arduino did not confirm completion")

    # Get measurement from user
    print(f"\n" + "-"*60)
    while True:
        try:
            dispensed_grams = float(input("Enter weight on scale (grams): "))
            if dispensed_grams <= 0:
                print("  Weight must be positive. Try again.")
                continue
            break
        except ValueError:
            print("  Invalid input. Enter a number (e.g., 8.45)")

    # Calculate flow rate
    flow_rate = dispensed_grams / test_duration_sec

    print(f"\n✓ Flow rate calculated: {flow_rate:.3f} g/s")
    print(f"  ({dispensed_grams}g in {test_duration_sec}s)")
    print("="*60)

    return flow_rate


def run_full_calibration(port: str = 'COM3', baud: int = 9600):
    """
    Run calibration wizard for all valves

    Args:
        port: Serial port for Arduino
        baud: Baud rate (must match Arduino)
    """
    # Load valve map to know which oils are on which valves
    valve_map_file = Path('valve_map.json')

    if not valve_map_file.exists():
        print("Error: valve_map.json not found")
        print("Please ensure valve_map.json exists in the current directory")
        return

    with open(valve_map_file, 'r') as f:
        valve_map = json.load(f)

    # Invert map: valve_id -> oil_name
    valve_to_oil = {valve_id: oil_name for oil_name, valve_id in valve_map.items()}

    print("\n" + "="*60)
    print("PERFUME DISPENSER CALIBRATION")
    print("="*60)
    print(f"\nFound {len(valve_to_oil)} valve assignments in valve_map.json")
    print("\nThis calibration measures flow rate (g/s) for each valve.")
    print("You'll need:")
    print("  - Digital scale (0.01g precision recommended)")
    print("  - Collection containers")
    print("  - All oils loaded in their assigned valves")

    input("\nPress ENTER to start calibration...")

    # Connect to Arduino
    try:
        print(f"\nConnecting to Arduino on {port}...")
        ser = serial.Serial(port, baud, timeout=10)
        time.sleep(2)

        # Wait for READY
        start_time = time.time()
        while time.time() - start_time < 10:
            if ser.in_waiting:
                line = ser.readline().decode().strip()
                print(f"Arduino: {line}")
                if line == "READY":
                    break

        print("✓ Arduino connected\n")

    except serial.SerialException as e:
        print(f"\n✗ Error: Could not connect to Arduino on {port}")
        print(f"Details: {e}")
        return

    # Calibration data storage
    calibration_data = {}

    # Calibrate each valve
    valve_ids = sorted(valve_to_oil.keys())

    for i, valve_id in enumerate(valve_ids, 1):
        oil_name = valve_to_oil[valve_id]

        print(f"\n[{i}/{len(valve_ids)}] Valve {valve_id}: {oil_name}")

        # Ask if user wants to calibrate this valve
        choice = input("Calibrate this valve? (y/n/skip all): ").lower()

        if choice == 'skip all':
            print("\nSkipping remaining valves...")
            break
        elif choice != 'y':
            print(f"Skipped valve {valve_id}")
            continue

        # Run calibration
        flow_rate = calibrate_valve(ser, valve_id, oil_name, test_duration_sec=10)

        if flow_rate > 0:
            # Store with key format: valve_X_OilName
            cal_key = f"valve_{valve_id}_{oil_name.replace(' ', '_').replace('(', '').replace(')', '').replace('%', '')}"
            calibration_data[cal_key] = round(flow_rate, 3)

        # Brief pause between calibrations
        time.sleep(1)

    # Close serial connection
    ser.close()

    # Save calibration data
    if calibration_data:
        output_file = 'valve_calibration.json'

        # Load existing calibration if exists
        if Path(output_file).exists():
            with open(output_file, 'r') as f:
                existing_data = json.load(f)
            # Merge (new data overwrites old)
            existing_data.update(calibration_data)
            calibration_data = existing_data

        with open(output_file, 'w') as f:
            json.dump(calibration_data, f, indent=2)

        print("\n" + "="*60)
        print("CALIBRATION COMPLETE")
        print("="*60)
        print(f"✓ Saved {len(calibration_data)} calibration values to {output_file}")
        print("\nCalibrated valves:")
        for key, flow_rate in sorted(calibration_data.items()):
            print(f"  {key}: {flow_rate} g/s")
        print("\n" + "="*60)
    else:
        print("\n⚠ No calibration data collected")


def calibrate_single_valve():
    """Interactive tool to calibrate just one valve"""
    print("\n" + "="*60)
    print("SINGLE VALVE CALIBRATION")
    print("="*60)

    # Get valve info from user
    try:
        valve_id = int(input("\nEnter valve number (0-15): "))
        if valve_id < 0 or valve_id > 15:
            print("Invalid valve number")
            return
    except ValueError:
        print("Invalid input")
        return

    oil_name = input("Enter oil name: ").strip()
    if not oil_name:
        print("Oil name required")
        return

    port = input("Serial port (default COM3): ").strip() or 'COM3'

    # Connect to Arduino
    try:
        print(f"\nConnecting to {port}...")
        ser = serial.Serial(port, 9600, timeout=10)
        time.sleep(2)

        # Wait for READY
        start_time = time.time()
        while time.time() - start_time < 10:
            if ser.in_waiting:
                line = ser.readline().decode().strip()
                print(f"Arduino: {line}")
                if line == "READY":
                    break

    except serial.SerialException as e:
        print(f"Error connecting: {e}")
        return

    # Run calibration
    flow_rate = calibrate_valve(ser, valve_id, oil_name)

    ser.close()

    if flow_rate > 0:
        # Save to file
        cal_key = f"valve_{valve_id}_{oil_name.replace(' ', '_').replace('(', '').replace(')', '').replace('%', '')}"

        output_file = 'valve_calibration.json'
        if Path(output_file).exists():
            with open(output_file, 'r') as f:
                data = json.load(f)
        else:
            data = {}

        data[cal_key] = round(flow_rate, 3)

        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"\n✓ Saved to {output_file}")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("PERFUME DISPENSER CALIBRATION TOOL")
    print("="*60)
    print("\nOptions:")
    print("  1. Full calibration (all valves from valve_map.json)")
    print("  2. Single valve calibration")
    print("  3. Exit")

    choice = input("\nSelect option (1-3): ").strip()

    if choice == '1':
        port = input("Serial port (default COM3): ").strip() or 'COM3'
        run_full_calibration(port=port)
    elif choice == '2':
        calibrate_single_valve()
    else:
        print("Exiting...")
