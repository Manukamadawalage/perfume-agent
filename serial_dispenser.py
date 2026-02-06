# serial_dispenser.py
"""
Serial communication module for Arduino-based perfume dispenser
Sends valve number and dispense time to Arduino Mega
Arduino handles Cartesian positioning and valve control
"""

import serial
import json
import time
from pathlib import Path
from typing import List, Dict, Optional

class ArduinoDispenser:
    """
    Interface to Arduino Mega dispenser with X-Y Cartesian system

    Protocol: <valve_number,time_ms>
    Arduino handles: positioning → dispensing → acknowledgment
    """

    def __init__(self, port: str = 'COM13', baud: int = 9600, calibration_file: str = 'valve_calibration.json'):
        """
        Initialize serial connection to Arduino

        Args:
            port: Serial port (e.g., 'COM13' on Windows, '/dev/ttyACM0' on Linux)
            baud: Baud rate (must match Arduino)
            calibration_file: Path to flow rate calibration JSON
        """
        self.port = port
        self.baud = baud
        self.calibration_file = calibration_file
        self.ser = None
        self.calibration = {}

        # Connect to Arduino
        self._connect()

        # Load calibration data
        self._load_calibration()

    def _connect(self):
        """Establish serial connection and wait for Arduino ready signal"""
        try:
            print(f"Connecting to Arduino on {self.port}...")
            self.ser = serial.Serial(self.port, self.baud, timeout=10)
            time.sleep(2)  # Wait for Arduino reset

            # Wait for READY signal
            start_time = time.time()
            while time.time() - start_time < 10:  # 10 second timeout
                if self.ser.in_waiting:
                    line = self.ser.readline().decode().strip()
                    print(f"Arduino: {line}")
                    if line == "READY":
                        print("✓ Arduino connected and ready\n")
                        return

            raise TimeoutError("Arduino did not send READY signal")

        except serial.SerialException as e:
            raise ConnectionError(f"Failed to connect to Arduino on {self.port}: {e}")

    def _load_calibration(self):
        """Load flow rate calibration data from JSON file"""
        cal_path = Path(self.calibration_file)

        if not cal_path.exists():
            print(f"⚠ Warning: Calibration file '{self.calibration_file}' not found")
            print("Using default flow rate of 0.5 g/s for all valves")
            print("Run 'python calibration.py' to calibrate your system\n")
            self.calibration = {}
            return

        try:
            with open(cal_path, 'r') as f:
                self.calibration = json.load(f)
            print(f"✓ Loaded calibration data for {len(self.calibration)} valve/oil combinations\n")
        except Exception as e:
            print(f"⚠ Warning: Error loading calibration file: {e}")
            print("Using default flow rate of 0.5 g/s for all valves\n")
            self.calibration = {}

    def calculate_dispense_time(self, valve_id: int, oil_name: str, target_grams: float) -> int:
        """
        Calculate dispense time in milliseconds based on calibration

        Args:
            valve_id: Valve number (0-15)
            oil_name: Name of oil (for lookup in calibration)
            target_grams: Target weight in grams

        Returns:
            Dispense time in milliseconds
        """
        # Create calibration key (matches format in calibration.py)
        cal_key = f"valve_{valve_id}_{oil_name.replace(' ', '_').replace('(', '').replace(')', '').replace('%', '')}"

        # Get flow rate (g/s), default to 0.5 if not calibrated
        flow_rate = self.calibration.get(cal_key, 0.5)

        # Calculate time
        time_seconds = target_grams / flow_rate
        time_ms = int(time_seconds * 1000)

        return time_ms

    def dispense(self, valve_id: int, oil_name: str, target_grams: float, timeout: int = 180) -> bool:
        """
        Dispense a specific amount from a valve

        Args:
            valve_id: Valve number (0-15)
            oil_name: Oil name (for display and calibration lookup)
            target_grams: Target weight in grams
            timeout: Maximum time to wait for completion (seconds)

        Returns:
            True if successful, False otherwise
        """
        # Calculate dispense time
        time_ms = self.calculate_dispense_time(valve_id, oil_name, target_grams)

        # Safety check
        if time_ms > 60000:
            print(f"  ⚠ ERROR: Dispense time too long ({time_ms}ms)")
            print(f"  Check calibration for valve {valve_id} / {oil_name}")
            return False

        if time_ms < 100:
            print(f"  ⚠ WARNING: Dispense time very short ({time_ms}ms)")

        # Send command to Arduino
        command = f"<{valve_id},{time_ms}>\n"
        print(f"→ Valve {valve_id} ({oil_name}): {target_grams}g → {time_ms}ms")

        try:
            self.ser.write(command.encode())

            # Monitor Arduino responses
            start_time = time.time()
            while time.time() - start_time < timeout:
                if self.ser.in_waiting:
                    response = self.ser.readline().decode().strip()

                    if not response:
                        continue

                    print(f"  {response}")

                    if response.startswith("DONE"):
                        # Parse: DONE,valve,time_ms
                        parts = response.split(',')
                        if len(parts) == 3:
                            print(f"  ✓ Complete\n")
                            return True
                        else:
                            print(f"  ✓ Complete (unexpected format)\n")
                            return True

                    elif response.startswith("ERROR"):
                        print(f"  ✗ Arduino error\n")
                        return False

            # Timeout
            print(f"  ✗ Timeout waiting for Arduino response\n")
            return False

        except Exception as e:
            print(f"  ✗ Communication error: {e}\n")
            return False

    def dispense_recipe(self, dispense_plan: List[Dict]) -> Dict:
        """
        Execute a complete recipe dispense plan

        Args:
            dispense_plan: List of dicts with keys: valve, oil_name, ml, grams
                          (output from logic_dispense.to_dispense_plan)

        Returns:
            Summary dict with success status and statistics
        """
        print("\n" + "="*60)
        print(f"STARTING RECIPE: {len(dispense_plan)} ingredients")
        print("="*60 + "\n")

        total_steps = len(dispense_plan)
        completed = 0
        failed = 0
        skipped = 0

        start_time = time.time()

        for step_num, item in enumerate(dispense_plan, 1):
            valve = item['valve']
            oil_name = item['oil_name']
            grams = item['grams']
            ml = item.get('ml', 0)

            print(f"[Step {step_num}/{total_steps}] {oil_name}")
            print(f"  Target: {grams}g ({ml}ml)")

            # Check if valve is assigned
            if valve == "UNASSIGNED" or not isinstance(valve, int):
                print(f"  ⚠ Skipped - no valve assigned\n")
                skipped += 1
                continue

            # Validate valve number
            if valve < 0 or valve > 15:
                print(f"  ⚠ Skipped - invalid valve number {valve}\n")
                skipped += 1
                continue

            # Dispense
            success = self.dispense(valve, oil_name, grams)

            if success:
                completed += 1
            else:
                failed += 1
                # Ask user if they want to continue
                retry = input("  Retry this step? (y/n): ").lower()
                if retry == 'y':
                    if self.dispense(valve, oil_name, grams):
                        completed += 1
                        failed -= 1
                    else:
                        cont = input("  Continue with recipe? (y/n): ").lower()
                        if cont != 'y':
                            print("\n✗ Recipe aborted by user\n")
                            break

            # Brief pause between steps
            time.sleep(0.5)

        # Summary
        elapsed = time.time() - start_time

        print("="*60)
        print("RECIPE COMPLETE")
        print("="*60)
        print(f"Total steps:     {total_steps}")
        print(f"Completed:       {completed}")
        print(f"Failed:          {failed}")
        print(f"Skipped:         {skipped}")
        print(f"Time elapsed:    {elapsed:.1f}s")
        print("="*60 + "\n")

        return {
            "success": failed == 0,
            "total_steps": total_steps,
            "completed": completed,
            "failed": failed,
            "skipped": skipped,
            "elapsed_seconds": elapsed
        }

    def home(self) -> bool:
        """Send home command to Arduino"""
        print("Homing Cartesian system...")
        try:
            self.ser.write(b"HOME\n")
            time.sleep(5)  # Wait for homing to complete

            while self.ser.in_waiting:
                response = self.ser.readline().decode().strip()
                print(f"  {response}")
                if "HOMED" in response or "READY" in response:
                    return True

            return True
        except Exception as e:
            print(f"  Error: {e}")
            return False

    def get_status(self) -> Optional[Dict]:
        """Query Arduino for current position status"""
        try:
            self.ser.write(b"STATUS\n")
            time.sleep(0.2)

            if self.ser.in_waiting:
                response = self.ser.readline().decode().strip()
                if response.startswith("POS,"):
                    # Parse: POS,x,y
                    parts = response.split(',')
                    if len(parts) == 3:
                        return {
                            "x": int(parts[1]),
                            "y": int(parts[2])
                        }
            return None
        except Exception as e:
            print(f"Error getting status: {e}")
            return None

    def close(self):
        """Close serial connection"""
        if self.ser and self.ser.is_open:
            print("Closing Arduino connection...")
            self.ser.close()
            print("✓ Disconnected\n")


# Convenience function for quick testing
def test_dispense(valve: int, grams: float, port: str = 'COM3'):
    """Quick test function to dispense from a single valve"""
    dispenser = ArduinoDispenser(port=port)

    try:
        dispenser.dispense(valve, f"Test_Valve_{valve}", grams)
    finally:
        dispenser.close()


if __name__ == "__main__":
    # Example usage
    print("Arduino Dispenser - Test Mode")
    print("-" * 60)

    # Test with sample dispense plan
    sample_plan = [
        {"valve": 2, "oil_name": "Lavender", "ml": 1.5, "grams": 1.35},
        {"valve": 0, "oil_name": "Bergamot", "ml": 1.0, "grams": 0.88},
        {"valve": 15, "oil_name": "Alcohol (Ethanol 96%)", "ml": 50, "grams": 39.5}
    ]

    try:
        dispenser = ArduinoDispenser(port='COM3')
        dispenser.dispense_recipe(sample_plan)
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'dispenser' in locals():
            dispenser.close()
