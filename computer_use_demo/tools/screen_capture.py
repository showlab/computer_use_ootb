import subprocess
import base64
from pathlib import Path
from PIL import ImageGrab
from uuid import uuid4
from screeninfo import get_monitors
import platform
if platform.system() == "Darwin":
    import Quartz  # uncomment this line if you are on macOS
    
from PIL import ImageGrab
from functools import partial
from .base import BaseAnthropicTool, ToolError, ToolResult


OUTPUT_DIR = "./tmp/outputs"

def get_screenshot(selected_screen: int = 0, resize: bool = True, target_width: int = 1920, target_height: int = 1080):
        # print(f"get_screenshot selected_screen: {selected_screen}")
        
        # Get screen width and height using Windows command
        display_num = None
        offset_x = 0
        offset_y = 0
        selected_screen = selected_screen   
        width, height = _get_screen_size()    
    
        """Take a screenshot of the current screen and return a ToolResult with the base64 encoded image."""
        output_dir = Path(OUTPUT_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"screenshot_{uuid4().hex}.png"

        ImageGrab.grab = partial(ImageGrab.grab, all_screens=True)

        # Detect platform
        system = platform.system()

        if system == "Windows":
            # Windows: Use screeninfo to get monitor details
            screens = get_monitors()

            # Sort screens by x position to arrange from left to right
            sorted_screens = sorted(screens, key=lambda s: s.x)

            if selected_screen < 0 or selected_screen >= len(screens):
                raise IndexError("Invalid screen index.")

            screen = sorted_screens[selected_screen]
            bbox = (screen.x, screen.y, screen.x + screen.width, screen.y + screen.height)

        elif system == "Darwin":  # macOS
            # macOS: Use Quartz to get monitor details
            max_displays = 32  # Maximum number of displays to handle
            active_displays = Quartz.CGGetActiveDisplayList(max_displays, None, None)[1]

            # Get the display bounds (resolution) for each active display
            screens = []
            for display_id in active_displays:
                bounds = Quartz.CGDisplayBounds(display_id)
                screens.append({
                    'id': display_id,
                    'x': int(bounds.origin.x),
                    'y': int(bounds.origin.y),
                    'width': int(bounds.size.width),
                    'height': int(bounds.size.height),
                    'is_primary': Quartz.CGDisplayIsMain(display_id)  # Check if this is the primary display
                })

            # Sort screens by x position to arrange from left to right
            sorted_screens = sorted(screens, key=lambda s: s['x'])
            # print(f"Darwin sorted_screens: {sorted_screens}")

            if selected_screen < 0 or selected_screen >= len(screens):
                raise IndexError("Invalid screen index.")

            screen = sorted_screens[selected_screen]
            
            bbox = (screen['x'], screen['y'], screen['x'] + screen['width'], screen['y'] + screen['height'])

        else:  # Linux or other OS
            cmd = "xrandr | grep ' primary' | awk '{print $4}'"
            try:
                output = subprocess.check_output(cmd, shell=True).decode()
                resolution = output.strip()
                # Parse the resolution format like "1920x1080+1920+0"
                parts = resolution.split('+')[0]  # Get just the "1920x1080" part
                width, height = map(int, parts.split('x'))
                
                # Create a screen object/dictionary similar to what's used in other platforms
                # Extract the offsets from the resolution string
                x_offset = int(resolution.split('+')[1]) if len(resolution.split('+')) > 1 else 0
                y_offset = int(resolution.split('+')[2]) if len(resolution.split('+')) > 2 else 0
                
                # Create a screen object with attributes similar to the screeninfo.Monitor object
                class LinuxScreen:
                    def __init__(self, x, y, width, height):
                        self.x = x
                        self.y = y
                        self.width = width
                        self.height = height
                
                screen = LinuxScreen(x_offset, y_offset, width, height)
                bbox = (x_offset, y_offset, x_offset + width, y_offset + height)
            except subprocess.CalledProcessError:
                raise RuntimeError("Failed to get screen resolution on Linux.")

        # Take screenshot using the bounding box
        screenshot = ImageGrab.grab(bbox=bbox)

        # Set offsets (for potential future use)
        offset_x = screen['x'] if system == "Darwin" else screen.x
        offset_y = screen['y'] if system == "Darwin" else screen.y

        # # Resize if 
        if resize:
            screenshot = screenshot.resize((target_width, target_height))

        # Save the screenshot
        screenshot.save(str(path))

        if path.exists():
            # Return a ToolResult instance instead of a dictionary
            return screenshot, path
        
        raise ToolError(f"Failed to take screenshot: {path} does not exist.")
    
    


def _get_screen_size(selected_screen: int = 0):
    if platform.system() == "Windows":
        # Use screeninfo to get primary monitor on Windows
        screens = get_monitors()

        # Sort screens by x position to arrange from left to right
        sorted_screens = sorted(screens, key=lambda s: s.x)
        if selected_screen is None:
            primary_monitor = next((m for m in get_monitors() if m.is_primary), None)
            return primary_monitor.width, primary_monitor.height
        elif selected_screen < 0 or selected_screen >= len(screens):
            raise IndexError("Invalid screen index.")
        else:
            screen = sorted_screens[selected_screen]
            return screen.width, screen.height
    elif platform.system() == "Darwin":
        # macOS part using Quartz to get screen information
        max_displays = 32  # Maximum number of displays to handle
        active_displays = Quartz.CGGetActiveDisplayList(max_displays, None, None)[1]

        # Get the display bounds (resolution) for each active display
        screens = []
        for display_id in active_displays:
            bounds = Quartz.CGDisplayBounds(display_id)
            screens.append({
                'id': display_id,
                'x': int(bounds.origin.x),
                'y': int(bounds.origin.y),
                'width': int(bounds.size.width),
                'height': int(bounds.size.height),
                'is_primary': Quartz.CGDisplayIsMain(display_id)  # Check if this is the primary display
            })

        # Sort screens by x position to arrange from left to right
        sorted_screens = sorted(screens, key=lambda s: s['x'])

        if selected_screen is None:
            # Find the primary monitor
            primary_monitor = next((screen for screen in screens if screen['is_primary']), None)
            if primary_monitor:
                return primary_monitor['width'], primary_monitor['height']
            else:
                raise RuntimeError("No primary monitor found.")
        elif selected_screen < 0 or selected_screen >= len(screens):
            raise IndexError("Invalid screen index.")
        else:
            # Return the resolution of the selected screen
            screen = sorted_screens[selected_screen]
            return screen['width'], screen['height']

    else:  # Linux or other OS
        cmd = "xrandr | grep ' primary' | awk '{print $4}'"
        try:
            output = subprocess.check_output(cmd, shell=True).decode()
            resolution = output.strip().split()[0]
            parts = resolution.split('+')[0]  # Get just the "1920x1080" part
            width, height = map(int, parts.split('x'))
            return width, height
        except subprocess.CalledProcessError:
            raise RuntimeError("Failed to get screen resolution on Linux.")
