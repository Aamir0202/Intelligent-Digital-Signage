# # -----------------------------------------------------------------------------------
# # Updated LaserGaze/main.py
# # -----------------------------------------------------------------------------------
from .GazeProcessor import GazeProcessor
from .VisualizationOptions import VisualizationOptions
import asyncio


async def main(callback):
    vo = VisualizationOptions()
    gp = GazeProcessor(visualization_options=vo, callback=callback)
    await gp.start()


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(main())
    finally:
        loop.close()
